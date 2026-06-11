"""接口调用统计：SQLite 账本，按 场景 × 日期 × 结果(ok/ng/error) 计数。

落盘在 {DATA_DIR}/stats.db（与数据回流同根目录，随卷挂载持久化，重启不丢）。
WAL + busy_timeout + 单条 UPSERT 原子累加，多 worker / 多线程并发安全；
每次操作用短连接（本服务 QPS 为工业质检级别，开销可忽略），
避免跨线程共享连接的正确性问题。
"""

import os
import sqlite3
from datetime import date
from typing import Optional

from config import settings
from utils import vision_logger

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_call_stats (
    scene   TEXT NOT NULL,
    date    TEXT NOT NULL,
    verdict TEXT NOT NULL,
    count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scene, date, verdict)
)
"""

_UPSERT_SQL = """
INSERT INTO api_call_stats (scene, date, verdict, count) VALUES (?, ?, ?, 1)
ON CONFLICT(scene, date, verdict) DO UPDATE SET count = count + 1
"""

_VERDICTS = ("ok", "ng", "error")


class CallStatsRecorder:
    """接口调用统计账本。record 写入一笔，query 按条件聚合。"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(_CREATE_TABLE_SQL)
        return conn

    def record(self, scene: str, verdict: str, day: Optional[str] = None) -> None:
        """记一笔调用。day 缺省取服务器本地日期（与回流目录 date_dir 一致）。

        verdict 边界校验保证账本里 total == ok+ng+error 恒成立；
        非法值抛 ValueError，经 record_call 入口时被吞掉并 warning。
        """
        if verdict not in _VERDICTS:
            raise ValueError(f"未知 verdict: {verdict!r}，应为 {_VERDICTS} 之一")
        day = day or date.today().isoformat()
        conn = self._connect()
        try:
            conn.execute(_UPSERT_SQL, (scene, day, verdict))
            conn.commit()
        finally:
            conn.close()

    def query(
        self,
        scene: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """按条件聚合查询，日期为闭区间，daily 按日期升序。"""
        result: dict = {"total": 0, "scenes": {}}
        if not os.path.exists(self.db_path):
            return result

        sql = "SELECT scene, date, verdict, count FROM api_call_stats"
        conds, params = [], []
        if scene:
            conds.append("scene = ?")
            params.append(scene)
        if start_date:
            conds.append("date >= ?")
            params.append(start_date)
        if end_date:
            conds.append("date <= ?")
            params.append(end_date)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY scene, date"

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        for scene_name, day, verdict, count in rows:
            # record() 已做边界校验，此处再防一手历史脏行：未知 verdict 整行跳过，
            # 保证各层 total == ok+ng+error 恒成立
            if verdict not in _VERDICTS:
                continue
            scene_stats = result["scenes"].setdefault(
                scene_name,
                {"total": 0, "verdicts": {v: 0 for v in _VERDICTS}, "daily": []},
            )
            daily = scene_stats["daily"]
            if not daily or daily[-1]["date"] != day:
                daily.append({"date": day, "ok": 0, "ng": 0, "error": 0, "total": 0})
            day_row = daily[-1]
            day_row[verdict] += count
            day_row["total"] += count
            scene_stats["verdicts"][verdict] += count
            scene_stats["total"] += count
            result["total"] += count
        return result


# 账本落在 DATA_DIR 下（与数据回流同根，cwd 锚定，生产随 ./data 卷挂载持久化）
call_stats_recorder = CallStatsRecorder(
    os.path.join(os.path.abspath(settings.DATA_DIR), "stats.db")
)


def record_call(scene: str, verdict: str) -> None:
    """埋点入口：吞掉一切异常，统计失败绝不影响检测主流程。"""
    try:
        call_stats_recorder.record(scene, verdict)
    except Exception as e:
        vision_logger.warning(f"调用统计写入失败 scene={scene} verdict={verdict}: {e}")
