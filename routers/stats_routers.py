"""接口调用统计查询路由。

裸 APIRouter（非 BaseRouter）：无图片上传、无检测器，仅查询账本。
RouterRegistry._collect_routers_from_module 会自动发现模块级 APIRouter
实例并挂到 /api/v1 前缀。
"""

import re
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from schemas.exceptions import InvalidParamsError
from services.call_stats import call_stats_recorder

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_SYS_KEY = "mobile_vision"
_SYS_NAME = "移动视觉检测服务"

# detector_type → 中文功能名（与 router_registry.py tag_map 保持一致）
_FUN_NAME_MAP = {
    "panel_label": "线标OCR检测",
    "dc_fuse": "直流熔丝检测",
    "indicator_light": "指示灯检测",
    "lap_surf": "搭接面检测",
    "plate_screw": "铁片螺丝检测",
    "line_squeeze": "线序检测",
}

stats_router = APIRouter()


def _build_response(stats: dict) -> dict:
    """将内部聚合结构转换为对外响应格式。

    每条 info 记录对应一个 (scene, date) 的统计行，
    date 格式为 YYYYMMDD，数值字段均转为字符串。
    """
    info = []
    for scene, scene_stats in stats["scenes"].items():
        fun_name = _FUN_NAME_MAP.get(scene, scene)
        for day_row in scene_stats["daily"]:
            info.append({
                "funKey": scene,
                "funName": fun_name,
                "date": day_row["date"].replace("-", ""),
                "summary": str(day_row["total"]),
                "ok": str(day_row["ok"]),
                "ng": str(day_row["ng"]),
                "error": str(day_row["error"]),
            })
    return {"sysKey": _SYS_KEY, "sysName": _SYS_NAME, "info": info}


@stats_router.get(
    "/stats",
    summary="接口调用统计查询",
    description="按场景(detector_type)统计各检测接口的调用次数，"
    "维度为 场景 × 日期 × 结果(ok/ng/error)，支持场景与日期范围（闭区间）过滤",
)
async def get_call_stats(
    scene: Optional[str] = Query(None, description="按场景(detector_type)过滤，如 panel_label"),
    start_date: Optional[str] = Query(None, description="起始日期 YYYY-MM-DD（含当天）"),
    end_date: Optional[str] = Query(None, description="截止日期 YYYY-MM-DD（含当天）"),
):
    for name, value in (("start_date", start_date), ("end_date", end_date)):
        if value is not None and not _DATE_RE.match(value):
            raise InvalidParamsError(f"{name} 格式非法，应为 YYYY-MM-DD")
    if start_date and end_date and start_date > end_date:
        raise InvalidParamsError("start_date 不应晚于 end_date")
    stats = await run_in_threadpool(
        call_stats_recorder.query, scene=scene, start_date=start_date, end_date=end_date
    )
    return _build_response(stats)
