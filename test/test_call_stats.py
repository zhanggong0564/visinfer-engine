"""接口调用统计单元测试：Recorder 账本、base_router 埋点、查询端点。"""
import asyncio
from datetime import date

import numpy as np
import pytest
from fastapi import BackgroundTasks

from routers.base_router import BaseRouter
from schemas.exceptions import InvalidParamsError
from services.call_stats import CallStatsRecorder, record_call


@pytest.fixture
def recorder(tmp_path):
    return CallStatsRecorder(str(tmp_path / "stats.db"))


class TestCallStatsRecorder:
    def test_record_accumulates(self, recorder):
        """同键累加，不同 verdict 互不影响，聚合结构完整。"""
        for _ in range(3):
            recorder.record("panel_label", "ok", day="2026-06-10")
        recorder.record("panel_label", "ng", day="2026-06-10")
        result = recorder.query()
        assert result["total"] == 4
        scene = result["scenes"]["panel_label"]
        assert scene["total"] == 4
        assert scene["verdicts"] == {"ok": 3, "ng": 1, "error": 0}
        assert scene["daily"] == [
            {"date": "2026-06-10", "ok": 3, "ng": 1, "error": 0, "total": 4}
        ]

    def test_record_default_day_is_today(self, recorder):
        """不传 day 时按服务器本地日期入账。"""
        recorder.record("panel_label", "ok")
        daily = recorder.query()["scenes"]["panel_label"]["daily"]
        assert daily[0]["date"] == date.today().isoformat()

    def test_scene_filter(self, recorder):
        recorder.record("panel_label", "ok", day="2026-06-01")
        recorder.record("plate_screw", "error", day="2026-06-01")
        result = recorder.query(scene="panel_label")
        assert set(result["scenes"]) == {"panel_label"}
        assert result["total"] == 1

    def test_date_range_filter_inclusive(self, recorder):
        """start_date / end_date 闭区间。"""
        recorder.record("panel_label", "ok", day="2026-06-01")
        recorder.record("panel_label", "ok", day="2026-06-05")
        recorder.record("panel_label", "ok", day="2026-06-08")
        result = recorder.query(start_date="2026-06-05", end_date="2026-06-08")
        days = [d["date"] for d in result["scenes"]["panel_label"]["daily"]]
        assert days == ["2026-06-05", "2026-06-08"]
        assert result["total"] == 2

    def test_daily_sorted_ascending(self, recorder):
        recorder.record("panel_label", "ok", day="2026-06-05")
        recorder.record("panel_label", "ok", day="2026-06-01")
        days = [d["date"] for d in recorder.query()["scenes"]["panel_label"]["daily"]]
        assert days == ["2026-06-01", "2026-06-05"]

    def test_query_empty_when_db_missing(self, recorder):
        """库文件尚未创建（零调用）时返回空结构，不报错。"""
        assert recorder.query() == {"total": 0, "scenes": {}}

    def test_record_rejects_unknown_verdict(self, recorder):
        """verdict 边界校验：非法值拒绝入账，保证 total == ok+ng+error 恒成立。"""
        with pytest.raises(ValueError):
            recorder.record("panel_label", "timeout")
        assert recorder.query() == {"total": 0, "scenes": {}}


class TestRecordCall:
    def test_record_call_swallows_exceptions(self, monkeypatch):
        """埋点入口必须吞掉一切异常，统计失败不能影响检测主流程。"""
        import services.call_stats as cs

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(cs.call_stats_recorder, "record", _raise)
        record_call("panel_label", "ok")  # 不应抛出

    def test_record_call_swallows_unknown_verdict(self):
        """非法 verdict 经埋点入口同样被吞掉（record 在建连前已拒绝，不落盘）。"""
        record_call("panel_label", "timeout")  # 不应抛出


class _FakeUpload:
    """最小 UploadFile 替身：只需 filename + async read。"""

    def __init__(self, filename="线标检验FU211-1779526099406.jpg"):
        self.filename = filename

    async def read(self):
        return b"fake-image-bytes"


class _Router(BaseRouter):
    def __init__(self):
        super().__init__(
            router_name="t_router",
            api_path="/t_detect",
            summary="t",
            description="t",
            detector_type="panel_label",
        )

    def request_schema(self, json_dict):
        return json_dict

    def get_inputs(self, request_params, image):
        return None


def _make_router(monkeypatch, detect_side_effect, patch_record_call=True):
    """构造测试路由：跳过解码/落盘，默认捕获 record_call 调用。

    patch_record_call=False 时保留真实 record_call，用于验证
    统计底层故障不影响检测主流程的集成契约。
    """
    router = _Router()

    async def _fake_process_image(file):
        return np.zeros((10, 10, 3), dtype=np.uint8), False

    monkeypatch.setattr(router, "_process_image", _fake_process_image)

    class _Detector:
        def detect(self, inputs):
            return detect_side_effect()

    monkeypatch.setattr(router, "get_detector_singleton", lambda: _Detector())
    monkeypatch.setattr(router, "_persist_record", lambda **kw: None)

    calls = []
    if patch_record_call:
        monkeypatch.setattr(
            "routers.base_router.record_call",
            lambda scene, verdict: calls.append((scene, verdict)),
        )
    return router, calls


_OK_RESULT = {"detailList": [], "status": "true", "error_msg": "", "message": "检测成功"}
_NG_RESULT = {"detailList": [], "status": "false", "error_msg": "", "message": "检测到异常"}


class TestCallStatsHook:
    def test_success_ok_recorded(self, monkeypatch):
        """status=true 的成功响应记一笔 ok（经 background task）。"""
        router, calls = _make_router(monkeypatch, lambda: dict(_OK_RESULT))
        bg = BackgroundTasks()
        asyncio.run(
            router._handle_request(
                background_tasks=bg,
                file=_FakeUpload(),
                json_data='{"modelParams": {"product_type": "FU211"}}',
            )
        )
        asyncio.run(bg())  # 显式执行后台任务
        assert calls == [("panel_label", "ok")]

    def test_success_ng_recorded(self, monkeypatch):
        """status=false 记一笔 ng。"""
        router, calls = _make_router(monkeypatch, lambda: dict(_NG_RESULT))
        bg = BackgroundTasks()
        asyncio.run(
            router._handle_request(
                background_tasks=bg,
                file=_FakeUpload(),
                json_data='{"modelParams": {"product_type": "FU211"}}',
            )
        )
        asyncio.run(bg())
        assert calls == [("panel_label", "ng")]

    def test_detect_exception_recorded_as_error(self, monkeypatch):
        """检测抛异常：记一笔 error 且异常原样上抛，只记一次。"""

        def _raise():
            raise RuntimeError("inference boom")

        router, calls = _make_router(monkeypatch, _raise)
        bg = BackgroundTasks()
        with pytest.raises(RuntimeError):
            asyncio.run(
                router._handle_request(
                    background_tasks=bg,
                    file=_FakeUpload(),
                    json_data='{"modelParams": {"product_type": "FU211"}}',
                )
            )
        assert calls == [("panel_label", "error")]

    def test_invalid_json_recorded_as_error(self, monkeypatch):
        """参数校验失败（进不了检测）也必须计数为 error。"""
        router, calls = _make_router(monkeypatch, lambda: dict(_OK_RESULT))
        bg = BackgroundTasks()
        with pytest.raises(InvalidParamsError):
            asyncio.run(
                router._handle_request(
                    background_tasks=bg, file=_FakeUpload(), json_data="not-json"
                )
            )
        assert calls == [("panel_label", "error")]

    def test_stats_failure_never_breaks_detection(self, monkeypatch):
        """统计底层故障（recorder 抛异常）不影响成功响应，也不替换原始检测异常。

        不 patch record_call（走真实埋点入口），只让底层 recorder.record 抛异常，
        锁定"统计绝不影响检测主流程"的集成契约。
        """
        import services.call_stats as cs

        def _raise_record(*args, **kwargs):
            raise RuntimeError("stats db down")

        monkeypatch.setattr(cs.call_stats_recorder, "record", _raise_record)

        # 成功路径：响应正常返回，后台 record_call 吞掉 recorder 异常
        router, _ = _make_router(
            monkeypatch, lambda: dict(_OK_RESULT), patch_record_call=False
        )
        bg = BackgroundTasks()
        result = asyncio.run(
            router._handle_request(
                background_tasks=bg,
                file=_FakeUpload(),
                json_data='{"modelParams": {"product_type": "FU211"}}',
            )
        )
        asyncio.run(bg())
        assert result.code == 1

        # 异常路径：原始检测异常原样上抛，不被统计异常替换
        def _boom():
            raise RuntimeError("inference boom")

        router2, _ = _make_router(monkeypatch, _boom, patch_record_call=False)
        with pytest.raises(RuntimeError, match="inference boom"):
            asyncio.run(
                router2._handle_request(
                    background_tasks=BackgroundTasks(),
                    file=_FakeUpload(),
                    json_data='{"modelParams": {"product_type": "FU211"}}',
                )
            )


class TestStatsEndpoint:
    def _call(self, monkeypatch, recorder, **params):
        from routers import stats_routers

        monkeypatch.setattr(stats_routers, "call_stats_recorder", recorder)
        kwargs = {"scene": None, "start_date": None, "end_date": None}
        kwargs.update(params)
        return asyncio.run(stats_routers.get_call_stats(**kwargs))

    def test_query_all(self, monkeypatch, recorder):
        recorder.record("panel_label", "ok", day="2026-06-10")
        recorder.record("plate_screw", "ng", day="2026-06-10")
        body = self._call(monkeypatch, recorder)
        assert body["code"] == 1
        assert body["message"] == "查询成功"
        assert body["result"]["total"] == 2
        assert set(body["result"]["scenes"]) == {"panel_label", "plate_screw"}

    def test_query_with_filters(self, monkeypatch, recorder):
        recorder.record("panel_label", "ok", day="2026-06-01")
        recorder.record("panel_label", "ok", day="2026-06-10")
        body = self._call(
            monkeypatch,
            recorder,
            scene="panel_label",
            start_date="2026-06-05",
            end_date="2026-06-10",
        )
        assert body["result"]["total"] == 1
        assert body["result"]["scenes"]["panel_label"]["daily"][0]["date"] == "2026-06-10"

    def test_query_empty_db(self, monkeypatch, recorder):
        body = self._call(monkeypatch, recorder)
        assert body["result"] == {"total": 0, "scenes": {}}

    @pytest.mark.parametrize("bad", ["2026/06/10", "20260610", "2026-6-1", "abc"])
    def test_invalid_date_param_rejected(self, monkeypatch, recorder, bad):
        with pytest.raises(InvalidParamsError):
            self._call(monkeypatch, recorder, start_date=bad)
        with pytest.raises(InvalidParamsError):
            self._call(monkeypatch, recorder, end_date=bad)
