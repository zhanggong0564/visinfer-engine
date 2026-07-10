"""base_router 可视化注入：响应含 vis_image 且不污染回流 record。"""
import asyncio
import sys
import types

import numpy as np
import pytest
from fastapi import BackgroundTasks

python_multipart = types.ModuleType("python_multipart")
python_multipart.__version__ = "0.0.20"
sys.modules.setdefault("python_multipart", python_multipart)

from config import settings
from routers.base_router import BaseRouter, DecodedUpload


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeUpload:
    def __init__(self, filename="screw-1.jpg"):
        self.filename = filename

    async def read(self):
        return b"fake-image-bytes"


class _Router(BaseRouter):
    def __init__(self):
        super().__init__(
            router_name="t_router", api_path="/t_detect", summary="t",
            description="t", detector_type="dc_fuse",
        )

    def request_schema(self, json_dict):
        return json_dict

    def get_inputs(self, request_params, image):
        return None


_DETECT_RESULT = {
    "detailList": [
        {"status": "true", "scene": "screw",
         "coordinate": [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
         "accuracy": 0.9, "name": "screw", "color": "#20ff4f"}
    ],
    "status": "true", "error_msg": "", "message": "ok",
}


def _make_router(monkeypatch):
    router = _Router()

    async def _fake_process_image(*args, **kwargs):
        return DecodedUpload(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            raw_bytes=None,
            extension=".jpg",
        )

    monkeypatch.setattr(router, "_process_image", _fake_process_image)

    class _Detector:
        def detect(self, inputs):
            return dict(_DETECT_RESULT)

    monkeypatch.setattr(router, "get_detector_singleton", lambda: _Detector())
    monkeypatch.setattr("routers.base_router.record_call", lambda scene, verdict: None)

    persisted = []
    monkeypatch.setattr(router, "_persist_record", lambda **kw: persisted.append(kw))
    return router, persisted


def test_response_contains_vis_image(monkeypatch):
    monkeypatch.setattr(settings, "VIS_ENABLED", True)
    router, _ = _make_router(monkeypatch)
    resp = _run(router._process_detect_request(
        background_tasks=BackgroundTasks(), file=_FakeUpload(), json_data="{}"))
    assert resp.result.vis_image != ""


def test_persisted_record_has_no_vis_image(monkeypatch):
    monkeypatch.setattr(settings, "VIS_ENABLED", True)
    router, persisted = _make_router(monkeypatch)
    bg = BackgroundTasks()
    _run(router._process_detect_request(
        background_tasks=bg, file=_FakeUpload(), json_data="{}"))
    # background_tasks 已登记但未执行；落盘走 run_sync(self._persist_record, **kw)，
    # kw 里的 result_dict 不应含 vis_image
    found = False
    for t in bg.tasks:
        kw = getattr(t, "kwargs", {}) or {}
        if "result_dict" in kw:
            found = True
            assert "vis_image" not in kw["result_dict"]
    assert found, "未找到 _persist_record 的 result_dict 参数"


def test_disabled_returns_empty_vis_image(monkeypatch):
    monkeypatch.setattr(settings, "VIS_ENABLED", False)
    router, _ = _make_router(monkeypatch)
    resp = _run(router._process_detect_request(
        background_tasks=BackgroundTasks(), file=_FakeUpload(), json_data="{}"))
    assert resp.result.vis_image == ""


def test_guideline_passed_to_render(monkeypatch):
    """panel_label 等场景：inputs.extra.guideline 存在时，应作为 guides 传给渲染器。"""
    from schemas.data_base import InputParamsBusiness
    monkeypatch.setattr(settings, "VIS_ENABLED", True)

    captured = {}

    def _fake_render(image, detail_list, **kwargs):
        captured["guides"] = kwargs.get("guides")
        return "FAKE"

    monkeypatch.setattr("routers.base_router.render_detection_overlay", _fake_render)

    class _GuideRouter(BaseRouter):
        def __init__(self):
            super().__init__(router_name="g", api_path="/g", summary="g",
                             description="g", detector_type="panel_label")

        def request_schema(self, json_dict):
            return json_dict

        def get_inputs(self, request_params, image):
            return InputParamsBusiness(
                image=image, product_type="TK2",
                extra={"guideline": (0.1, 0.2, 0.5, 0.4)},
            )

    router = _GuideRouter()

    async def _fake_process_image(*args, **kwargs):
        return DecodedUpload(
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            raw_bytes=None,
            extension=".jpg",
        )

    monkeypatch.setattr(router, "_process_image", _fake_process_image)

    class _Detector:
        def detect(self, inputs):
            return {"detailList": [], "status": "true", "error_msg": "", "message": "ok"}

    monkeypatch.setattr(router, "get_detector_singleton", lambda: _Detector())
    monkeypatch.setattr("routers.base_router.record_call", lambda scene, verdict: None)
    monkeypatch.setattr(router, "_persist_record", lambda **kw: None)

    _run(router._process_detect_request(
        background_tasks=BackgroundTasks(), file=_FakeUpload(), json_data="{}"))
    assert captured["guides"] == [(0.1, 0.2, 0.5, 0.4)]
