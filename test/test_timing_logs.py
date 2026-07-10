"""阶段耗时日志测试。"""
import asyncio
from unittest.mock import MagicMock

import numpy as np
from fastapi import BackgroundTasks

from routers.base_router import BaseRouter
from schemas.data_base import DetectResult, DetectionItem, InputParamsBusiness, MoMResult
from schemas.inference_context import InferenceContext
from services.base.business_logic_base import BusinessLogicBase


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _LogCapture:
    def __init__(self):
        self.messages = []

    def info(self, message, *args, **kwargs):
        if args:
            try:
                message = message.format(*args)
            except Exception:
                pass
        self.messages.append(str(message))

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


class _ConcreteLogic(BusinessLogicBase):
    def _initialize_model(self, settings):
        self.detector = MagicMock()

    def business_post_process(self, ctx: InferenceContext):
        result = MoMResult()
        result.detailList = [
            DetectionItem(status=True, scene="test", coordinate=[1, 2, 3, 4])
        ]
        result.status = True
        ctx.result = result


def test_business_logic_detect_logs_template_stage_timings(monkeypatch):
    import services.base.business_logic_base as module

    logs = _LogCapture()
    monkeypatch.setattr(module, "vision_logger", logs)

    logic = _ConcreteLogic(MagicMock())
    logic.detector.infer.return_value = DetectResult()
    params = InputParamsBusiness(
        image=np.zeros((10, 20, 3), dtype=np.uint8),
        product_type="T1",
    )

    logic.detect(params)

    timing_logs = [m for m in logs.messages if "业务检测阶段耗时" in m]
    assert timing_logs
    log = timing_logs[-1]
    for stage in (
        "build_context",
        "preprocess_hook",
        "detector_infer",
        "business_post_process",
        "normalize_hook",
        "finalize_hook",
        "total",
    ):
        assert f"{stage}=" in log


class _FakeUpload:
    filename = "sample.jpg"


class _Router(BaseRouter):
    def __init__(self):
        self.router_name = "timing_router"
        self.instance = None
        self.detector_type = "timing_scene"
        self.tag = None

    def request_schema(self, json_dict):
        return json_dict

    def get_inputs(self, request_params, image):
        return {"image": image}


async def _handle_with_background(router, background_tasks, file, json_data):
    return await router._handle_request(
        background_tasks=background_tasks,
        file=file,
        json_data=json_data,
    )


def test_base_router_logs_request_stage_timings(monkeypatch):
    import routers.base_router as module

    logs = _LogCapture()
    monkeypatch.setattr(module, "vision_logger", logs)

    router = _Router()

    async def _process_image(
        file, original_filename, received_at, fallback_product_type,
        stage_recorder=None,
    ):
        if stage_recorder:
            stage_recorder("image_read", 1.0)
            stage_recorder("image_format_detect", 0.1)
            stage_recorder("image_decode", 2.0)
            stage_recorder("image_stage_write", 1.5)
            stage_recorder("image_commit", 0.1)
        return module.DecodedUpload(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_bytes=None,
            extension=".jpg",
        )

    class _Detector:
        def detect(self, inputs):
            return {
                "detailList": [],
                "status": "true",
                "error_msg": "",
                "message": "ok",
            }

    monkeypatch.setattr(router, "_process_image", _process_image)
    monkeypatch.setattr(router, "_persist_record", lambda **kwargs: None)
    monkeypatch.setattr(router, "get_detector_singleton", lambda: _Detector())
    monkeypatch.setattr(module, "record_call", lambda scene, verdict: None)

    async def _inline_run_sync(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(module, "run_sync", _inline_run_sync)

    response = _run(
        _handle_with_background(
            router,
            BackgroundTasks(),
            _FakeUpload(),
            '{"modelParams": {"product_type": "T1"}}',
        )
    )

    assert response.code == 1
    timing_logs = [m for m in logs.messages if "请求阶段耗时" in m]
    assert timing_logs
    log = timing_logs[-1]
    for stage in (
        "validate_params",
        "image_read",
        "image_decode",
        "process_image",
        "image_format_detect",
        "image_stage_write",
        "image_commit",
        "build_inputs",
        "get_detector",
        "detect",
        "result_to_dict",
        "sanitize_result",
        "response_build",
        "total",
    ):
        assert f"{stage}=" in log
