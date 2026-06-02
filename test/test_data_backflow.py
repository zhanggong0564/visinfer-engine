"""数据回流落盘单元测试

回归点：检测失败（如型号未注册）时，图片仍须落盘以便收集新型号样本。
"""
import asyncio

import numpy as np
import pytest
from fastapi import BackgroundTasks

from routers.base_router import BaseRouter
from schemas.exceptions import ProductNotRegisteredError


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


def _make_router(monkeypatch, detect_side_effect):
    router = _Router()

    # 跳过真实图片解码
    async def _fake_process_image(file):
        return np.zeros((10, 10, 3), dtype=np.uint8), False

    monkeypatch.setattr(router, "_process_image", _fake_process_image)

    class _Detector:
        def detect(self, inputs):
            return detect_side_effect()

    monkeypatch.setattr(router, "get_detector_singleton", lambda: _Detector())

    # _persist_record 记录调用而不真正落盘
    calls = []
    monkeypatch.setattr(router, "_persist_record", lambda **kw: calls.append(kw))
    return router, calls


def test_persist_called_when_detect_fails(monkeypatch):
    """检测抛异常（型号未注册）时，仍应调用数据回流落盘并传出型号兜底目录。"""

    def _raise():
        raise ProductNotRegisteredError(
            "产品型号 'FU211' 未注册", product_type="FU211", scenario="panel_label"
        )

    router, calls = _make_router(monkeypatch, _raise)
    bg = BackgroundTasks()

    with pytest.raises(ProductNotRegisteredError):
        asyncio.run(
            router._handle_request(
                background_tasks=bg,
                file=_FakeUpload(),
                json_data='{"modelParams": {"product_type": "FU211"}}',
            )
        )

    # 即便检测失败，回流也必须落盘
    assert len(calls) == 1, "检测失败时数据回流未落盘"
    assert calls[0]["original_filename"] == "线标检验FU211-1779526099406.jpg"
