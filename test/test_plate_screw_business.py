"""plate_screw 业务逻辑单元测试"""
import numpy as np
import pytest
from unittest.mock import patch
from schemas.data_base import DetectResult, MoMResult
from schemas.inference_context import InferenceContext


@pytest.fixture
def api_instance():
    with patch("services.plate_screw.business_logic.PlateScrewDetect"):
        from services.plate_screw.business_logic import PlateScrewJudgeApi
        from config import settings
        yield PlateScrewJudgeApi(settings)


def _ctx(raw, w=1000, h=1000):
    ctx = InferenceContext(image=np.zeros((h, w, 3), dtype=np.uint8), h=h, w=w)
    ctx.raw_result = raw
    return ctx


class TestBusinessPostProcess:
    def test_screw_present_status_true(self, api_instance):
        raw = DetectResult(
            boxes=[[10, 10, 100, 100], [20, 20, 40, 40]],
            scores=[0.9, 0.8],
            class_ids=[0, 1],
            class_names=["metal_plate", "metal_screw"],
        )
        ctx = _ctx(raw)
        api_instance.business_post_process(ctx)
        assert isinstance(ctx.result, MoMResult)
        assert ctx.result.status is True

    def test_missing_screw_status_false(self, api_instance):
        # no_metal_screw 落在 metal_plate 内部 → 该铁片组里出现 'no' 标签 → status False
        raw = DetectResult(
            boxes=[[10, 10, 200, 200], [50, 50, 80, 80]],
            scores=[0.9, 0.85],
            class_ids=[0, 1],
            class_names=["metal_plate", "no_metal_screw"],
        )
        ctx = _ctx(raw)
        api_instance.business_post_process(ctx)
        assert ctx.result.status is False
