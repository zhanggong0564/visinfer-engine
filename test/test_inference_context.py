"""InferenceContext / PreprocMeta 数据结构单元测试"""
import numpy as np
from schemas.inference_context import InferenceContext, PreprocMeta


class TestInferenceContext:
    def test_defaults(self):
        img = np.zeros((800, 400, 3), dtype=np.uint8)
        ctx = InferenceContext(image=img, h=800, w=400)
        assert ctx.product_type == ""
        assert ctx.rule == "all"
        assert ctx.is_registered is False
        assert ctx.raw_result is None
        assert ctx.result is None
        assert ctx.skip_normalize is False

    def test_fields_assignable(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        ctx = InferenceContext(
            image=img, h=10, w=10, product_type="T1", rule="front", is_registered=True
        )
        assert ctx.product_type == "T1"
        assert ctx.rule == "front"
        assert ctx.is_registered is True


class TestPreprocMeta:
    def test_fields(self):
        meta = PreprocMeta(r=0.5, dw=12.0, dh=8.0, src_shape=(480, 640, 3))
        assert meta.r == 0.5
        assert meta.dw == 12.0
        assert meta.dh == 8.0
        assert meta.src_shape == (480, 640, 3)
        assert meta.ori_img is None
