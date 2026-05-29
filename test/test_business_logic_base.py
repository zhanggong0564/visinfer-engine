"""BusinessLogicBase 业务逻辑基类单元测试"""
import numpy as np
from unittest.mock import MagicMock
from schemas.data_base import MoMResult, DetectionItem, DetectResult, InputParamsBusiness
from schemas.inference_context import InferenceContext
from services.base.business_logic_base import BusinessLogicBase


class ConcreteLogic(BusinessLogicBase):
    """测试用具体实现"""

    def _initialize_model(self, settings):
        self.detector = MagicMock()

    def business_post_process(self, ctx):
        mom = MoMResult()
        mom.detailList = [
            DetectionItem(status=True, scene="test", coordinate=[10, 20, 100, 200], accuracy=0.95, name="item1")
        ]
        mom.status = True
        mom.message = "ok"
        ctx.result = mom


class TestNormalizeHook:
    def _ctx(self, coordinate):
        ctx = InferenceContext(image=np.zeros((800, 400, 3), dtype=np.uint8), h=800, w=400)
        mom = MoMResult()
        mom.detailList = [DetectionItem(coordinate=coordinate)]
        mom.status = True
        ctx.result = mom
        return ctx

    def test_4value_coordinate_normalization(self):
        logic = ConcreteLogic(MagicMock())
        ctx = self._ctx([10, 20, 100, 200])
        logic.normalize_hook(ctx)
        coord = ctx.result.detailList[0].coordinate
        assert len(coord) == 8
        assert coord[0] == 10 / 400
        assert coord[1] == 20 / 800
        assert coord[2] == 100 / 400
        assert coord[5] == 200 / 800
        assert coord[6] == 10 / 400

    def test_8value_coordinate_normalization(self):
        logic = ConcreteLogic(MagicMock())
        ctx = self._ctx([10, 20, 100, 20, 100, 200, 10, 200])
        logic.normalize_hook(ctx)
        coord = ctx.result.detailList[0].coordinate
        assert len(coord) == 8
        assert all(0 <= c <= 1 for c in coord)

    def test_empty_detail_list(self):
        logic = ConcreteLogic(MagicMock())
        ctx = InferenceContext(image=np.zeros((800, 400, 3), dtype=np.uint8), h=800, w=400)
        ctx.result = MoMResult()
        logic.normalize_hook(ctx)
        assert ctx.result.detailList == []


class TestDetectFlow:
    def test_detect_calls_detector_and_post_process(self):
        logic = ConcreteLogic(MagicMock())
        input_params = InputParamsBusiness(
            image=np.zeros((800, 400, 3), dtype=np.uint8),
            SN="test_sn", product_type="TYPE1", is_registered=False,
        )
        logic.detector.infer.return_value = DetectResult()
        result = logic.detect(input_params)
        logic.detector.infer.assert_called_once_with(input_params.image)
        assert isinstance(result, MoMResult)
        # 坐标已被默认 normalize_hook 归一化为 8 值
        assert len(result.detailList[0].coordinate) == 8

    def test_build_context_sets_dimensions(self):
        logic = ConcreteLogic(MagicMock())
        params = InputParamsBusiness(
            image=np.zeros((800, 400, 3), dtype=np.uint8), product_type="T1", rule="front",
        )
        ctx = logic.build_context(params)
        assert ctx.h == 800 and ctx.w == 400
        assert ctx.product_type == "T1" and ctx.rule == "front"
