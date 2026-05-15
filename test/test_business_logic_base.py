"""BusinessLogicBase 业务逻辑基类单元测试"""
import numpy as np
import pytest
from unittest.mock import MagicMock
from schemas.data_base import (
    MoMResult,
    DetectionItem,
    DetectResult,
    InputParamsBusiness,
)
from services.base.business_logic_base import BusinessLogicBase


class ConcreteLogic(BusinessLogicBase):
    """测试用具体实现"""

    def _initialize_model(self, settings):
        self.detector = MagicMock()

    def business_logic_post_process(self, result, product_type):
        mom = MoMResult()
        mom.detailList = [
            DetectionItem(
                status=True,
                scene="test",
                coordinate=[10, 20, 100, 200],
                accuracy=0.95,
                name="item1",
            )
        ]
        mom.status = True
        mom.message = "ok"
        return mom


class TestResultPostProcess:
    def test_4value_coordinate_normalization(self):
        logic = ConcreteLogic(MagicMock())
        mom = MoMResult()
        mom.detailList = [
            DetectionItem(coordinate=[10, 20, 100, 200])
        ]
        mom.status = True
        logic.w = 400
        logic.h = 800

        result = logic.result_post_process(mom)
        coord = result.detailList[0].coordinate
        # 4值坐标 → 扩展为8值，然后归一化
        assert len(coord) == 8
        assert coord[0] == 10 / 400
        assert coord[1] == 20 / 800
        assert coord[2] == 100 / 400
        assert coord[3] == 20 / 800
        assert coord[4] == 100 / 400
        assert coord[5] == 200 / 800
        assert coord[6] == 10 / 400
        assert coord[7] == 200 / 800

    def test_8value_coordinate_normalization(self):
        logic = ConcreteLogic(MagicMock())
        mom = MoMResult()
        mom.detailList = [
            DetectionItem(coordinate=[10, 20, 100, 20, 100, 200, 10, 200])
        ]
        mom.status = True
        logic.w = 400
        logic.h = 800

        result = logic.result_post_process(mom)
        coord = result.detailList[0].coordinate
        assert len(coord) == 8
        assert all(0 <= c <= 1 for c in coord)

    def test_empty_detail_list(self):
        logic = ConcreteLogic(MagicMock())
        mom = MoMResult()
        logic.w = 400
        logic.h = 800

        result = logic.result_post_process(mom)
        assert result.detailList == []


class TestDetectFlow:
    def test_detect_calls_detector_and_post_process(self):
        logic = ConcreteLogic(MagicMock())
        logic.w = 400
        logic.h = 800

        input_params = InputParamsBusiness(
            image=np.zeros((800, 400, 3), dtype=np.uint8),
            SN="test_sn",
            product_type="TYPE1",
            is_registered=False,
        )
        # 设置 mock detector 返回
        mock_result = DetectResult()
        logic.detector.infer.return_value = mock_result

        result = logic.detect(input_params)
        logic.detector.infer.assert_called_once_with(input_params.image)
        assert isinstance(result, MoMResult)
