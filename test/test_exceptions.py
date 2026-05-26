"""VisionAPIError 异常体系测试"""
import pytest
from schemas.error_codes import ErrorCode, ERROR_CODE_MESSAGES
from schemas.exceptions import (
    VisionAPIError,
    InvalidParamsError,
    InvalidImageError,
    ProductNotRegisteredError,
    ModelInferenceError,
    InternalError,
)


class TestVisionAPIErrorBase:
    def test_default_code_is_internal(self):
        exc = VisionAPIError()
        assert exc.code == ErrorCode.INTERNAL_ERROR

    def test_default_message_from_dict(self):
        exc = VisionAPIError()
        assert exc.error_msg == ERROR_CODE_MESSAGES[ErrorCode.INTERNAL_ERROR]

    def test_custom_message(self):
        exc = VisionAPIError("自定义错误信息")
        assert exc.error_msg == "自定义错误信息"

    def test_context_stored(self):
        exc = VisionAPIError("err", product_type="X1", scenario="panel_label")
        assert exc.context == {"product_type": "X1", "scenario": "panel_label"}

    def test_empty_context_by_default(self):
        exc = VisionAPIError("err")
        assert exc.context == {}


class TestSubclassCodes:
    def test_invalid_params(self):
        assert InvalidParamsError().code == ErrorCode.INVALID_PARAMS

    def test_invalid_image(self):
        assert InvalidImageError().code == ErrorCode.INVALID_IMAGE

    def test_product_not_registered(self):
        assert ProductNotRegisteredError().code == ErrorCode.PRODUCT_NOT_REGISTERED

    def test_model_inference_error(self):
        assert ModelInferenceError().code == ErrorCode.MODEL_INFERENCE_ERROR

    def test_internal_error(self):
        assert InternalError().code == ErrorCode.INTERNAL_ERROR


class TestDefaultMessageUsesSubclassCode:
    def test_invalid_params_default_msg(self):
        exc = InvalidParamsError()
        assert exc.error_msg == ERROR_CODE_MESSAGES[ErrorCode.INVALID_PARAMS]

    def test_product_not_registered_custom_msg(self):
        exc = ProductNotRegisteredError(
            "产品型号 'X1_2' 未在 panel_label PRODUCT_TYPE 中注册",
            product_type="X1_2",
            scenario="panel_label",
        )
        assert "X1_2" in exc.error_msg
        assert exc.context["product_type"] == "X1_2"
