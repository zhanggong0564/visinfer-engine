"""业务异常体系。

业务代码 raise VisionAPIError 子类；app.py 全局 handler 统一翻译为
CommonResponse(code=..., error_msg=...)，HTTP 状态码固定 200。
"""
from .error_codes import ErrorCode, ERROR_CODE_MESSAGES


class VisionAPIError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(self, error_msg: str = "", **context):
        self.error_msg = error_msg or ERROR_CODE_MESSAGES[self.code]
        self.context = context
        super().__init__(self.error_msg)


class InvalidParamsError(VisionAPIError):
    code = ErrorCode.INVALID_PARAMS


class InvalidImageError(VisionAPIError):
    code = ErrorCode.INVALID_IMAGE


class ProductNotRegisteredError(VisionAPIError):
    code = ErrorCode.PRODUCT_NOT_REGISTERED


class ModelInferenceError(VisionAPIError):
    code = ErrorCode.MODEL_INFERENCE_ERROR


class InternalError(VisionAPIError):
    code = ErrorCode.INTERNAL_ERROR
