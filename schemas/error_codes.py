"""公共错误码与默认文案"""
from enum import IntEnum


class ErrorCode(IntEnum):
    SUCCESS = 1

    # 1xxx 客户端错误
    INVALID_PARAMS = 1001
    INVALID_IMAGE = 1002
    PRODUCT_NOT_REGISTERED = 1003

    # 5xxx 服务端错误
    MODEL_INFERENCE_ERROR = 5001
    INTERNAL_ERROR = 5000


ERROR_CODE_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.SUCCESS: "检测成功",
    ErrorCode.INVALID_PARAMS: "请求参数格式错误",
    ErrorCode.INVALID_IMAGE: "图片读取失败，请检查文件格式",
    ErrorCode.PRODUCT_NOT_REGISTERED: "产品型号未注册",
    ErrorCode.MODEL_INFERENCE_ERROR: "算法推理失败",
    ErrorCode.INTERNAL_ERROR: "算法内部错误",
}
