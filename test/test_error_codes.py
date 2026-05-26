"""ErrorCode 枚举与默认文案测试"""
import pytest
from schemas.error_codes import ErrorCode, ERROR_CODE_MESSAGES


class TestErrorCodeValues:
    def test_success(self):
        assert int(ErrorCode.SUCCESS) == 1

    def test_client_errors(self):
        assert int(ErrorCode.INVALID_PARAMS) == 1001
        assert int(ErrorCode.INVALID_IMAGE) == 1002
        assert int(ErrorCode.PRODUCT_NOT_REGISTERED) == 1003

    def test_server_errors(self):
        assert int(ErrorCode.MODEL_INFERENCE_ERROR) == 5001
        assert int(ErrorCode.INTERNAL_ERROR) == 5000

    def test_is_int_enum(self):
        # 必须能直接和 int 比较，否则 JSON 序列化时会出问题
        assert ErrorCode.SUCCESS == 1
        assert ErrorCode.INVALID_PARAMS == 1001


class TestErrorCodeMessages:
    def test_all_codes_have_default_message(self):
        for code in ErrorCode:
            assert code in ERROR_CODE_MESSAGES
            assert isinstance(ERROR_CODE_MESSAGES[code], str)
            assert len(ERROR_CODE_MESSAGES[code]) > 0

    def test_success_message(self):
        assert ERROR_CODE_MESSAGES[ErrorCode.SUCCESS] == "检测成功"

    def test_messages_are_chinese(self):
        # 至少包含一个中文字符
        for code, msg in ERROR_CODE_MESSAGES.items():
            assert any("一" <= ch <= "鿿" for ch in msg), f"{code} 文案非中文: {msg}"
