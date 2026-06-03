"""全局异常处理器集成测试（FastAPI TestClient）"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from schemas.exceptions import (
    InvalidParamsError, InvalidImageError, ProductNotRegisteredError,
    ModelInferenceError, InternalError, VisionAPIError,
)
from schemas.error_codes import ErrorCode


@pytest.fixture
def client():
    """构造一个最小 app 并挂上全局 handler，用测试路由触发各种异常"""
    from app import (
        vision_api_exception_handler,
        validation_exception_handler,
        global_exception_handler,
    )

    app = FastAPI()
    app.add_exception_handler(VisionAPIError, vision_api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    @app.get("/raise/invalid_params")
    def _r1():
        raise InvalidParamsError("json_data 格式非法")

    @app.get("/raise/invalid_image")
    def _r2():
        raise InvalidImageError("图片解码失败")

    @app.get("/raise/product_not_registered")
    def _r3():
        raise ProductNotRegisteredError(
            "产品型号 'X1_2' 未注册", product_type="X1_2", scenario="panel_label"
        )

    @app.get("/raise/model_inference")
    def _r4():
        raise ModelInferenceError("ONNX 推理失败")

    @app.get("/raise/internal")
    def _r5():
        raise InternalError()

    @app.get("/raise/uncaught")
    def _r6():
        raise ValueError("意外异常")

    class _Req(BaseModel):
        x: int

    @app.post("/raise/validation")
    def _r7(req: _Req):
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


class TestExceptionHandlers:
    def test_invalid_params_response(self, client):
        resp = client.get("/raise/invalid_params")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.INVALID_PARAMS)
        assert body["result"]["status"] == "false"
        assert body["result"]["detailList"] == []
        assert body["result"]["error_msg"] == "json_data 格式非法"
        assert body["message"] == "请求参数格式错误"

    def test_invalid_image_response(self, client):
        resp = client.get("/raise/invalid_image")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.INVALID_IMAGE)
        assert body["result"]["error_msg"] == "图片解码失败"

    def test_product_not_registered(self, client):
        resp = client.get("/raise/product_not_registered")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.PRODUCT_NOT_REGISTERED)
        assert "X1_2" in body["result"]["error_msg"]
        # context 不暴露给客户端
        assert "context" not in body
        assert "product_type" not in body

    def test_model_inference_error(self, client):
        resp = client.get("/raise/model_inference")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.MODEL_INFERENCE_ERROR)

    def test_internal_error_default_msg(self, client):
        resp = client.get("/raise/internal")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.INTERNAL_ERROR)
        assert body["result"]["error_msg"] == "算法内部错误"

    def test_uncaught_exception_falls_through(self, client):
        resp = client.get("/raise/uncaught")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.INTERNAL_ERROR)
        assert "意外异常" in body["result"]["error_msg"]

    def test_request_validation_error(self, client):
        resp = client.post("/raise/validation", json={"x": "not_an_int"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.INVALID_PARAMS)
        assert "参数校验失败" in body["result"]["error_msg"]

    def test_response_schema_complete_on_error(self, client):
        """错误响应也保持 CommonResponse 完整 schema"""
        resp = client.get("/raise/invalid_params")
        body = resp.json()
        assert set(body.keys()) == {"code", "message", "result"}
        assert set(body["result"].keys()) == {"detailList", "status", "error_msg", "message"}


class TestBaseRouterExceptionTranslation:
    """验证 base_router 抛出的异常被全局 handler 正确翻译"""

    @pytest.fixture
    def real_app_client(self):
        from app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_invalid_json_data(self, real_app_client):
        """传入非法 JSON 字符串 → code=1001"""
        files = {"file": ("test.jpg", b"\xff\xd8\xff\xe0fake_jpeg", "image/jpeg")}
        data = {"json_data": "this is not json"}
        resp = real_app_client.post("/api/v1/panel_label_detect", files=files, data=data)
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.INVALID_PARAMS)

    def test_invalid_image_bytes(self, real_app_client):
        """图片字节不可解码 → code=1002"""
        files = {"file": ("test.jpg", b"definitely_not_an_image", "image/jpeg")}
        data = {"json_data": '{"product": "wind_power", "type": "1017KM1_1", "modelParams": {"product_type": "1017KM1_1", "line_order": "1017KM1-1", "guideline_coordinates": "0.1,0.1,0.8,0.8"}}'}
        resp = real_app_client.post("/api/v1/panel_label_detect", files=files, data=data)
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == int(ErrorCode.INVALID_IMAGE)
