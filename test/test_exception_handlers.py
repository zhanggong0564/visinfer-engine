"""全局异常处理器集成测试（HTTPX ASGITransport）。"""
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
import httpx
from pydantic import BaseModel

from schemas.exceptions import (
    InvalidParamsError, InvalidImageError, ProductNotRegisteredError,
    ModelInferenceError, InternalError, VisionAPIError,
)
from schemas.error_codes import ErrorCode, ERROR_CODE_MESSAGES


class _ASGIClient:
    """绕开 PPOCR Python 3.10.0 下 AnyIO blocking portal 的测试客户端。"""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def _send():
            transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_send())
        finally:
            loop.close()

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


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
    async def _r1():
        raise InvalidParamsError("json_data 格式非法")

    @app.get("/raise/invalid_image")
    async def _r2():
        raise InvalidImageError("图片解码失败")

    @app.get("/raise/product_not_registered")
    async def _r3():
        raise ProductNotRegisteredError(
            "产品型号 'X1_2' 未注册", product_type="X1_2", scenario="panel_label"
        )

    @app.get("/raise/model_inference")
    async def _r4():
        raise ModelInferenceError("ONNX 推理失败")

    @app.get("/raise/internal")
    async def _r5():
        raise InternalError()

    @app.get("/raise/uncaught")
    async def _r6():
        raise ValueError("意外异常")

    class _Req(BaseModel):
        x: int

    @app.post("/raise/validation")
    async def _r7(req: _Req):
        return {"ok": True}

    return _ASGIClient(app)


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
        # 未捕获异常对外只回固定文案，不透传 str(exc) 内部细节（防信息泄露）
        assert body["result"]["error_msg"] == ERROR_CODE_MESSAGES[ErrorCode.INTERNAL_ERROR]
        assert "意外异常" not in body["result"]["error_msg"]

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
        return _ASGIClient(app)

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
