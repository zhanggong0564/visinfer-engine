'''
@Author       : gongzhang4
@Date         : 2026-01-07 05:45:30
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 09:18:18
@FilePath     : app.py
@Description  :
'''

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from config import settings
from utils import vision_logger
from schemas.error_codes import ErrorCode, ERROR_CODE_MESSAGES
from schemas.exceptions import VisionAPIError
from routers import RouterRegistry
import uvicorn
from services import detection_factory

# 路由注册器：导入期只发现/注册路由，模型预加载延后到 lifespan
router_registry = RouterRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：startup 统一预加载模型，shutdown 预留清理位。"""
    vision_logger.info("应用启动：开始预加载检测模型 ...")
    router_registry.preload_all()
    vision_logger.info(f"模型预加载完成，可用场景: {detection_factory.list_scenarios()}")
    yield
    vision_logger.info("应用关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="移动视觉算法API服务，提供直流熔丝检测等视觉算法功能",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应配置具体域名
    # allow_origins="*" 与 allow_credentials=True 是浏览器禁止的非法组合，
    # 这里关闭凭证；若需携带 cookie/凭证，请改为具体域名白名单并设为 True
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """统一访问日志钩子：为每个请求生成 request-id，记录耗时与状态码。

    覆盖所有端点（含健康检查/异常），把计时与访问日志从各业务 handler 中抽离。
    """
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        vision_logger.exception(
            f"[{request_id}] {request.method} {request.url.path} 异常 耗时={latency_ms:.1f}ms"
        )
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    vision_logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} 耗时={latency_ms:.1f}ms"
    )
    response.headers["X-Request-ID"] = request_id
    return response


# 注册路由（仅发现与注册，不加载模型）
router_registry.register_all_routers(app, "routers")


# 全局异常处理
def _build_error_response(code: ErrorCode, error_msg: str) -> dict:
    """统一错误响应体，保持 CommonResponse 完整 schema"""
    public_message = ERROR_CODE_MESSAGES[code]
    return {
        "code": int(code),
        "message": public_message,
        "result": {
            "detailList": [],
            "status": "false",
            "error_msg": error_msg,
            "message": public_message,
        },
    }


@app.exception_handler(VisionAPIError)
async def vision_api_exception_handler(request: Request, exc: VisionAPIError):
    """业务层主动抛出的异常 → 翻译为 CommonResponse"""
    vision_logger.error(
        f"业务异常 code={int(exc.code)} msg={exc.error_msg} context={exc.context}"
    )
    return JSONResponse(
        status_code=200,
        content=_build_error_response(exc.code, exc.error_msg),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic / FastAPI 自带的参数校验失败 → INVALID_PARAMS"""
    vision_logger.error(f"参数校验失败: {exc.errors()}")
    return JSONResponse(
        status_code=200,
        content=_build_error_response(
            ErrorCode.INVALID_PARAMS, f"参数校验失败: {exc.errors()}"
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """未捕获服务端异常 → INTERNAL_ERROR 兜底"""
    vision_logger.exception(f"未捕获异常: {exc}")
    return JSONResponse(
        status_code=200,
        content=_build_error_response(ErrorCode.INTERNAL_ERROR, str(exc)),
    )


# 健康检查接口
@app.get("/", tags=["健康检查"])
async def root():
    """根路径健康检查"""
    return {
        "code": 1,
        "message": f"{settings.API_TITLE} 服务运行正常",
        "result": {"service": settings.API_TITLE, "version": settings.API_VERSION, "status": "running"},
        "service": detection_factory.list_scenarios(),
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    return {"code": 1, "message": "服务健康", "result": {"status": "healthy", "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}}


# API文档自定义配置
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI界面"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# 应用启动函数
def main():
    """应用启动入口"""
    vision_logger.info(f"启动 {settings.API_TITLE} v{settings.API_VERSION}")

    # reload=True 时 uvicorn 会忽略 workers，故二者按配置互斥：
    # 开发用 RELOAD=True 单进程热重载；生产用 RELOAD=False + 多 WORKERS
    run_kwargs = dict(
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,  # 禁用uvicorn的访问日志，使用自定义日志
    )
    if settings.RELOAD:
        run_kwargs["reload"] = True
    else:
        run_kwargs["workers"] = settings.WORKERS
    uvicorn.run("app:app", **run_kwargs)


if __name__ == "__main__":
    main()
