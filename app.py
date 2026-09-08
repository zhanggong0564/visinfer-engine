'''
@Author       : gongzhang4
@Date         : 2026-01-07 05:45:30
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 09:18:18
@FilePath     : app.py
@Description  :
'''

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from config import settings
from utils import vision_logger
from utils.openapi_docs import configure_openapi_docs
from utils.request_logging import RequestLoggingMiddleware, log_request_error
from schemas.error_codes import ErrorCode, ERROR_CODE_MESSAGES
from schemas.exceptions import VisionAPIError
from routers import RouterRegistry
import uvicorn
from services.scenario_registry import scenario_registry
from services.inference import runtime_status_registry
from services.inference.admission import inference_admission_controller

# 路由注册器：导入期只发现/注册路由，模型预加载延后到 lifespan
router_registry = RouterRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：startup 预加载，shutdown 释放模型资源。"""
    vision_logger.info("应用启动：开始预加载检测模型 ...")
    router_registry.preload_all()
    vision_logger.info(f"模型预加载完成，可用场景: {scenario_registry.list_scenarios()}")
    try:
        yield
    finally:
        router_registry.close_all()
        vision_logger.info("应用关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="移动视觉算法API服务，提供直流熔丝检测等视觉算法功能",
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

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


app.add_middleware(RequestLoggingMiddleware, skip_paths=settings.ACCESS_LOG_SKIP_PATHS)


# 注册路由（仅发现与注册，不加载模型）
router_registry.register_all_routers(app, "routers")
configure_openapi_docs(app)


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
            "verdict": None,
            "error_msg": error_msg,
            "message": public_message,
        },
    }


@app.exception_handler(VisionAPIError)
async def vision_api_exception_handler(request: Request, exc: VisionAPIError):
    """业务层主动抛出的异常 → 翻译为 CommonResponse"""
    headers = log_request_error(request, exc, exc.code, exc.context.get("validation_errors"))
    return JSONResponse(
        status_code=200,
        content=_build_error_response(exc.code, exc.error_msg),
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic / FastAPI 自带的参数校验失败 → INVALID_PARAMS"""
    details = [{"loc": error["loc"], "type": error["type"], "msg": error["msg"]} for error in exc.errors()]
    headers = log_request_error(request, exc, ErrorCode.INVALID_PARAMS, details)
    return JSONResponse(
        status_code=200,
        content=_build_error_response(
            ErrorCode.INVALID_PARAMS, f"参数校验失败: {exc.errors()}"
        ),
        headers=headers,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """未捕获服务端异常 → INTERNAL_ERROR 兜底。

    对外只回固定文案，不把 str(exc)（含路径/栈片段等内部细节）透传给调用方；
    异常详情仅进日志（含 request-id 由访问日志中间件输出）便于排障。
    """
    headers = log_request_error(request, exc, ErrorCode.INTERNAL_ERROR)
    return JSONResponse(
        status_code=200,
        content=_build_error_response(
            ErrorCode.INTERNAL_ERROR, ERROR_CODE_MESSAGES[ErrorCode.INTERNAL_ERROR]
        ),
        headers=headers,
    )


# 健康检查接口
@app.get("/", tags=["健康检查"])
async def root():
    """根路径健康检查"""
    return {
        "code": 1,
        "message": f"{settings.API_TITLE} 服务运行正常",
        "result": {"service": settings.API_TITLE, "version": settings.API_VERSION, "status": "running"},
        "service": scenario_registry.list_scenarios(),
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    return {"code": 1, "message": "服务健康", "result": {"status": "healthy", "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}}


@app.get("/health/ready", tags=["健康检查"])
async def readiness_check():
    """就绪检查：启用场景的模型全部预加载成功后才对外接流量。"""
    failed_scenes = router_registry.failed_scenes()
    admission = inference_admission_controller.snapshot()
    models = runtime_status_registry.public_snapshot()
    cuda_runtime_ready = bool(models) and all(
        "CUDAExecutionProvider" in model["providers"]
        for model in models
    )
    ready = router_registry.is_ready() and (
        not settings.ONNX_REQUIRE_CUDA or cuda_runtime_ready
    )
    runtime = {
        "require_cuda": settings.ONNX_REQUIRE_CUDA,
        "models": models,
        "inference": {
            "max_concurrency": admission.max_concurrency,
            "active": admission.active,
            "waiting": admission.waiting,
        },
    }
    content = {
        "code": 1 if ready else int(ErrorCode.INTERNAL_ERROR),
        "message": "服务已就绪" if ready else "服务未就绪",
        "result": {
            "status": "ready" if ready else "not_ready",
            "failed_scenes": failed_scenes,
            "runtime": runtime,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    return JSONResponse(status_code=200 if ready else 503, content=content)


# API文档自定义配置
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI界面"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
        swagger_ui_parameters={"validatorUrl": None},
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
