'''
@Author       : gongzhang4
@Date         : 2026-01-07 05:45:30
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 09:18:18
@FilePath     : app.py
@Description  :
'''

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from config import settings
from utils import vision_logger
from routers import RouterRegistry
import uvicorn
from services import detection_factory


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="移动视觉算法API服务，提供直流熔丝检测等视觉算法功能",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 注册路由
router_registry = RouterRegistry()
router_registry.register_all_routers(app, "routers")


# 全局异常处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求参数验证异常"""
    vision_logger.error(f"参数验证失败: {exc.errors()}")
    return JSONResponse(
        status_code=422, content={"code": 0, "message": "请求参数格式错误", "result": {"errors": exc.errors()}}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """处理全局异常"""
    vision_logger.error(f"服务器内部错误: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "code": 0,
            "message": "算法内部错误",
            "result": {
                "detailList": [{"status": False, "scene": "", "accuracy": 0, "coordinate": []}],
                "status": False,
                "error_msg": str(exc),
                "message": "算法内部错误",
            },
        },
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
    return {"code": 1, "message": "服务健康", "result": {"status": "healthy", "timestamp": "2026-01-07T08:30:00Z"}}


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

    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,  # 开发模式开启热重载
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,  # 禁用uvicorn的访问日志，使用自定义日志
        workers=settings.WORKERS,
    )


if __name__ == "__main__":
    main()
