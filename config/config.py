'''
@Author       : gongzhang4
@Date         : 2026-01-23 06:35:58
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 11:53:17
@FilePath     : config.py
@Description  :
'''

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_TITLE: str = "Mobile Vision alg API"
    API_VERSION: str = "2.0.0"

    HOST: str = "0.0.0.0"
    PORT: int = 3001

    LOG_DIR: str = "logs"
    # 数据回流落盘根目录（相对路径按运行 cwd 解析；容器内 cwd=/app/workspace）。
    # 设为可配置 + cwd 锚定，便于通过卷挂载持久化到宿主，避免编译为 .so 后写入 venv 内。
    DATA_DIR: str = "data"
    LOG_LEVEL: str = "INFO"
    # 健康探针等高频端点的访问日志静默名单：命中且响应正常（<400）时不打 INFO，
    # 避免 30s 一次的 /health 把访问日志刷爆；非 2xx/异常仍照常记录以便排障。
    ACCESS_LOG_SKIP_PATHS: set[str] = {"/health"}
    # 启用的检测场景白名单（按 detector_type，如 panel_label / dc_fuse）。
    # 留空 = 全部启用（向后兼容）；指定后仅注册并预加载列表内场景，未列出的场景
    # 路由不注册、模型不预加载——便于单场景部署（如服务器只上线 panel_label），
    # 避免缺失其它场景权重导致启动报错或刷无关日志。基础路由（如 /stats）不受影响。
    ENABLED_SCENES: set[str] = set()
    WORKERS: int = 1
    # 开发模式热重载；生产应保持 False（reload=True 时 WORKERS 会被 uvicorn 忽略）
    RELOAD: bool = False
    # 上传图片大小上限（MB），超出直接拒绝，避免大文件占满内存
    MAX_UPLOAD_MB: int = 20
    # 严格启动：任一检测器预加载失败则拒绝启动（生产建议 True，避免带病运行、端点静默缺失）
    STRICT_STARTUP: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
