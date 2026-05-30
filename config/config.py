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
    API_VERSION: str = "1.1.2"

    HOST: str = "0.0.0.0"
    PORT: int = 3001

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
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
