'''
@Author       : gongzhang4
@Date         : 2026-01-23 06:35:58
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-07 08:22:09
@FilePath     : config.py
@Description  :
'''

from pydantic_settings import BaseSettings
from .plate_screw_congfig import PlateScrewConfig


class Settings(BaseSettings):
    API_TITLE: str = "Mobile Vision alg API"
    API_VERSION: str = "1.1.2"

    HOST: str = "0.0.0.0"
    PORT: int = 3007

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    plate_screw: PlateScrewConfig = PlateScrewConfig()
    WORKERS: int = 1

    class Config:
        env_file = ".env"


settings = Settings()
