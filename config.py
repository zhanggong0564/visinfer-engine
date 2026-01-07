'''
@Author       : gongzhang4
@Date         : 2026-01-07 05:45:41
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 06:35:15
@FilePath     : config.py
@Description  :
'''

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_TITLE: str = "Mobile Vision alg API"
    API_VERSION: str = "v1.0.0"

    LOG_DIR: str = "logs"

    class Config:
        env_file = ".env"


settings = Settings()
