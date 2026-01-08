'''
@Author       : gongzhang4
@Date         : 2026-01-07 05:45:41
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 08:59:54
@FilePath     : config.py
@Description  :
'''

import os
from pydantic_settings import BaseSettings


class DcFuseConfig:
    model_path: str = "./weights/dc_fuse_v5.onnx"
    confThreshold: float = 0.6


class LapSufConfig:
    model_path: str = "./weights/LapJointSurfRec_v2.onnx"
    confThreshold: float = 0.4


class Settings(BaseSettings):
    API_TITLE: str = "Mobile Vision alg API"
    API_VERSION: str = "1.0.0"

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    dc_fuse: DcFuseConfig = DcFuseConfig()
    lap_surf: LapSufConfig = LapSufConfig()

    class Config:
        env_file = ".env"


settings = Settings()
