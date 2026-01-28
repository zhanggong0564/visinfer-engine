'''
@Author       : gongzhang4
@Date         : 2026-01-23 06:35:58
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-28 02:52:51
@FilePath     : config.py
@Description  :
'''

import os
from pydantic_settings import BaseSettings
from .dc_fuse_confg import DcFuseConfig
from .indicator_light_config import IndicatorLightConfig
from .lap_surf_conf import LapSufConfig
from .line_squeeze_config import LineSqueezeConfig


class PlateScrewConfig:
    model_path: str = "./weights/mobile_vision_plate_v2.onnx"
    confThreshold: float = 0.25


class Settings(BaseSettings):
    API_TITLE: str = "Mobile Vision alg API"
    API_VERSION: str = "1.1.2"

    HOST: str = "0.0.0.0"
    PORT: int = 3007

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    dc_fuse: DcFuseConfig = DcFuseConfig()
    lap_surf: LapSufConfig = LapSufConfig()
    plate_screw: PlateScrewConfig = PlateScrewConfig()
    indicator_light: IndicatorLightConfig = IndicatorLightConfig()
    line_squeeze: LineSqueezeConfig = LineSqueezeConfig()

    WORKERS: int = 1

    class Config:
        env_file = ".env"


settings = Settings()
