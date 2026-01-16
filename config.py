'''
@Author       : gongzhang4
@Date         : 2026-01-07 05:45:41
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-16 05:02:57
@FilePath     : config.py
@Description  :
'''

import os
from pydantic_settings import BaseSettings


class IndicatorLightConfig:
    class ModelPath:
        det_model_path: str = "./weights/IndicatorLightDet_v2.onnx"
        rec_model_path: str = "./weights/IndicatorLightRec_v2.onnx"

    class ConfThreshold:
        det: float = 0.25
        rec: float = 0.25

    JSON_PATH: str = "weights/jsons/standard_embeddings.json"
    SIM_THR: float = 0.7


class PlateScrewConfig:
    model_path: str = "./weights/mobile_vision_plate_v2.onnx"
    confThreshold: float = 0.25


class DcFuseConfig:
    model_path: str = "./weights/dc_fuse_v5.onnx"
    confThreshold: float = 0.6


class LapSufConfig:
    model_path: str = "./weights/LapJointSurfRec_v2.onnx"
    confThreshold: float = 0.4


class Settings(BaseSettings):
    API_TITLE: str = "Mobile Vision alg API"
    API_VERSION: str = "1.0.0"

    HOST: str = "0.0.0.0"
    PORT: int = 3007

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    dc_fuse: DcFuseConfig = DcFuseConfig()
    lap_surf: LapSufConfig = LapSufConfig()
    plate_screw: PlateScrewConfig = PlateScrewConfig()
    indicator_light: IndicatorLightConfig = IndicatorLightConfig()

    class Config:
        env_file = ".env"


settings = Settings()
