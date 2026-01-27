'''
@Author       : gongzhang4
@Date         : 2026-01-27 07:19:30
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 08:42:54
@FilePath     : indicator_light_config.py
@Description  :
'''


class IndicatorLightConfig:
    class ModelPath:
        det_model_path: str = "./weights/IndicatorLightDet_v2.onnx"
        rec_model_path: str = "./weights/IndicatorLightRec_v2.onnx"

    class ConfThreshold:
        det: float = 0.25
        rec: float = 0.25

    JSON_PATH: str = "weights/jsons/standard_embeddings.json"
    SIM_THR: float = 0.7
    IS_CACHE: bool = True
