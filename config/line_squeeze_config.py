'''
@Author       : gongzhang4
@Date         : 2026-01-28 02:49:24
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-28 02:51:48
@FilePath     : line_squeeze_config.py
@Description  :
'''


class LineSqueezeConfig:
    class ModelPath:
        det_model_path: str = "./weights/LineSqueeze_v3.onnx"
        oct_model_path: str = "./weights/official_models"

    class ConfThreshold:
        det: float = 0.25
        oct: float = 0.25
