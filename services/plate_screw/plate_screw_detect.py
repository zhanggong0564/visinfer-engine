'''
@Author       : gongzhang4
@Date         : 2026-01-29 12:18:12
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-29 12:29:21
@FilePath     : plate_screw_detect.py
@Description  :
'''

from utils import vision_logger
from ..yolo import YoloOnnxInfer


class PlateScrewDetect(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, 4, confThreshold, nmsThreshold, task)
        self.id2name = {
            0: "metal_plate_7",
            1: "metal_screw_5",
            2: "no_metal_plate_7",
            3: "no_metal_screw_5",
        }
