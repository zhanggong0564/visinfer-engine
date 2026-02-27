'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:20:56
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-27 03:19:39
@FilePath     : panel_label_detect.py
@Description  : 面板标签检测
'''

from ..yolo import YoloOnnxInfer
from ..utils import *
import numpy as np
from schemas.data_base import DetectResult


class PanelLabelDetect(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="seg"):
        super().__init__(model_path, 2, confThreshold, nmsThreshold, task)
        self.id2name = {
            0: "line",
            1: "QFU",
        }
