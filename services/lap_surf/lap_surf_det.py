'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:38:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 09:48:18
@FilePath     : lap_surf_det.py
@Description  :
'''

import sys
from collections import defaultdict
import os

sys.path.append(os.getcwd())
import numpy as np
from utils import vision_logger
from ..yolo import YoloOnnxInfer


class LapSurfDetONNX(YoloOnnxInfer):
    def __init__(self, model_path, nc, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, nc, confThreshold, nmsThreshold, task)
        self.id2name = {0: "roi", 1: "螺丝", 2: "螺母", 3: "搭接面"}
