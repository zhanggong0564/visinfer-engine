'''
@Author       : gongzhang4
@Date         : 2026-01-13 05:04:15
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-13 06:36:05
@FilePath     : yolo.py
@Description  :
'''

'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:38:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 02:35:56
@FilePath     : yolov11.py
@Description  :
'''

import sys
from collections import defaultdict
import os

sys.path.append(os.getcwd())
import numpy as np
import onnxruntime
from utils import vision_logger
from ..utils import *
from ..box import non_max_suppression_v8
from ..base import BaseOnnxInfer

onnxruntime.set_default_logger_severity(3)


class yolo11ONNX(BaseOnnxInfer):
    def __init__(self, model_path, nc, confThreshold=0.5, nmsThreshold=0.5, providers=None, task="det"):
        super().__init__(model_path, confThreshold, nmsThreshold, providers)
        self.task = task
        self.filter_classes = None
        self.agnostic = False
        self.nc = nc
        self.id2class = {
            0: "metal_plate_7",
            1: "metal_screw_5",
            2: "no_metal_plate_7",
            3: "no_metal_screw_5",
        }

    def post_process(self, preds):
        p = non_max_suppression_v8(
            preds[0],
            task=self.task,
            conf_thres=self.confThreshold,
            iou_thres=self.nmsThreshold,
            classes=self.filter_classes,
            agnostic=self.agnostic,
            multi_label=False,
            nc=self.nc,
        )
        image_shape = self.image_src_shape[:2]
        input_shape = self.input_model_shape[2:]
        res = defaultdict(list)
        pred = p[0]
        pred[:, :4] = scale_boxes(input_shape, pred[:, :4], image_shape, xywh=False)
        pred = np.concatenate([pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1)
        bbox = pred[:, :4]  # xywh
        if self.task == "obb":
            bbox = xywhr2xyxyxyxy(bbox)
        # else:
        #     bbox = xywh2xyxy(bbox)
        conf = pred[:, -2]
        clas = pred[:, -1]
        for box, cls, score in zip(bbox, clas, conf):
            info = []
            # 归一化
            box = box / np.array(image_shape[::-1] * 2)
            info.append(box.tolist())
            info.append(score)
            res[self.id2class[int(cls)]].append(info)
        return res
