'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:38:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 06:38:08
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
from .utils import *
from .box import non_max_suppression_v8
from .base import BaseOnnxInfer

onnxruntime.set_default_logger_severity(3)


class yolo11ONNX(BaseOnnxInfer):
    def __init__(
        self, model_path, nc, confThreshold=0.5, nmsThreshold=0.5, providers=None, input_shape=(1024, 1024), task="det"
    ):
        super().__init__(model_path, confThreshold, nmsThreshold, providers)
        self.task = task
        self.filter_classes = None
        self.agnostic = False
        self.nc = nc
        self.input_shape = input_shape

    def preprocess(self, im):
        """预处理输入图像

        Args:
            im (np.ndarray): 输入图像

        Returns:
            np.ndarray: 处理后的图像
        """
        img, self.r, self.dw, self.dh = letterbox(im=im, auto=False, new_shape=self.input_shape)
        im = np.stack([img])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im).astype(np.float32)
        im /= 255.0  # 归一化到0-1
        return im

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
        res = defaultdict()
        pred = p[0]
        pred[:, :4] = scale_boxes(input_shape, pred[:, :4], image_shape, xywh=False)
        pred = np.concatenate([pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1)
        bbox = pred[:, :5]  # xywh
        if self.task == "obb":
            bbox = xywhr2xyxyxyxy(bbox)
        # else:
        #     bbox = xywh2xyxy(bbox)
        conf = pred[:, -2]
        clas = pred[:, -1]
        res["rect"] = bbox.tolist()
        res["score"] = conf.tolist()
        res["cls"] = clas.tolist()
        return res
