'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:38:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-16 02:51:46
@FilePath     : yolo.py
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
from typing import List, Tuple

onnxruntime.set_default_logger_severity(3)


class IndicatorLightDet(BaseOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, providers=None):
        super().__init__(model_path, confThreshold, nmsThreshold, providers)
        self.task = "det"
        self.filter_classes = None
        self.agnostic = False
        self.nc = 1

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
        boxes = []
        for box, score in zip(bbox.tolist(), conf.tolist()):
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append([x1, y1, x2, y2, score])
        sorted_boxes = sort_boxes(boxes)  # 按x1坐标排序
        return sorted_boxes


class IndicatorLightRecognition(BaseOnnxInfer):
    """图像分类模型实现示例"""

    def __init__(self, model_path: str, img_size: Tuple[int, int] = (224, 224), providers=None):
        super().__init__(model_path, providers=providers)
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, im: np.ndarray) -> np.ndarray:
        # 1. 读取图像
        # img = cv2.imread(image_path)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # 2. 调整大小并归一化
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # 3. 转换维度格式 [H, W, C] -> [C, H, W] -> [1, C, H, W]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # 返回符合模型输入格式的数据
        return img

    def post_process(self, output_data: List[np.ndarray]) -> np.ndarray:
        # 获取第一个输出（假设是分类概率）
        embedding = output_data[0]
        return embedding
