'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:38:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-03 03:01:58
@FilePath     : indicator_light_det.py
@Description  :
'''

from collections import defaultdict
import numpy as np
from utils import vision_logger
from typing import List, Tuple

from ..yolo import YoloOnnxInfer
from ..base import BaseOnnxInfer
import cv2
from ..utils import sort_boxes
from ..data_base import IndicatorLightEmbedding


class IndicatorLightDet(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, nc=1, confThreshold=confThreshold, nmsThreshold=nmsThreshold, task=task)
        self.id2name = {0: "roi"}


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


class IndicatorLightDetRec:
    def __init__(self, det_model_path, rec_model_path, confThreshold=0.5, nmsThreshold=0.5):
        self.det = IndicatorLightDet(det_model_path, confThreshold, nmsThreshold)
        self.rec = IndicatorLightRecognition(rec_model_path)

    def infer(self, image: np.ndarray):
        # DetectResult
        det_result = self.det.infer(image)
        h, w, _ = image.shape
        boxes = []
        for box, score in zip(det_result.boxes, det_result.scores):
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append([x1, y1, x2, y2, score])
        sorted_boxes = np.array(sort_boxes(boxes)[0])  # 按x1坐标排序
        embeddings = []
        for box in sorted_boxes:
            x_min, y_min, x_max, y_max, score = box
            roi = image[
                max(int(y_min - 10), 0) : min(int(y_max + 10), h),
                max(int(x_min - 10), 0) : min(int(x_max + 10), w),
            ]
            embedding = self.rec.infer(roi)
            embeddings.append(embedding.tolist())
        indicator_light_embedding = IndicatorLightEmbedding(
            embeddings=embeddings, boxes=sorted_boxes[:, :4].tolist(), scores=sorted_boxes[:, 4].tolist()
        )
        return indicator_light_embedding
