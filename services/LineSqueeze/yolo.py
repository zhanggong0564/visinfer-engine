'''
@Author       : gongzhang4
@Date         : 2026-01-17 06:31:48
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-17 06:44:26
@FilePath     : yolo.py
@Description  :
'''

from ..base import BaseOnnxInfer, letterbox, scale_boxes, xywhr2xyxyxyxy
from ..box import non_max_suppression_v8
from collections import defaultdict
import numpy as np


class RoiDet(BaseOnnxInfer):
    def __init__(self, model_path, nc, input_model_shape, confThreshold=0.5, nmsThreshold=0.5, providers=None):
        super().__init__(model_path, confThreshold, nmsThreshold, providers)
        self.task = "rect"
        self.filter_classes = None
        self.agnostic = False
        self.nc = nc

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
        bbox = pred[:, :4]  # xywh
        if self.task == "obb":
            bbox = xywhr2xyxyxyxy(bbox)
        # else:
        #     bbox = xywh2xyxy(bbox)
        conf = pred[:, -2]
        clas = pred[:, -1]
        res["rect"] = bbox.tolist()
        res["score"] = conf.tolist()
        res["cls"] = clas.tolist()
        # 根据分数过滤
        res["rect"] = [box for box, score in zip(res["rect"], res["score"]) if score >= self.confThreshold]
        res["score"] = [score for score in res["score"] if score >= self.confThreshold]
        res["cls"] = [cls for cls, score in zip(res["cls"], res["score"]) if score >= self.confThreshold]
        return res
