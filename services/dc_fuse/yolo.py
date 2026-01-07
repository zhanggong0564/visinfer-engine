'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:48:03
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 07:20:17
@FilePath     : yolo.py
@Description  :
'''

from ..base import BaseOnnxInfer
from ..utils import *
from ..box import non_max_suppression_v8
from collections import defaultdict
from utils import vision_logger


class DCFuseDetector(BaseOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, providers=None, task="det"):
        super().__init__(model_path, confThreshold, nmsThreshold, providers)
        self.task = task
        self.filter_classes = None
        self.agnostic = False
        self.nc = 12
        self.id2name = {
            0: "brass_plate_6",
            1: "lower_crossbeam_screw_10",
            2: "metal_piece_4",
            3: "no_lower_crossbeam_screw_10",
            4: "no_nut2",
            5: "no_screw_1",
            6: "no_small_screw_8",
            7: "no_upper_crossbeam_screw_9",
            8: "nut_2",
            9: "screw_1",
            10: "small_screw_8",
            11: "upper_crossbeam_screw_9",
        }

    def post_process(self, preds):
        """后处理输出"""
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
        res["cls_name"] = [self.id2name[int(cls)] for cls in clas]
        return res
