'''
@Author       : gongzhang4
@Date         : 2026-01-27 02:06:28
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-04-01 03:15:56
@FilePath     : yolo.py
@Description  :
'''

import numpy as np

from .base import BaseVisionInfer
from .inference import InferenceRunner
from .yolo_ops import (
    prepare_yolo_input,
    restore_yolo_boxes,
    run_yolo_nms,
)
from .vision.boxes import xywhr2xyxyxyxy
from .vision.masks import masks2segments_with_boxes, process_mask, scale_masks
from collections import defaultdict
from schemas.data_base import DetectResult
from schemas.inference_context import PreprocMeta
import time
from utils import vision_logger


class YoloInfer(BaseVisionInfer):
    def __init__(
        self,
        nc,
        runner: InferenceRunner,
        confThreshold=0.5,
        nmsThreshold=0.5,
        task="det",
    ):
        super().__init__(
            runner,
            confThreshold=confThreshold,
            nmsThreshold=nmsThreshold,
        )
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.agnostic = False
        self.nc = nc
        self.filter_classes = None
        self.task = task

    def preprocess(self, im):
        """预处理输入图像。

        Returns:
            tuple: (模型输入张量, PreprocMeta)
        """
        return prepare_yolo_input(im, self._input_model_shape[2:])

    def post_process(self, preds, meta):
        """后处理输出（无状态：缩放/原图信息来自 meta）"""
        p = run_yolo_nms(
            preds[0],
            task=self.task,
            conf_threshold=self.confThreshold,
            iou_threshold=self.nmsThreshold,
            classes=self.filter_classes,
            agnostic=self.agnostic,
            nc=self.nc,
        )
        image_shape = meta.src_shape[:2]
        input_shape = self.input_model_shape[2:]
        pred = restore_yolo_boxes(p[0], input_shape, meta.src_shape)
        masks = []
        mask_polygons = []
        if self.task == "seg":
            protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
            mask_in = p[0][:, 6:]
            bboxes = p[0][:, :4]
            start = time.time()
            masks = process_mask(protos, mask_in, bboxes, input_shape)
            end = time.time()
            vision_logger.debug("process_mask: {:.4f}秒", end - start)
            start = time.time()
            if len(masks) != 0:
                masks = scale_masks(
                    masks, (image_shape[1], image_shape[0]), meta.r, meta.dw, meta.dh
                ).transpose(2, 0, 1)
            end = time.time()
            vision_logger.debug("scale_masks: {:.4f}秒", end - start)
            start = time.time()
            # 逐检测提取掩膜多边形：轮廓退化（空掩膜/面积过小）的检测单独丢弃，
            # 并同步剔除其 pred/masks 行，保持三者严格 1:1 对齐——
            # 不能因个别退化掩膜就 return 作废整帧检测。
            mask_polygons = []
            keep = []
            for mask, box in zip(masks, pred[:, :4]):
                segs = masks2segments_with_boxes(mask, box)
                if segs:
                    mask_polygons.append(segs[0])
                    keep.append(True)
                else:
                    keep.append(False)
            keep = np.array(keep, dtype=bool)
            if not keep.all():
                vision_logger.warning(
                    f"丢弃 {int((~keep).sum())} 个掩膜退化的检测（共 {len(keep)} 个），保留 {int(keep.sum())} 个"
                )
                pred = pred[keep]
                masks = masks[keep]
            end = time.time()
            vision_logger.debug("masks2segments: {:.4f}秒", end - start)

        pred = np.concatenate([pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1)
        bbox = pred[:, :4]  # xywh
        if self.task == "obb":
            bbox = xywhr2xyxyxyxy(pred[:, :5])
        detect_result = DetectResult(
            bbox.tolist(),
            pred[:, -2].tolist(),
            pred[:, -1].tolist(),
            [self.id2name[int(cls)] for cls in pred[:, -1]],
            masks=masks if self.task == "seg" else [],
            mask_polygons=mask_polygons if self.task == "seg" else [],
            ori_img=meta.ori_img,
        )
        return detect_result
