"""RF-DETR 分割模型推理适配器。"""

from typing import Literal

import cv2
import numpy as np

from schemas.data_base import DetectResult
from schemas.inference_context import PreprocMeta
from utils import vision_logger

from .base import BaseVisionInfer
from .inference import InferenceRunner
from .vision.masks import masks2segments_with_boxes


class RFDetrInfer(BaseVisionInfer):
    """适配固定输入尺寸 RF-DETR 检测/分割模型。"""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        nc,
        runner: InferenceRunner,
        confThreshold=0.5,
        task="seg",
        mask_output: Literal["full", "polygons_only"] = "full",
    ):
        super().__init__(
            runner,
            confThreshold=confThreshold,
        )
        self.nc = nc
        self.task = task
        if mask_output not in ("full", "polygons_only"):
            raise ValueError("mask_output must be 'full' or 'polygons_only'")
        self.mask_output = mask_output

    def preprocess(self, im):
        """将 BGR 图像缩放、归一化为 RF-DETR 所需的 RGB NCHW 张量。"""
        if im.ndim != 3 or im.shape[2] != 3:
            raise ValueError(f"RF-DETR expects a 3-channel BGR image, got {im.shape}")

        input_h, input_w = self.input_model_shape[2:]
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self._MEAN) / self._STD
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        meta = PreprocMeta(r=1.0, dw=0.0, dh=0.0, src_shape=im.shape)
        return tensor, meta

    def post_process(self, outputs, meta):
        """解码 RF-DETR boxes、类别 logits 与分割 mask logits。"""
        mask_output = getattr(self, "mask_output", "full")
        if mask_output not in ("full", "polygons_only"):
            raise ValueError("mask_output must be 'full' or 'polygons_only'")
        if len(outputs) != 3:
            raise ValueError(f"RF-DETR expects 3 outputs, got {len(outputs)}")

        dets, labels, mask_logits = outputs
        if (
            dets.ndim != 3
            or dets.shape[0] != 1
            or dets.shape[2] != 4
            or labels.ndim != 3
            or labels.shape[:2] != dets.shape[:2]
            or labels.shape[2] < self.nc
            or mask_logits.ndim != 4
            or mask_logits.shape[:2] != dets.shape[:2]
        ):
            raise ValueError(
                "RF-DETR output shapes must be dets=[1,N,4], labels=[1,N,C], "
                "masks=[1,N,H,W]"
            )

        foreground_logits = np.clip(labels[0, :, :self.nc], -88.0, 88.0)
        foreground_scores = 1.0 / (1.0 + np.exp(-foreground_logits))
        scores = foreground_scores.max(axis=1)
        class_ids = foreground_scores.argmax(axis=1)
        keep = scores > self.confThreshold

        source_h, source_w = meta.src_shape[:2]
        selected_boxes = dets[0, keep]
        selected_scores = scores[keep]
        selected_class_ids = class_ids[keep]
        selected_masks = mask_logits[0, keep]

        boxes = self._restore_boxes(selected_boxes, source_w, source_h)
        valid_boxes = []
        valid_scores = []
        valid_class_ids = []
        valid_masks = []
        mask_polygons = []
        for box, score, class_id, raw_mask in zip(
            boxes, selected_scores, selected_class_ids, selected_masks
        ):
            binary_mask = None
            if mask_output == "polygons_only":
                segments = self._polygon_from_box_mask(
                    raw_mask, box, source_w, source_h
                )
            else:
                segments, binary_mask = self._full_mask_polygon(
                    raw_mask, box, source_w, source_h
                )
            if not segments and mask_output == "polygons_only":
                self._warn_mask_fallback_once()
                segments, _ = self._full_mask_polygon(
                    raw_mask, box, source_w, source_h
                )
            if not segments:
                continue
            valid_boxes.append(box.tolist())
            valid_scores.append(float(score))
            valid_class_ids.append(int(class_id))
            if binary_mask is not None:
                valid_masks.append(binary_mask)
            mask_polygons.append(segments[0])

        return DetectResult(
            boxes=valid_boxes,
            scores=valid_scores,
            class_ids=valid_class_ids,
            class_names=[self.id2name[class_id] for class_id in valid_class_ids],
            masks=valid_masks,
            mask_polygons=mask_polygons,
            ori_img=meta.ori_img,
        )

    @staticmethod
    def _full_mask_polygon(raw_mask, box, source_w, source_h):
        mask = cv2.resize(
            raw_mask, (source_w, source_h), interpolation=cv2.INTER_LINEAR
        )
        binary_mask = (mask > 0.0).astype(np.uint8) * 255
        return masks2segments_with_boxes(binary_mask, box), binary_mask

    def _warn_mask_fallback_once(self):
        if getattr(self, "_mask_fallback_warned", False):
            return
        self._mask_fallback_warned = True
        vision_logger.warning(
            "RF-DETR 局部 mask 后处理失败，当前及后续异常候选回退完整 mask 路径"
        )

    @staticmethod
    def _polygon_from_box_mask(raw_mask, box, source_w, source_h):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(source_w, x1))
        x2 = max(0, min(source_w, x2))
        y1 = max(0, min(source_h, y1))
        y2 = max(0, min(source_h, y2))
        if x2 <= x1 or y2 <= y1:
            return []

        mask_roi = RFDetrInfer._resize_mask_roi(
            raw_mask, (x1, y1, x2, y2), source_w, source_h
        )
        binary_roi = (mask_roi > 0.0).astype(np.uint8) * 255
        segments = masks2segments_with_boxes(
            binary_roi, (0, 0, x2 - x1, y2 - y1)
        )
        for segment in segments:
            segment[:, 0] += x1
            segment[:, 1] += y1
        return segments

    @staticmethod
    def _resize_mask_roi(raw_mask, box, source_w, source_h):
        """Sample one destination ROI with cv2.resize half-pixel geometry."""
        x1, y1, x2, y2 = map(int, box)
        mask_h, mask_w = raw_mask.shape[:2]
        xs = (
            (np.arange(x1, x2, dtype=np.float32) + 0.5)
            * (mask_w / source_w)
            - 0.5
        )
        ys = (
            (np.arange(y1, y2, dtype=np.float32) + 0.5)
            * (mask_h / source_h)
            - 0.5
        )
        x_floor = np.floor(xs).astype(np.int32)
        y_floor = np.floor(ys).astype(np.int32)
        x_weight = (xs - x_floor).astype(np.float32)
        y_weight = (ys - y_floor).astype(np.float32)
        x0 = np.clip(x_floor, 0, mask_w - 1)
        x_next = np.clip(x_floor + 1, 0, mask_w - 1)
        y0 = np.clip(y_floor, 0, mask_h - 1)
        y_next = np.clip(y_floor + 1, 0, mask_h - 1)
        top = (
            raw_mask[y0[:, None], x0[None, :]]
            * (1.0 - x_weight)[None, :]
            + raw_mask[y0[:, None], x_next[None, :]]
            * x_weight[None, :]
        )
        bottom = (
            raw_mask[y_next[:, None], x0[None, :]]
            * (1.0 - x_weight)[None, :]
            + raw_mask[y_next[:, None], x_next[None, :]]
            * x_weight[None, :]
        )
        return (
            top * (1.0 - y_weight)[:, None]
            + bottom * y_weight[:, None]
        )

    @staticmethod
    def _restore_boxes(boxes, source_w, source_h):
        """将归一化 cxcywh 检测框转为裁剪后的原图 xyxy 坐标。"""
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        cx, cy, width, height = boxes.T
        restored = np.stack(
            (
                (cx - width / 2) * source_w,
                (cy - height / 2) * source_h,
                (cx + width / 2) * source_w,
                (cy + height / 2) * source_h,
            ),
            axis=1,
        ).astype(np.float32)
        restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, source_w)
        restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, source_h)
        return restored
