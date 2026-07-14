"""RF-DETR ONNX 分割模型推理适配器。"""

import cv2
import numpy as np

from schemas.data_base import DetectResult
from schemas.inference_context import PreprocMeta

from .base import BaseOnnxInfer
from .utils.utils import masks2segments_with_boxes


class RFDetrOnnxInfer(BaseOnnxInfer):
    """适配固定输入尺寸 RF-DETR ONNX 检测/分割模型。"""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path, nc, confThreshold=0.5, task="seg"):
        super().__init__(model_path, confThreshold=confThreshold)
        self.nc = nc
        self.task = task

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
            mask = cv2.resize(
                raw_mask, (source_w, source_h), interpolation=cv2.INTER_LINEAR
            )
            binary_mask = (mask > 0.0).astype(np.uint8) * 255
            segments = masks2segments_with_boxes(binary_mask, box)
            if not segments:
                continue
            valid_boxes.append(box.tolist())
            valid_scores.append(float(score))
            valid_class_ids.append(int(class_id))
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
