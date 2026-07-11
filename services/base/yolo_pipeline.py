from typing import Sequence

import numpy as np

from schemas.inference_context import PreprocMeta
from services.utils import letterbox, scale_boxes
from services.utils.box import non_max_suppression_v8


def prepare_yolo_input(
    image: np.ndarray, input_shape: Sequence[int]
) -> tuple[np.ndarray, PreprocMeta]:
    resized, r, dw, dh = letterbox(
        im=image, auto=False, new_shape=input_shape
    )
    tensor = np.stack([resized])
    tensor = tensor[..., ::-1].transpose((0, 3, 1, 2))
    tensor = np.ascontiguousarray(tensor).astype(np.float32)
    tensor /= 255.0
    return tensor, PreprocMeta(r=r, dw=dw, dh=dh, src_shape=image.shape)


def run_yolo_nms(
    prediction,
    *,
    task: str,
    conf_threshold: float,
    iou_threshold: float,
    classes,
    agnostic: bool,
    nc: int,
):
    return non_max_suppression_v8(
        prediction,
        task=task,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        classes=classes,
        agnostic=agnostic,
        multi_label=False,
        nc=nc,
    )


def restore_yolo_boxes(
    detections: np.ndarray,
    input_shape: Sequence[int],
    src_shape: Sequence[int],
) -> np.ndarray:
    restored = detections.copy()
    restored[:, :4] = scale_boxes(
        input_shape, restored[:, :4], src_shape[:2], xywh=False
    )
    return restored
