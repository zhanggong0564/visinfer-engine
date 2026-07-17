'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:39:34
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 06:39:35
@FilePath     : box.py
@Description  :
'''

import numpy as np
from .boxes import xywh2xyxy


def non_max_suppression_v8(
    prediction,
    task="det",
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, \
        with support for masks and multiple labels per box.

    Arguments:
        prediction (np.array):
            A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks.
            The tensor should be in the format output by a model, such as YOLO.
        task: `det` | `seg` | `track` | `obb`
        conf_thres (float):
            The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float):
            The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider.
            If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes,
            and all classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, np.array]]]):
            A list of lists, where each inner list contains the apriori labels \
            for a given image. The list should be in the format output by a dataloader, \
            with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. \
            Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into numpy_nms.
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[np.array]):
            A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes,
            with columns (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, \
        valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, \
        valid values are between 0.0 and 1.0"
    if task == "seg" and nc == 0:
        raise ValueError("The value of nc must be set when the mode is 'seg'.")
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output
    bs = prediction.shape[0]  # batch size
    if task in ["det", "track"]:
        nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    # shape(1,84,6300) to shape(1,6300,84)
    prediction = np.transpose(prediction, (0, 2, 1))
    if task != "obb":
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) |
        # (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        if labels and len(labels[xi]) and task != "obb":
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            # x 已在上面按 xc[xi] 过滤过，这里直接拼先验标签，不能再用整张 xc 二次索引
            x = np.concatenate((x, v), axis=0)

        if not x.shape[0]:
            continue

        box = x[:, :4]
        cls = x[:, 4 : 4 + nc]
        mask = x[:, 4 + nc : 4 + nc + nm]

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate(
                (box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]),
                axis=1,
            )
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.flatten() > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        if task == "obb":
            boxes = np.concatenate((x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1)  # xywhr
            i = numpy_nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            i = numpy_nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return output


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])



def numpy_nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    areas = box_area(boxes)
    eps = np.finfo(np.float32).eps

    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        left_top = np.maximum(boxes[current, :2], boxes[remaining, :2])
        right_bottom = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        wh = np.maximum(right_bottom - left_top, 0)
        intersection = wh[:, 0] * wh[:, 1]
        union = areas[current] + areas[remaining] - intersection
        iou = intersection / np.maximum(union, eps)
        order = remaining[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def numpy_nms_rotated(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)

    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.triu(ious, k=1)
    pick = np.nonzero(np.max(ious, axis=0) < iou_threshold)[0]
    return sorted_idx[pick]


def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        ((a1 + a2) * (np.power(y1 - y2, 2)) + (b1 + b2) * (np.power(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)) * 0.5

    t3 = (
        np.log(
            ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)))
            / (4 * np.sqrt((a1 * b1 - np.power(c1, 2)).clip(0) * (a2 * b2 - np.power(c2, 2)).clip(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = np.clip(bd, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def _get_covariance_matrix(boxes):
    gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    return (
        a * np.cos(c) ** 2 + b * np.sin(c) ** 2,
        a * np.sin(c) ** 2 + b * np.cos(c) ** 2,
        a * np.cos(c) * np.sin(c) - b * np.sin(c) * np.cos(c),
    )
