"""Segmentation-mask operations."""

import cv2
import numpy as np


def _restore_hwc_mask_stack(masks: np.ndarray) -> np.ndarray:
    """Keep OpenCV single-channel resize results in HWC mask-stack form."""
    if masks.ndim == 2:
        return masks[..., None]
    return masks


def process_mask(protos, masks_in, bboxes, shape):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (np.ndarray): A tensor of shape [batch, mask_dim, mask_h, mask_w].
        masks_in (np.ndarray): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (np.ndarray): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).

    Returns:
        np.ndarray: Binary mask stack in HWC layout, shape [h, w, n].
    """
    if len(masks_in) == 0:
        return masks_in
    _, c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    # chw->hwc
    masks = masks.transpose(1, 2, 0)
    masks = cv2.resize(masks, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    masks = _restore_hwc_mask_stack(masks)
    masks = (sigmoid(masks) > 0.9).astype(np.uint8) * 255
    return masks


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks (torch.Tensor): [n, h, w] tensor of masks
      boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
      (torch.Tensor): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_masks(masks, shape, gain, dw, dh):
    """
    Rescale segment masks from letterboxed input size to original image size.

    Args:
        masks (np.ndarray): Binary mask stack in HWC layout, shape [h, w, n].
        shape (tuple): Output size in OpenCV order, (width, height).
        gain (float): Letterbox scale ratio.
        dw (float): Letterbox x padding.
        dh (float): Letterbox y padding.

    Returns:
        np.ndarray: Rescaled mask stack in HWC layout, shape [height, width, n].
    """
    if len(masks) == 0:
        return masks
    if masks.ndim != 3:
        raise ValueError(f"scale_masks expects HWC masks with shape [h, w, n], got {masks.shape}")
    mh, mw, _ = masks.shape
    pad = (dw, dh)
    top, left = (int(pad[1]), int(pad[0]))  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, shape)
    masks = _restore_hwc_mask_stack(masks)
    return masks




def sigmoid(x):
    return 1 / (1 + np.exp(-x))




def masks2segments_with_boxes(
    mask,
    box,
    min_area=4000,
    chain_mode="simple",
    topk=1,  # 新增：返回前k个最大轮廓，默认1=只取最大
    area_keep_ratio=0.05,  # 新增：动态阈值=最大面积*ratio
):
    """
    mask: [H, W]，uint8/bool，值可为0/1或0/255
    box:  [x1, y1, x2, y2]，原图坐标
    min_area: 绝对面积下限（像素）
    chain_mode: "simple" 或 "tc89"
    topk: 返回面积最大的前k个轮廓（1表示只保留最大轮廓）
    area_keep_ratio: 动态面积阈值比例，相对于最大轮廓面积

    return:
        segments: List[np.ndarray(K,2)]，float32，原图坐标
    """
    H, W = mask.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # 1) clip box 到图像边界
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return []

    # 2) ROI裁剪
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    # 3) 转 uint8 二值
    if roi.dtype != np.uint8:
        roi = roi.astype(np.uint8)
    if roi.max() == 1:
        roi = roi * 255

    if cv2.countNonZero(roi) == 0:
        return []

    # 4) 轮廓提取
    cm = cv2.CHAIN_APPROX_TC89_KCOS if chain_mode == "tc89" else cv2.CHAIN_APPROX_SIMPLE
    contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cm)[0]
    if len(contours) == 0:
        return []

    # 5) 动态选取：按面积排序，保留最大的 topk，且做动态面积过滤
    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    order = np.argsort(-areas)  # 降序
    max_area = float(areas[order[0]])

    # 动态阈值：绝对阈值 与 相对最大面积阈值 取更严格者
    dynamic_thr = max(float(min_area), max_area * float(area_keep_ratio))

    segments = []
    kept = 0
    for idx in order:
        a = float(areas[idx])
        if a < dynamic_thr:
            continue

        c = contours[idx]
        pts = c.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += x1
        pts[:, 1] += y1
        segments.append(pts)

        kept += 1
        if topk is not None and topk > 0 and kept >= topk:
            break

    return segments
