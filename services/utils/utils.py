'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:20:56
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-02 07:52:57
@FilePath     : utils.py
@Description  :
'''

import cv2
import random
import numpy as np
import base64
import numpy as np
from typing import List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor


def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (np.ndarray): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (np.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.
      padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
        rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
      boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0]] -= pad[0]  # x padding
        boxes[..., [1]] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, return_int=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, dw, dh


def scale_coords(coords, gain, dw, dh, target_shape):
    if len(coords) == 0:
        return coords
    pad = (dw, dh, dw, dh)
    coords -= pad  # x padding
    coords /= gain
    coords[:, 0] = coords[:, 0].clip(0, target_shape[1])  # x1
    coords[:, 1] = coords[:, 1].clip(0, target_shape[0])  # y1
    coords[:, 2] = coords[:, 2].clip(0, target_shape[1])  # x2
    coords[:, 3] = coords[:, 3].clip(0, target_shape[0])  # y2
    return coords


def visualize(img, bbox_array, scores, labels):
    for temp, s, l in zip(bbox_array, scores, labels):
        xmin = int(temp[0])
        ymin = int(temp[1])
        xmax = int(temp[2])
        ymax = int(temp[3])
        clas = int(l)
        score = s
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
        img = cv2.putText(
            img,
            "class:" + str(clas) + " " + str(round(score, 2)),
            (xmin, int(ymin) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (105, 237, 249),
            1,
        )
    return img


def visualizeobb(img, bbox_array, scores, labels):

    for temp, s, l in zip(bbox_array, scores, labels):
        p1, p2 = (int(temp[3][0]), int(temp[3][1])), (int(temp[2][0]), int(temp[2][1]))

        clas = int(l)
        class2name = {0: "OK", 1: "NG"}
        class2color = {0: (46, 139, 87), 1: (0, 0, 255)}
        label_txt = class2name[clas] + " " + str(round(s, 2))
        color = class2color[clas]
        txt_color = (255, 255, 255)
        score = s
        cv2.polylines(
            img,
            [np.asarray(temp, dtype=int)],
            True,
            color,
            4,
        )
        # img = cv2.putText(
        #     img,
        #     "class:" + str(clas) + " " + str(round(score, 2)),
        #     (xmin, int(ymin) - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (105, 237, 249),
        #     1,
        # )

        tf = 2  # font thickness
        w, h = cv2.getTextSize(label_txt, 0, fontScale=1, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = int(p1[0]) + w, int(p1[1]) - h - 3 if outside else int(p1[1]) + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        img = cv2.putText(
            img,
            label_txt,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            1,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
        # img = cv2.putText(
        #     img,
        #     "class:" + str(class2name[clas]) + " " + str(round(score, 2)),
        #     (int(p1[0]), int(p1[1]) - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (105, 237, 249),
        #     1,
        # )
    return img


def sort_boxes(boxes: List[List[Any]]) -> Tuple[List[List[Any]], List[int]]:
    """
    将目标框按从左到右、从上到下排序

    :param boxes: list of [x_min, y_min, x_max, y_max, ...]
    :return: (排序后的boxes列表, 排序后对应的原始索引列表)
    """
    if not boxes:
        return [], []

    # 向量化提取坐标
    coords = np.array([[box[0], box[1], box[2], box[3]] for box in boxes])
    x_mins, y_mins, x_maxs, y_maxs = coords.T
    center_xs = (x_mins + x_maxs) / 2

    # 行分组阈值（基于平均高度）
    row_threshold = np.mean(y_maxs - y_mins)

    # 按 Y 坐标排序
    y_order = np.argsort(y_mins)

    # 分组为行（只存储原始索引）
    rows = [[y_order[0]]]
    base_y = y_mins[y_order[0]]

    for idx in y_order[1:]:
        if abs(y_mins[idx] - base_y) < row_threshold:
            rows[-1].append(idx)
        else:
            rows.append([idx])
            base_y = y_mins[idx]

    # 每行按 X 排序，收集最终索引
    sorted_indices = []
    for row in rows:
        sorted_indices.extend(list(map(int, sorted(row, key=lambda i: center_xs[i]))))

    sorted_boxes = [boxes[i] for i in sorted_indices]
    return sorted_boxes, sorted_indices


def decode2cv(image_base64):
    """
    image_base64:str image base64
    return :numpy cv
    """
    image = bytes(image_base64, encoding="utf8")
    img_byte = base64.b64decode(
        image
    )  # img_byte是字节型数据，二进制编码。b64decode对字节型b64编码数据进行解码。bytes->bytes
    image = np.frombuffer(img_byte, np.uint8)
    img_src = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return img_src


def process_mask(protos, masks_in, bboxes, shape):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
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

    def process_single_mask(mask):
        resized_mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        # resized_mask = cv2.GaussianBlur(resized_mask, (5, 5), 0)
        # resized_mask = cv2.GaussianBlur(resized_mask, (15, 15), 0.5)
        binary_mask = (sigmoid(resized_mask) > 0.9).astype(np.uint8)
        # mask_polygons = masks2segments(binary_mask)[0]
        # x, y = mask_polygons[:, 0], mask_polygons[:, 1]
        # tck, u = splprep([x, y], s=20)  # s 控制平滑程度
        # new_points = splev(np.linspace(0, 1, 100), tck)
        # # 将平滑后的点转换为整数
        # smooth_contour = np.array(new_points).T.astype(np.int32)
        # binary_mask = segments2masks([smooth_contour], binary_mask.shape)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # binary_mask = cv2.morphologyEx(binary_mask * 255, cv2.MORPH_OPEN, kernel, iterations=2)
        # binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
        # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return binary_mask

    with ThreadPoolExecutor() as executor:
        final_masks = list(executor.map(process_single_mask, masks))
    # final_masks = []
    # for mask in masks:
    #     resized_mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    #     binary_mask = (resized_mask > 0.5).astype(np.uint8)
    #     final_masks.append(binary_mask)
    return final_masks


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
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    if len(masks) == 0:
        return masks
    blur_size = (int(1 / gain), int(1 / gain))
    mh, mw = masks.shape
    pad = (dw, dh)
    top, left = (int(pad[1]), int(pad[0]))  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, shape)
    masks = cv2.blur(masks, blur_size)
    return masks


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywhr2xyxyxyxy(center):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        center (numpy.ndarray): Input data in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin = (np.cos, np.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1)
    vec2 = np.concatenate(vec2, axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def vis_box_mask(image, res):
    img_src = visualize(image, res["rect"], res["score"], res["cls"])
    # color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    # mask_color = np.expand_dims(res["masks"], -1).repeat(3, axis=-1) * color
    # post_image = cv2.addWeighted(img_src, 0.5, mask_color.astype(np.uint8), 0.5, 0)
    return img_src


def vis_box_mask_obb(image, res):
    img_src = visualizeobb(image, res["rect"], res["score"], res["cls"])
    # color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    # mask_color = np.expand_dims(res["masks"], -1).repeat(3, axis=-1) * color
    # post_image = cv2.addWeighted(img_src, 0.5, mask_color.astype(np.uint8), 0.5, 0)
    return img_src


def masks2segments(masks):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
      masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
      strategy (str): 'concat' or 'largest'. Defaults to largest

    Returns:
      segments (List): list of segment masks
    """
    segments = []
    c = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if c:
        for p in c:
            segments.append(p.reshape(-1, 2).astype("float32"))
    else:
        segments = np.zeros((0, 2))  # no segments found
    return segments


def segments2masks(points, mask_shape):
    h, w = mask_shape
    gt = np.zeros((h, w), dtype=np.uint8)
    for p in points:
        cv2.fillPoly(gt, p.astype(np.int32)[np.newaxis, :, :], int(1))
    return gt


def rotate_points(res, src_w, src_h):
    w = src_h
    h = src_w

    detailList = res.get("detailList", [])
    for detail in detailList:
        # 归一化坐标还原,并限制wh
        x1, y1, x2, y2, x3, y3, x4, y4 = detail.get("coordinate", [])
        x1, y1, x2, y2, x3, y3, x4, y4 = (
            min(w, max(0, int(x1 * w))),
            min(h, max(0, int(y1 * h))),
            min(w, max(0, int(x2 * w))),
            min(h, max(0, int(y2 * h))),
            min(w, max(0, int(x3 * w))),
            min(h, max(0, int(y3 * h))),
            min(w, max(0, int(x4 * w))),
            min(h, max(0, int(y4 * h))),
        )

        x2 = x3
        y2 = y3

        x_1 = h - y2
        y_1 = x1

        x_2 = h - y1
        y_2 = x2

        x_3 = x_2
        y_3 = y_1

        x_4 = x_1
        y_4 = y_2
        detail["coordinate"] = [
            x_1 / src_w,
            y_1 / src_h,
            x_3 / src_w,
            y_3 / src_h,
            x_2 / src_w,
            y_2 / src_h,
            x_4 / src_w,
            y_4 / src_h,
        ]
    return res
