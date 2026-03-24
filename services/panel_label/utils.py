'''
@Author       : gongzhang4
@Date         : 2026-02-28 01:22:13
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-02 06:40:37
@FilePath     : utils.py
@Description  :
'''

import cv2
import numpy as np
from typing import List


def sort_mask(
    ori_img,
    points: np.array,
    sort_by: str = "y",
) -> List[np.array]:

    # opencv求轮廓的重心
    centroids = []
    temp = np.zeros_like(ori_img)
    image_cx, image_cy = ori_img.shape[1] // 2, ori_img.shape[0] // 2
    for point in points:
        point = np.array(point, dtype=np.int32).reshape((-1, 1, 2))
        M = cv2.moments(point)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

#     if sort_by == "y":
#         sorted_idx = np.argsort([cy for cx, cy in centroids])
#     elif sort_by == "xy":
#         # 先分上下，再分对上面的按x排序，对下面的按x排序
#         top_idx = np.where([cy < image_cy for cx, cy in centroids])[0]
#         bottom_idx = np.where([cy >= image_cy for cx, cy in centroids])[0]
#         sorted_idx = np.concatenate(
#             [
#                 top_idx[np.argsort(np.array([cx for cx, cy in centroids])[top_idx])],
#                 bottom_idx[np.argsort(np.array([cx for cx, cy in centroids])[bottom_idx])],
#             ]
#         )
#     elif sort_by == "x":
#         sorted_idx = np.argsort([cx for cx, cy in centroids])
#     return [points[idx] for idx in sorted_idx], sorted_idx

# print(sorted_idx)
# for i, idx in enumerate(sorted_idx):
#     cv2.circle(temp, (centroids[idx]), 5, (0, 255, 0), -1)
#     cv2.putText(temp, str(i), (centroids[idx][0], centroids[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# cv2.imwrite('temp.jpg', temp)
from typing import List, Tuple
import numpy as np
import cv2


def sort_mask(
    ori_img: np.ndarray,
    points: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    统一排序规则：
    1) 先按重心 y 分行（从上到下）
    2) 行内按重心 x 排序（从左到右）

    适配：
    - 两行：第一行左->右，再第二行左->右
    - 一列：上->下
    - 一行：左->右
    """
    if points is None or len(points) == 0:
        return [], np.array([], dtype=np.int64)

    # 1) 计算每个 polygon 的重心
    items = []  # (orig_idx, cx, cy, h)
    for i, point in enumerate(points):
        contour = np.array(point, dtype=np.int32).reshape((-1, 1, 2))
        M = cv2.moments(contour)

        arr = np.array(point, dtype=np.float32).reshape(-1, 2)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            # 兜底：退化轮廓用均值中心
            cx = float(arr[:, 0].mean())
            cy = float(arr[:, 1].mean())

        h = float(arr[:, 1].max() - arr[:, 1].min() + 1e-6)  # 框高
        items.append((i, cx, cy, h))

    # 2) 先按 y 从上到下
    items.sort(key=lambda t: t[2])

    # 3) 按 y 聚类成“行”
    # 阈值：行内允许的 y 浮动，取中位高度的一部分
    heights = np.array([it[3] for it in items], dtype=np.float32)
    row_thr = max(5.0, float(np.median(heights) * 0.5))

    rows = []  # 每行是若干 item
    for it in items:
        if not rows:
            rows.append([it])
            continue

        last_row = rows[-1]
        row_cy = float(np.mean([x[2] for x in last_row]))

        if abs(it[2] - row_cy) <= row_thr:
            last_row.append(it)
        else:
            rows.append([it])

    # 4) 行内按 x 从左到右
    sorted_indices = []
    for row in rows:
        row.sort(key=lambda t: t[1])  # cx
        sorted_indices.extend([x[0] for x in row])

    sorted_idx = np.array(sorted_indices, dtype=np.int64)
    sorted_points = [points[idx] for idx in sorted_idx]
    return sorted_points, sorted_idx


def points_to_mask(shape_hw, points):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def rotate_upright(img, mask):

    ys, xs = np.where(mask > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    h, w = mask.shape[:2]
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    x_min = np.clip(x_min, 0, w)
    y_min = np.clip(y_min, 0, h)
    x_max = np.clip(x_max, 0, w)
    y_max = np.clip(y_max, 0, h)
    roi = img[y_min:y_max, x_min:x_max]
    mask_roi = mask[y_min:y_max, x_min:x_max]

    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
    cx -= x_min
    cy -= y_min
    diag = int(np.sqrt(rw**2 + rh**2))  # 对角线长度
    new_w, new_h = diag, diag

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += (new_w - roi.shape[1]) / 2
    M[1, 2] += (new_h - roi.shape[0]) / 2
    # h, w = roi.shape[:2]
    img_r = cv2.warpAffine(roi, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask_r = cv2.warpAffine(
        mask_roi, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return img_r, mask_r


def smooth_1d(y, k):
    if k is None or k <= 1:
        return y
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(y.reshape(-1, 1), (k, 1), 0).ravel()


def contour_top_bottom(mask):
    ##
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("mask 里没找到轮廓")
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]  # (N,2)

    xs = cnt[:, 0]
    ys = cnt[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())

    minY = {}
    maxY = {}
    for x, y in zip(xs, ys):
        x = int(x)
        y = float(y)
        if x not in minY:
            minY[x] = y
            maxY[x] = y
        else:
            if y < minY[x]:
                minY[x] = y
            if y > maxY[x]:
                maxY[x] = y

    x_all = np.arange(x_min, x_max + 1, dtype=np.int32)
    top = np.stack(
        [x_all.astype(np.float32), np.array([minY.get(x, np.nan) - 10 for x in x_all], dtype=np.float32)], axis=1
    )
    bot = np.stack(
        [x_all.astype(np.float32), np.array([maxY.get(x, np.nan) + 10 for x in x_all], dtype=np.float32)], axis=1
    )

    def fill_nan(y):
        n = len(y)
        idx = np.where(~np.isnan(y))[0]
        if len(idx) < 2:
            # 极端情况：退化了
            y[np.isnan(y)] = 0
            return y
        return np.interp(np.arange(n), idx, y[idx]).astype(np.float32)

    top[:, 1] = fill_nan(top[:, 1])
    bot[:, 1] = fill_nan(bot[:, 1])
    return top, bot


def mask2roi(img: np.ndarray, points: np.array, smooth=21, sample_step=1, border_mode="replicate") -> List[np.ndarray]:
    rois = []
    for point in points:
        mask = points_to_mask(img.shape[:2], point)
        img_r, mask_r = rotate_upright(img, mask)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        # mask_r = cv2.erode(mask_r, kernel, iterations=2)
        # roi = cv2.bitwise_and(img_r, img_r, mask=mask_r)
        ###求矩形框，crop得到roi
        top, bot = contour_top_bottom(mask_r)

        # 平滑边界，减少展平后的“波纹”
        top[:, 1] = smooth_1d(top[:, 1], smooth)
        bot[:, 1] = smooth_1d(bot[:, 1], smooth)

        # 下采样加速
        top = top[::sample_step]
        bot = bot[::sample_step]

        W = len(top)
        H = int(np.max(bot[:, 1] - top[:, 1]))
        # 固定H
        # H = 256

        # 构建 remap
        map_x = np.empty((H, W), dtype=np.float32)
        map_y = np.empty((H, W), dtype=np.float32)

        # 逐列线性插值（从 top 到 bot）
        for i in range(W):
            x_t, y_t = top[i]
            x_b, y_b = bot[i]
            a = np.linspace(0.0, 1.0, H, dtype=np.float32)
            map_x[:, i] = x_t + a * (x_b - x_t)
            map_y[:, i] = y_t + a * (y_b - y_t)

        bm = cv2.BORDER_REPLICATE if border_mode == "replicate" else cv2.BORDER_REFLECT_101
        flat = cv2.remap(img_r, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=bm)
        rois.append(flat)
        cv2.imwrite("roi.jpg", flat)

    return rois


def Points_to_Mask(image_src, points, sort_by="y"):
    points_line, sorted_idx = sort_mask(image_src, points, sort_by=sort_by)
    mask_rois = mask2roi(image_src, points_line)
    return mask_rois, sorted_idx
