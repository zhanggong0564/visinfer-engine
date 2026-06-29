'''
@Description : 服务端检测结果可视化：把 detailList 绘制到缩图上并编码为 JPEG base64。

框架层通用绘制，复用各场景 detailList（coordinate+color+status+name+scene），
插件零改动。坐标系兼容归一化 8 点与像素两种；文字仅 ASCII；任何异常降级返回 ""。
'''

import base64

import cv2
import numpy as np

from utils import vision_logger

# BGR；与 DetectionItem 颜色约定对齐：true→#20ff4f 绿，false→#FFFF00 黄
_GREEN_BGR = (79, 255, 32)
_YELLOW_BGR = (0, 255, 255)
_FILL_ALPHA = 0.3  # NG 半透明填充权重
_BLUE_BGR = (255, 0, 0)  # 引导框：蓝色(BGR)，区别于检测框绿/黄


def _hex_to_bgr(hex_color, default=_GREEN_BGR):
    """'#RRGGBB' → (B, G, R)；非法输入回退 default。"""
    try:
        h = str(hex_color).lstrip("#")
        if len(h) != 6:
            return default
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    except Exception:
        return default


def _coords_to_points(coord, new_w, new_h, scale):
    """扁平坐标 [x1,y1,...] → int32 点阵 (N,2)，按坐标系自动换算到缩图尺寸。

    归一化（max(|coord|)≤1.5）：乘缩图宽高；像素：乘缩放比 scale。空输入返回 None。
    """
    pts = np.asarray(coord, dtype=np.float64)
    if pts.size == 0 or pts.size % 2 != 0:
        return None
    pts = pts.reshape(-1, 2)
    if np.max(np.abs(pts)) <= 1.5:
        pts[:, 0] *= new_w
        pts[:, 1] *= new_h
    else:
        pts *= scale
    return pts.astype(np.int32)


def _draw_dashed_line(img, p1, p2, color, thickness, dash=12, gap=8):
    """两点间画虚线（OpenCV 无原生虚线，按 dash/gap 分段画实线段）。"""
    x1, y1 = p1
    x2, y2 = p2
    dist = int(round(float(np.hypot(x2 - x1, y2 - y1))))
    if dist == 0:
        return
    step = dash + gap
    for i in range(0, dist, step):
        s = i / dist
        e = min(i + dash, dist) / dist
        sx = int(round(x1 + (x2 - x1) * s))
        sy = int(round(y1 + (y2 - y1) * s))
        ex = int(round(x1 + (x2 - x1) * e))
        ey = int(round(y1 + (y2 - y1) * e))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def _draw_dashed_rect(img, x1, y1, x2, y2, color, thickness, dash=12, gap=8):
    """四条边分别画虚线，组成虚线矩形。"""
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash, gap)
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash, gap)
    _draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness, dash, gap)
    _draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness, dash, gap)


def render_detection_overlay(image, detail_list, *, guides=None, max_side=1280, jpeg_quality=85):
    """把 detailList 绘制到缩图上并返回 JPEG base64（不含 data: 前缀）。

    异常或空图一律返回 ""，绝不抛出，避免影响检测主响应。
    """
    try:
        if image is None or getattr(image, "size", 0) == 0:
            return ""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return ""

        scale = min(1.0, max_side / float(max(h, w)))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if scale < 1.0:
            canvas = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            canvas = image.copy()

        thickness = max(2, int(round(max(new_w, new_h) / 400)))
        font_scale = max(0.4, max(new_w, new_h) / 1600.0)

        # 引导框（归一化 x,y,w,h → 像素），蓝色虚线、细一档，画在检测框之下
        guide_thickness = max(2, thickness - 1)
        for g in guides or []:
            if not isinstance(g, (list, tuple)) or len(g) != 4:
                continue
            try:
                gx, gy, gw, gh = (float(v) for v in g)
            except (TypeError, ValueError):
                continue
            x1 = int(round(gx * new_w))
            y1 = int(round(gy * new_h))
            x2 = int(round((gx + gw) * new_w))
            y2 = int(round((gy + gh) * new_h))
            _draw_dashed_rect(canvas, x1, y1, x2, y2, _BLUE_BGR, guide_thickness)

        # 解析为可绘制元素：(pts, color, is_ng, label)
        drawables = []
        for item in detail_list or []:
            if not isinstance(item, dict):
                continue
            coord = item.get("coordinate") or []
            if not isinstance(coord, (list, tuple)) or len(coord) < 6 or len(coord) % 2 != 0:
                continue
            pts = _coords_to_points(coord, new_w, new_h, scale)
            if pts is None or len(pts) < 3:
                continue
            is_ng = str(item.get("status", "")).strip().lower() == "false"
            color = _hex_to_bgr(item.get("color"), _YELLOW_BGR if is_ng else _GREEN_BGR)
            label = item.get("name") or item.get("scene") or ""
            drawables.append((pts, color, is_ng, label))

        # NG 半透明填充（先在副本填充再整体混合，保证框/文字仍全不透明）
        if any(d[2] for d in drawables):
            overlay = canvas.copy()
            for pts, color, is_ng, _ in drawables:
                if is_ng:
                    cv2.fillPoly(overlay, [pts], color)
            canvas = cv2.addWeighted(overlay, _FILL_ALPHA, canvas, 1.0 - _FILL_ALPHA, 0)

        # 目标框 + 标签
        for pts, color, _, label in drawables:
            cv2.polylines(canvas, [pts], True, color, thickness)
            if label and label.isascii():
                x = int(pts[0][0])
                y = int(pts[0][1])
                ty = y - 5 if y - 5 > 0 else y + 15
                cv2.putText(canvas, label, (x, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:
        vision_logger.warning(f"可视化绘制失败: {e}")
        return ""
