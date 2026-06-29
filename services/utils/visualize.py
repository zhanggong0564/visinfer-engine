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


_LABEL_ALPHA = 0.7  # 旋转文字底条+文字整体合成不透明度


def _draw_rotated_label(canvas, text, pts, color, font_scale, thickness):
    """把 text 渲染成带深色底条的小图，沿框最长边角度旋转，偏到框外一侧贴到 canvas。

    朝向用最长边角度（不依赖 cv2.minAreaRect 的版本相关 angle）；
    非 ASCII 文本跳过（仅 ASCII 约定）；越界由 warpAffine 到画布尺寸自然裁剪，不抛异常。
    """
    if not text or not text.isascii():
        return
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return
    H, W = canvas.shape[:2]

    # 最长边 → 角度（度），归一到 [-90,90] 防上下颠倒
    edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    (pa, pb) = max(edges, key=lambda e: np.hypot(*(e[1] - e[0])))
    dx, dy = (pb - pa)
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    # 短边半长（用于把文字偏到框外）
    edge_lens = sorted(float(np.hypot(*(b - a))) for a, b in edges)
    short_half = (edge_lens[0] if edge_lens else 0.0) / 2.0
    cx, cy = pts.mean(axis=0)

    # 渲染文字小图（深色底条 + 彩色字）。文字描边固定细一档，避免发糊发粗。
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, thickness - 2)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    pad = 4
    sw, sh = tw + 2 * pad, th + baseline + 2 * pad
    strip = np.full((sh, sw, 3), 40, dtype=np.uint8)  # 深灰底条
    cv2.putText(strip, text, (pad, pad + th), font, font_scale, color, text_thickness, cv2.LINE_AA)
    mask = np.full((sh, sw), 255, dtype=np.uint8)

    # 偏到框外：沿最长边法线方向外移 (short_half + 半个底条高 + 留白)
    rad = np.radians(angle)
    nx, ny = -np.sin(rad), np.cos(rad)  # 法线
    off = short_half + sh / 2.0 + 4
    tx, ty = cx + nx * off, cy + ny * off

    # 引导细线：从框中心拉细线到标签中心，明确"标签↔框"对应（先画，文字底条压其末端）
    cv2.line(canvas, (int(round(cx)), int(round(cy))), (int(round(tx)), int(round(ty))),
             color, 1, cv2.LINE_AA)

    # 旋转小图+掩膜并放到目标点（warpAffine 到画布尺寸，越界自动裁剪）
    M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), angle, 1.0)
    M[0, 2] += tx - sw / 2.0
    M[1, 2] += ty - sh / 2.0
    rot = cv2.warpAffine(strip, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rmask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    idx = rmask > 0
    if not idx.any():
        return
    blended = (canvas.astype(np.float32) * (1 - _LABEL_ALPHA)
               + rot.astype(np.float32) * _LABEL_ALPHA)
    canvas[idx] = blended[idx].astype(np.uint8)


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

        # 目标框 + 旋转贴框文字
        for pts, color, _, label in drawables:
            cv2.polylines(canvas, [pts], True, color, thickness)
            _draw_rotated_label(canvas, label, pts, color, font_scale, thickness)

        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:
        vision_logger.warning(f"可视化绘制失败: {e}")
        return ""
