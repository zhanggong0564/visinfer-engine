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
_OVERLAY_ALPHA = 0.6  # 标号徽标+图例整体合成不透明度（半透明，不遮挡画面）
_DATA_URI_PREFIX = "data:image/jpeg;base64,"  # vis_image 前缀，前端可直接用于 <img src>，失败时返回空串不带前缀
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


def _label_text_color(bgr):
    """据底色亮度选前景色：亮底用黑字、暗底用白字，保证序号可读。"""
    b, g, r = bgr
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _draw_index_badge(canvas, center, number, color, radius):
    """在指定位置画状态色实心圆 + 序号，作为与左上角图例一一对应的编号。"""
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    cv2.circle(canvas, (cx, cy), radius, color, -1)
    cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)
    txt = str(number)
    fs = max(0.35, radius / 14.0)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.putText(canvas, txt, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, _label_text_color(color), 1, cv2.LINE_AA)


def _short_edge_top_mid(pts):
    """取多边形较短两条边里更靠上(y 较小)那条的中点，用于把编号放到短边上方。

    假设框为细长形(线标场景)：最短两条边即两端短边；近方形框可能取到侧边中点，可接受。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    order = sorted(range(n), key=lambda i: np.hypot(*(edges[i][1] - edges[i][0])))
    short = order[:2] if n >= 4 else order[:1]
    mids = [(edges[i][0] + edges[i][1]) / 2.0 for i in short]
    return min(mids, key=lambda m: m[1])


def _draw_legend(canvas, entries):
    """在原图左上角就地画半透明白底图例，逐行列 "序号徽标 + 文本"（文本仅 ASCII）。

    entries: list[(number, text, color)]；空则不改 canvas。原图零扩展，仅占左上空白区。
    """
    if not entries:
        return
    H, W = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.45, min(0.9, W / 1400.0))
    line_h = max(20, int(34 * fs))
    r = max(7, int(line_h * 0.3))
    pad = 8
    text_x0 = pad + 2 * r + 8

    max_tw = 0
    for _, text, _ in entries:
        t = text if (text and text.isascii()) else ""
        (tw, _), _ = cv2.getTextSize(t, font, fs, 1)
        max_tw = max(max_tw, tw)

    x0, y0 = 6, 6
    block_w = min(W - x0, text_x0 + max_tw + pad)
    block_h = min(H - y0, pad + len(entries) * line_h + pad)

    # 半透明白底，保证文字在任意背景上可读
    roi = canvas[y0:y0 + block_h, x0:x0 + block_w].astype(np.float32)
    canvas[y0:y0 + block_h, x0:x0 + block_w] = (roi * 0.25 + 255 * 0.75).astype(np.uint8)

    for i, (number, text, color) in enumerate(entries):
        cy = y0 + pad + i * line_h + line_h // 2
        bx = x0 + pad + r
        cv2.circle(canvas, (bx, cy), r, color, -1)
        cv2.circle(canvas, (bx, cy), r, (180, 180, 180), 1, cv2.LINE_AA)
        ntxt = str(number)
        nfs = max(0.3, r / 14.0)
        (ntw, nth), _ = cv2.getTextSize(ntxt, font, nfs, 1)
        cv2.putText(canvas, ntxt, (bx - ntw // 2, cy + nth // 2),
                    font, nfs, _label_text_color(color), 1, cv2.LINE_AA)
        t = text if (text and text.isascii()) else ""
        (tw, th), _ = cv2.getTextSize(t, font, fs, 1)
        cv2.putText(canvas, t, (x0 + text_x0, cy + th // 2), font, fs, (20, 20, 20), 1, cv2.LINE_AA)


def render_detection_overlay(image, detail_list, *, guides=None, max_side=1280, jpeg_quality=85):
    """把 detailList 绘制到缩图上并返回 JPEG base64（带 data:image/jpeg;base64, 前缀，可直接塞 <img>）。

    异常或空图一律返回 ""（不带前缀），绝不抛出，避免影响检测主响应。
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

        # 引导框（归一化 x,y,w,h → 像素），蓝色虚线、细一档，画在检测框之下
        # 下限取 2：1px 虚线在 JPEG(q85) 压缩后会消失(test_guides_draw_blue 验证)；
        # 大图 thickness≥3 时仍比框线细一档，达成"参考线"视觉
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

        # 目标框（实色）+ 短边上方编号徽标 + 左上角图例
        badge_r = max(10, int(round(max(new_w, new_h) / 80)))
        items_idx = []
        for i, (pts, color, _, label) in enumerate(drawables, start=1):
            cv2.polylines(canvas, [pts], True, color, thickness)
            items_idx.append((i, pts, color, label))

        # 标号徽标 + 图例画到 overlay，再半透明合成（透明色，不遮挡画面/线标）
        if items_idx:
            overlay = canvas.copy()
            for i, pts, color, _ in items_idx:
                _draw_index_badge(overlay, _short_edge_top_mid(pts), i, color, badge_r)
            _draw_legend(overlay, [(i, label, color) for i, pts, color, label in items_idx])
            canvas = cv2.addWeighted(overlay, _OVERLAY_ALPHA, canvas, 1.0 - _OVERLAY_ALPHA, 0)

        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return _DATA_URI_PREFIX + b64
    except Exception as e:
        vision_logger.warning(f"可视化绘制失败: {e}")
        return ""
