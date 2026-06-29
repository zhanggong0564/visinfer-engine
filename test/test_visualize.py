"""services/utils/visualize 绘制模块单元测试"""
import base64

import cv2
import numpy as np

from services.utils.visualize import (
    render_detection_overlay,
    _hex_to_bgr,
    _coords_to_points,
    _draw_dashed_rect,
    _draw_rotated_label,
)


def _decode_b64_jpeg(b64: str):
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class TestHexToBgr:
    def test_green(self):
        assert _hex_to_bgr("#20ff4f") == (79, 255, 32)

    def test_yellow(self):
        assert _hex_to_bgr("#FFFF00") == (0, 255, 255)

    def test_invalid_falls_back_to_default(self):
        assert _hex_to_bgr("nope", default=(1, 2, 3)) == (1, 2, 3)

    def test_none_falls_back_to_default(self):
        assert _hex_to_bgr(None, default=(1, 2, 3)) == (1, 2, 3)


class TestCoordsToPoints:
    def test_normalized_scaled_to_new_dims(self):
        pts = _coords_to_points([0, 0, 1, 0, 1, 1, 0, 1], new_w=100, new_h=50, scale=1.0)
        assert pts.tolist() == [[0, 0], [100, 0], [100, 50], [0, 50]]

    def test_pixel_scaled_by_ratio(self):
        pts = _coords_to_points([0, 0, 200, 0, 200, 100, 0, 100], new_w=100, new_h=50, scale=0.5)
        assert pts.tolist() == [[0, 0], [100, 0], [100, 50], [0, 50]]

    def test_empty_returns_none(self):
        assert _coords_to_points([], new_w=100, new_h=50, scale=1.0) is None


class TestRenderDetectionOverlay:
    def _item(self, **kw):
        base = {
            "status": "true",
            "scene": "screw",
            "coordinate": [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            "accuracy": 0.9,
            "name": "screw",
            "color": "#20ff4f",
        }
        base.update(kw)
        return base

    def test_returns_decodable_jpeg(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        b64 = render_detection_overlay(img, [self._item()])
        assert isinstance(b64, str) and b64 != ""
        decoded = _decode_b64_jpeg(b64)
        assert decoded is not None

    def test_none_image_returns_empty(self):
        assert render_detection_overlay(None, [self._item()]) == ""

    def test_empty_detail_list_still_returns_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        b64 = render_detection_overlay(img, [])
        assert b64 != ""
        assert _decode_b64_jpeg(b64) is not None

    def test_downscales_longest_side_to_max(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        b64 = render_detection_overlay(img, [self._item()], max_side=1280)
        decoded = _decode_b64_jpeg(b64)
        assert max(decoded.shape[:2]) == 1280

    def test_small_image_not_upscaled(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        b64 = render_detection_overlay(img, [self._item()], max_side=1280)
        decoded = _decode_b64_jpeg(b64)
        assert decoded.shape[:2] == (100, 200)

    def test_pixel_coords_and_ng_and_chinese_label_no_crash(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        items = [
            self._item(status="false", color="#FFFF00",
                       coordinate=[10, 10, 90, 10, 90, 90, 10, 90], name="中文标签"),
        ]
        b64 = render_detection_overlay(img, items)
        assert _decode_b64_jpeg(b64) is not None

    def test_malformed_coordinate_skipped(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        b64 = render_detection_overlay(img, [self._item(coordinate=[1, 2, 3])])
        assert _decode_b64_jpeg(b64) is not None


class TestGuides:
    def _decode(self, b64):
        import base64
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _blue_pixel_count(self, img):
        # 蓝(BGR 255,0,0)：B 高、G/R 低
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return int(np.count_nonzero((b > 150) & (g < 100) & (r < 100)))

    def test_guides_draw_blue(self):
        img = np.full((200, 200, 3), 255, dtype=np.uint8)  # 白底
        b64 = render_detection_overlay(img, [], guides=[(0.2, 0.2, 0.5, 0.5)])
        out = self._decode(b64)
        assert self._blue_pixel_count(out) > 0

    def test_no_guides_no_blue(self):
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        b64 = render_detection_overlay(img, [], guides=None)
        out = self._decode(b64)
        assert self._blue_pixel_count(out) == 0

    def test_malformed_guide_skipped(self):
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        b64 = render_detection_overlay(img, [], guides=[(0.1, 0.1, 0.5)])  # 长度非4
        assert self._decode(b64) is not None

    def test_dashed_rect_draws_segments(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _draw_dashed_rect(img, 10, 10, 90, 90, (255, 0, 0), 2)
        # 虚线：边上有蓝点但不连续（存在间隙）
        top = img[10, 10:90, 0]
        assert int(np.count_nonzero(top > 150)) > 0
        assert int(np.count_nonzero(top <= 150)) > 0


class TestRotatedLabel:
    def test_ascii_label_draws_pixels(self):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        before = canvas.copy()
        pts = np.array([[40, 90], [160, 90], [160, 110], [40, 110]], np.int32)  # 横向长框
        _draw_rotated_label(canvas, "S2-14", pts, (0, 0, 255), 0.8, 2)
        assert not np.array_equal(canvas, before)  # 确有像素被改（画了字+底条）

    def test_chinese_label_no_change(self):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        before = canvas.copy()
        pts = np.array([[40, 90], [160, 90], [160, 110], [40, 110]], np.int32)
        _draw_rotated_label(canvas, "中文标签", pts, (0, 0, 255), 0.8, 2)
        assert np.array_equal(canvas, before)  # 非 ASCII 跳过

    def test_vertical_box_no_crash(self):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        pts = np.array([[95, 30], [115, 30], [115, 170], [95, 170]], np.int32)  # 纵向长框
        _draw_rotated_label(canvas, "J27-1", pts, (0, 255, 0), 0.8, 2)  # 不抛异常即可

    def test_out_of_bounds_box_no_crash(self):
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
        pts = np.array([[80, 5], [180, 5], [180, 20], [80, 20]], np.int32)  # 越界到画布外
        _draw_rotated_label(canvas, "EDGE-1", pts, (0, 255, 0), 0.8, 2)

    def test_render_uses_rotated_label_end_to_end(self):
        import base64
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        item = {
            "status": "true", "scene": "x", "name": "S2-14",
            "coordinate": [40, 90, 160, 90, 160, 110, 40, 110], "color": "#20ff4f",
        }
        b64 = render_detection_overlay(img, [item])
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        assert cv2.imdecode(arr, cv2.IMREAD_COLOR) is not None
