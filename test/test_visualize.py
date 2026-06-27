"""services/utils/visualize 绘制模块单元测试"""
import base64

import cv2
import numpy as np

from services.utils.visualize import (
    render_detection_overlay,
    _hex_to_bgr,
    _coords_to_points,
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
