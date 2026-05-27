"""auto_annotate 单元测试"""
import json
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# 将项目根目录加入 path（与其他测试保持一致）
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBuildLabelmeJson:
    """_build_labelme_json 纯函数测试"""

    def test_basic_structure(self):
        """返回的 dict 必须包含 LabelMe 所有顶层字段"""
        from tools.auto_annotate import _build_labelme_json

        result = _build_labelme_json(
            shapes=[],
            image_filename="img.jpg",
            image_height=100,
            image_width=200,
        )
        assert result["version"] == "3.3.9"
        assert result["flags"] == {}
        assert result["shapes"] == []
        assert result["imagePath"] == "img.jpg"
        assert result["imageData"] is None
        assert result["imageHeight"] == 100
        assert result["imageWidth"] == 200
        assert result["description"] == ""

    def test_shape_fields(self):
        """shapes 列表中每个元素必须包含所有 LabelMe shape 字段"""
        from tools.auto_annotate import _build_labelme_json

        shapes = [
            {
                "label": "text",
                "score": 0.95,
                "points": [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]],
                "description": "PE1-J5",
            }
        ]
        result = _build_labelme_json(
            shapes=shapes,
            image_filename="img.jpg",
            image_height=100,
            image_width=200,
        )
        shape = result["shapes"][0]
        assert shape["label"] == "text"
        assert shape["score"] == 0.95
        assert shape["points"] == [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]
        assert shape["group_id"] == 0
        assert shape["description"] == "PE1-J5"
        assert shape["difficult"] is False
        assert shape["shape_type"] == "polygon"
        assert shape["flags"] is None
        assert shape["attributes"] == {}
        assert shape["kie_linking"] == []

    def test_empty_description_when_ocr_failed(self):
        """OCR 未识别时 description 应为空字符串"""
        from tools.auto_annotate import _build_labelme_json

        shapes = [
            {
                "label": "text",
                "score": 0.0,
                "points": [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]],
                "description": "",
            }
        ]
        result = _build_labelme_json(shapes=shapes, image_filename="x.jpg", image_height=50, image_width=100)
        assert result["shapes"][0]["description"] == ""
