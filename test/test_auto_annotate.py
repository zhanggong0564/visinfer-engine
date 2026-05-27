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


class TestAutoAnnotatorInit:
    """AutoAnnotator 初始化测试（mock PaddleOCR）"""

    def test_init_loads_three_models(self):
        """初始化时应加载 TextDetection、TextLineOrientationClassification、TextRecognition"""
        with (
            patch("tools.auto_annotate.TextDetection") as mock_det,
            patch("tools.auto_annotate.TextLineOrientationClassification") as mock_ori,
            patch("tools.auto_annotate.TextRecognition") as mock_rec,
            patch("tools.auto_annotate.CropByPolys") as mock_crop,
        ):
            from tools.auto_annotate import AutoAnnotator
            ann = AutoAnnotator(
                orient_model_path="fake/orient",
                rec_model_path="fake/rec",
                score_thresh=0.8,
            )
            mock_det.assert_called_once()
            mock_ori.assert_called_once()
            mock_rec.assert_called_once()
            mock_crop.assert_called_once_with(det_box_type="quad")
            assert ann.score_thresh == 0.8


class TestAutoAnnotatorInferImage:
    """infer_image 测试：mock 全部 PaddleOCR 调用"""

    @pytest.fixture
    def annotator(self):
        """构造 AutoAnnotator，绕过真实模型加载"""
        with (
            patch("tools.auto_annotate.TextDetection"),
            patch("tools.auto_annotate.TextLineOrientationClassification"),
            patch("tools.auto_annotate.TextRecognition"),
            patch("tools.auto_annotate.CropByPolys"),
        ):
            from tools.auto_annotate import AutoAnnotator
            ann = AutoAnnotator(orient_model_path="x", rec_model_path="y", score_thresh=0.7)
        return ann

    def test_no_text_detected_returns_empty_shapes(self, annotator):
        """文本检测无结果时，返回 shapes 为空列表的 LabelMe JSON"""
        # TextDetection 返回空 dt_polys
        annotator.text_det.predict = MagicMock(return_value=[{"dt_polys": []}])

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = annotator.infer_image(image, "img.jpg")

        assert result["shapes"] == []
        assert result["imagePath"] == "img.jpg"
        assert result["imageHeight"] == 100
        assert result["imageWidth"] == 200

    def test_one_text_recognized(self, annotator):
        """检测到1个文本框且 OCR 分数达标时，shapes 有1条记录"""
        dt_poly = [[10, 20], [50, 20], [50, 40], [10, 40]]
        annotator.text_det.predict = MagicMock(return_value=[{"dt_polys": [dt_poly]}])

        crop_img = np.zeros((20, 40, 3), dtype=np.uint8)
        annotator._crop = MagicMock(return_value=iter([crop_img]))

        annotator.text_ori.predict = MagicMock(return_value=[{"class_ids": [0]}])
        annotator.text_rec.predict = MagicMock(
            return_value=[{"rec_text": "PE1-J5", "rec_score": 0.95}]
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = annotator.infer_image(image, "img.jpg")

        assert len(result["shapes"]) == 1
        shape = result["shapes"][0]
        assert shape["description"] == "PE1-J5"
        assert shape["score"] == pytest.approx(0.95)
        assert shape["points"] == [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]

    def test_low_score_gives_empty_description(self, annotator):
        """OCR 识别分数低于阈值时，description 为空字符串，框仍保留"""
        dt_poly = [[10, 20], [50, 20], [50, 40], [10, 40]]
        annotator.text_det.predict = MagicMock(return_value=[{"dt_polys": [dt_poly]}])

        crop_img = np.zeros((20, 40, 3), dtype=np.uint8)
        annotator._crop = MagicMock(return_value=iter([crop_img]))

        annotator.text_ori.predict = MagicMock(return_value=[{"class_ids": [0]}])
        annotator.text_rec.predict = MagicMock(
            return_value=[{"rec_text": "BLURRY", "rec_score": 0.3}]  # 低于 0.7
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = annotator.infer_image(image, "img.jpg")

        assert len(result["shapes"]) == 1
        assert result["shapes"][0]["description"] == ""
        assert result["shapes"][0]["score"] == pytest.approx(0.3)
