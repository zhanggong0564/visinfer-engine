'''
@Author       : gongzhang4
@Date         : 2026-02-07 10:21:52
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-07 10:35:55
@FilePath     : test_yolo.py
@Description  :
'''

"""YoloOnnxInfer 单元测试"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def test_preprocess_delegates_to_shared_pipeline(monkeypatch):
    from services.yolo import YoloOnnxInfer

    model = YoloOnnxInfer.__new__(YoloOnnxInfer)
    model._input_model_shape = [1, 3, 8, 10]
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    sentinel = (object(), object())
    captured = {}

    def fake_prepare(value, shape):
        captured["args"] = (value, shape)
        return sentinel

    monkeypatch.setattr("services.yolo.prepare_yolo_input", fake_prepare)
    assert model.preprocess(image) is sentinel
    assert captured["args"][0] is image
    assert captured["args"][1] == [8, 10]


def test_post_process_delegates_to_shared_pipeline(monkeypatch):
    from schemas.inference_context import PreprocMeta
    from services.yolo import YoloOnnxInfer

    model = YoloOnnxInfer.__new__(YoloOnnxInfer)
    model.task = "det"
    model.confThreshold = 0.4
    model.nmsThreshold = 0.6
    model.filter_classes = [1]
    model.agnostic = True
    model.nc = 2
    model._input_model_shape = [1, 3, 8, 10]
    model.id2name = {1: "target"}
    raw_detection = np.array([[1, 2, 3, 4, 0.9, 1]], dtype=np.float32)
    restored = np.array([[10, 20, 30, 40, 0.9, 1]], dtype=np.float32)
    captured = {}

    def fake_nms(prediction, **kwargs):
        captured["nms"] = (prediction, kwargs)
        return [raw_detection]

    def fake_restore(detections, input_shape, src_shape):
        captured["restore"] = (detections, input_shape, src_shape)
        return restored

    monkeypatch.setattr("services.yolo.run_yolo_nms", fake_nms)
    monkeypatch.setattr("services.yolo.restore_yolo_boxes", fake_restore)
    prediction = np.zeros((1, 6, 1), dtype=np.float32)
    original = np.zeros((6, 7, 3), dtype=np.uint8)
    meta = PreprocMeta(r=1.0, dw=0, dh=0, src_shape=original.shape, ori_img=original)

    result = model.post_process([prediction], meta)

    assert captured["nms"] == (
        prediction,
        {
            "task": "det",
            "conf_threshold": 0.4,
            "iou_threshold": 0.6,
            "classes": [1],
            "agnostic": True,
            "nc": 2,
        },
    )
    assert captured["restore"][0] is raw_detection
    assert captured["restore"][1] == [8, 10]
    assert captured["restore"][2] == original.shape
    assert result.boxes == [[10.0, 20.0, 30.0, 40.0]]
    assert result.scores == [pytest.approx(0.9)]
    assert result.class_ids == [1.0]
    assert result.class_names == ["target"]
    assert result.ori_img is original


class TestYoloOnnxInfer:
    """YoloOnnxInfer 测试"""

    @pytest.fixture
    def mock_model(self):
        """模拟 ONNX 模型加载"""
        with patch.object(
            __import__("services.base", fromlist=["BaseOnnxInfer"]).BaseOnnxInfer,
            "__init__",
            lambda self, *args, **kwargs: None,
        ):
            from services.yolo import YoloOnnxInfer

            model = YoloOnnxInfer(model_path="fake_model.onnx", nc=80, confThreshold=0.5, nmsThreshold=0.5, task="det")
            # 模拟必要属性
            model._input_model_shape = (1, 3, 640, 640)
            model.id2name = {i: f"class_{i}" for i in range(80)}

            return model

    def test_init_params(self, mock_model):
        """测试初始化参数"""
        assert mock_model.nc == 80
        assert mock_model.confThreshold == 0.5
        assert mock_model.nmsThreshold == 0.5
        assert mock_model.task == "det"
        assert mock_model.agnostic is False
        assert mock_model.filter_classes is None

    def test_preprocess_returns_tensor_and_meta(self, mock_model):
        """测试预处理返回 (tensor, PreprocMeta) 元组"""
        from schemas.inference_context import PreprocMeta
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with patch("services.base.yolo_pipeline.letterbox") as mock_letterbox:
            mock_letterbox.return_value = (
                np.zeros((640, 640, 3), dtype=np.uint8), 1.0, 0, 0,
            )
            tensor, meta = mock_model.preprocess(input_image)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert tensor.max() <= 1.0 and tensor.min() >= 0.0
        assert isinstance(meta, PreprocMeta)
        assert meta.r == 1.0
        assert meta.src_shape == (480, 640, 3)

    def test_preprocess_normalization(self, mock_model):
        """测试预处理归一化"""
        # 全白图像
        white_image = np.ones((480, 640, 3), dtype=np.uint8) * 255

        with patch("services.base.yolo_pipeline.letterbox") as mock_letterbox:
            mock_letterbox.return_value = (np.ones((640, 640, 3), dtype=np.uint8) * 255, 1.0, 0, 0)

            tensor, _ = mock_model.preprocess(white_image)

        # 归一化后应接近 1.0
        assert np.allclose(tensor, 1.0)

    @patch("services.yolo.run_yolo_nms")
    @patch("services.yolo.restore_yolo_boxes")
    def test_post_process_det(self, mock_scale_boxes, mock_nms, mock_model):
        """测试检测任务后处理"""
        # 模拟 NMS 输出: [x1, y1, x2, y2, conf, cls]
        mock_pred = np.array(
            [
                [100, 100, 200, 200, 0.9, 0],
                [300, 300, 400, 400, 0.8, 1],
            ]
        )
        mock_nms.return_value = [mock_pred]
        mock_scale_boxes.return_value = mock_pred

        # 模拟模型输出
        preds = [np.random.rand(1, 84, 8400)]

        from schemas.inference_context import PreprocMeta
        meta = PreprocMeta(r=1.0, dw=0, dh=0, src_shape=(480, 640, 3),
                           ori_img=np.zeros((480, 640, 3), dtype=np.uint8))

        result = mock_model.post_process(preds, meta)

        # 验证返回结构
        assert hasattr(result, "boxes")
        assert hasattr(result, "scores")
        assert hasattr(result, "class_ids")
        assert hasattr(result, "class_names")
        assert len(result.boxes) == 2
        assert len(result.scores) == 2
        assert result.class_ids[0] == 0
        assert result.class_ids[1] == 1
        assert result.class_names[0] == "class_0"
        assert result.class_names[1] == "class_1"

    @patch("services.yolo.run_yolo_nms")
    @patch("services.yolo.restore_yolo_boxes")
    @patch("services.yolo.xywhr2xyxyxyxy")
    def test_post_process_obb(self, mock_xywhr, mock_scale_boxes, mock_nms, mock_model):
        """测试旋转框任务后处理"""
        mock_model.task = "obb"

        mock_pred = np.array(
            [
                [150, 150, 50, 50, 0.85, 2],
            ]
        )
        mock_nms.return_value = [mock_pred]
        mock_scale_boxes.return_value = mock_pred
        mock_xywhr.return_value = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])

        preds = [np.random.rand(1, 84, 8400)]

        from schemas.inference_context import PreprocMeta
        meta = PreprocMeta(r=1.0, dw=0, dh=0, src_shape=(480, 640, 3),
                           ori_img=np.zeros((480, 640, 3), dtype=np.uint8))

        result = mock_model.post_process(preds, meta)

        # 验证调用了旋转框转换
        mock_xywhr.assert_called_once()

    @patch("services.yolo.run_yolo_nms")
    @patch("services.yolo.restore_yolo_boxes")
    def test_post_process_empty_detection(self, mock_scale_boxes, mock_nms, mock_model):
        """测试无检测结果"""
        # 空预测
        mock_pred = np.empty((0, 6))
        mock_nms.return_value = [mock_pred]
        mock_scale_boxes.return_value = mock_pred

        preds = [np.random.rand(1, 84, 8400)]

        from schemas.inference_context import PreprocMeta
        meta = PreprocMeta(r=1.0, dw=0, dh=0, src_shape=(480, 640, 3),
                           ori_img=np.zeros((480, 640, 3), dtype=np.uint8))

        result = mock_model.post_process(preds, meta)

        assert len(result.boxes) == 0
        assert len(result.scores) == 0
        assert len(result.class_ids) == 0
        assert len(result.class_names) == 0

    @patch("services.yolo.masks2segments_with_boxes")
    @patch("services.yolo.scale_masks")
    @patch("services.yolo.process_mask")
    @patch("services.yolo.run_yolo_nms")
    @patch("services.yolo.restore_yolo_boxes")
    def test_post_process_seg_drops_degenerate_mask_keeps_others(
        self, mock_scale_boxes, mock_nms, mock_process_mask, mock_scale_masks, mock_m2s, mock_model
    ):
        """seg: 个别掩膜退化(0 轮廓)时应丢弃该检测并保留其余，而非作废整帧。"""
        mock_model.task = "seg"
        # 两个检测，列含 mask 系数：[x,y,x,y,conf,cls, c1, c2]
        mock_pred = np.array(
            [
                [100, 100, 200, 200, 0.9, 0, 0.1, 0.2],
                [300, 300, 400, 400, 0.8, 0, 0.1, 0.2],
            ]
        )
        mock_nms.return_value = [mock_pred]
        mock_scale_boxes.return_value = mock_pred
        mock_process_mask.return_value = np.zeros((2, 160, 160), dtype=np.uint8)
        # scale_masks 返回 (H, W, N)，post_process 会 .transpose(2, 0, 1)
        mock_scale_masks.return_value = np.zeros((480, 640, 2), dtype=np.uint8)
        seg0 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
        # 第一个检测有有效轮廓，第二个掩膜退化返回空
        mock_m2s.side_effect = [[seg0], []]

        preds = [np.zeros((1, 38, 8400), dtype=np.float32), np.zeros((1, 32, 160, 160), dtype=np.float32)]

        from schemas.inference_context import PreprocMeta
        meta = PreprocMeta(r=1.0, dw=0, dh=0, src_shape=(480, 640, 3),
                           ori_img=np.zeros((480, 640, 3), dtype=np.uint8))

        result = mock_model.post_process(preds, meta)

        np.testing.assert_array_equal(
            mock_process_mask.call_args.args[2], mock_pred[:, :4]
        )

        # 退化的检测被丢弃，有效检测保留，且各字段长度严格对齐
        assert len(result.mask_polygons) == 1
        assert len(result.boxes) == 1
        assert len(result.scores) == 1
        assert len(result.class_ids) == 1
        assert len(result.masks) == 1

    def test_class_name_mapping(self, mock_model):
        """测试类别名称映射"""
        assert mock_model.id2name[0] == "class_0"
        assert mock_model.id2name[79] == "class_79"


class TestYoloOnnxInferIntegration:
    """集成测试（需要真实模型文件时跳过）"""

    @pytest.mark.skip(reason="需要真实模型文件")
    def test_full_inference_pipeline(self):
        """完整推理流程测试"""
        from services.yolo import YoloOnnxInfer

        model = YoloOnnxInfer("path/to/model.onnx", nc=80)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = model(image)

        assert result is not None
