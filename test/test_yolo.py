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
        with patch("services.yolo.letterbox") as mock_letterbox:
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

        with patch("services.yolo.letterbox") as mock_letterbox:
            mock_letterbox.return_value = (np.ones((640, 640, 3), dtype=np.uint8) * 255, 1.0, 0, 0)

            tensor, _ = mock_model.preprocess(white_image)

        # 归一化后应接近 1.0
        assert np.allclose(tensor, 1.0)

    @patch("services.yolo.non_max_suppression_v8")
    @patch("services.yolo.scale_boxes")
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
        mock_scale_boxes.return_value = mock_pred[:, :4]

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

    @patch("services.yolo.non_max_suppression_v8")
    @patch("services.yolo.scale_boxes")
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
        mock_scale_boxes.return_value = mock_pred[:, :4]
        mock_xywhr.return_value = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])

        preds = [np.random.rand(1, 84, 8400)]

        from schemas.inference_context import PreprocMeta
        meta = PreprocMeta(r=1.0, dw=0, dh=0, src_shape=(480, 640, 3),
                           ori_img=np.zeros((480, 640, 3), dtype=np.uint8))

        result = mock_model.post_process(preds, meta)

        # 验证调用了旋转框转换
        mock_xywhr.assert_called_once()

    @patch("services.yolo.non_max_suppression_v8")
    @patch("services.yolo.scale_boxes")
    def test_post_process_empty_detection(self, mock_scale_boxes, mock_nms, mock_model):
        """测试无检测结果"""
        # 空预测
        mock_pred = np.empty((0, 6))
        mock_nms.return_value = [mock_pred]
        mock_scale_boxes.return_value = mock_pred[:, :4]

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
    @patch("services.yolo.non_max_suppression_v8")
    @patch("services.yolo.scale_boxes")
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
        mock_scale_boxes.return_value = mock_pred[:, :4]
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
