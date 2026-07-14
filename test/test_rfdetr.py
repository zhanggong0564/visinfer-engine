"""RF-DETR ONNX 推理器单元测试。"""

import numpy as np
import pytest
import warnings

from schemas.inference_context import PreprocMeta


def _model(confidence=0.5):
    from services.rfdetr import RFDetrOnnxInfer

    model = RFDetrOnnxInfer.__new__(RFDetrOnnxInfer)
    model._input_model_shape = [1, 3, 2, 2]
    model.nc = 2
    model.confThreshold = confidence
    model.task = "seg"
    model.id2name = {0: "line", 1: "QFU"}
    return model


def _meta(height=10, width=20):
    return PreprocMeta(
        r=1.0,
        dw=0.0,
        dh=0.0,
        src_shape=(height, width, 3),
        ori_img=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _outputs():
    dets = np.array(
        [[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]], dtype=np.float32
    )
    labels = np.array(
        [[[2.0, -2.0, -8.0], [-3.0, -3.0, 8.0]]], dtype=np.float32
    )
    masks = np.ones((1, 2, 2, 2), dtype=np.float32)
    return [dets, labels, masks]


def test_preprocess_converts_bgr_to_normalized_rgb_nchw():
    model = _model()
    image = np.array([[[0, 0, 255]]], dtype=np.uint8)

    tensor, meta = model.preprocess(image)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)
    assert tensor[0, 1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224)
    assert meta.src_shape == (1, 1, 3)
    assert (meta.r, meta.dw, meta.dh) == (1.0, 0.0, 0.0)


def test_post_process_filters_background_and_keeps_mask_alignment(monkeypatch):
    model = _model()
    monkeypatch.setattr(
        "services.rfdetr.masks2segments_with_boxes",
        lambda mask, box: [np.array([[5, 2], [15, 2], [15, 7]], dtype=np.float32)],
    )

    result = model.post_process(_outputs(), _meta())

    assert result.class_ids == [0]
    assert result.class_names == ["line"]
    assert result.scores == pytest.approx([1.0 / (1.0 + np.exp(-2.0))])
    assert np.asarray(result.boxes) == pytest.approx(
        np.array([[5.0, 2.5, 15.0, 7.5]])
    )
    assert len(result.masks) == len(result.mask_polygons) == 1
    assert result.masks[0].shape == (10, 20)


def test_preprocess_rejects_non_bgr_image():
    model = _model()

    with pytest.raises(ValueError, match="BGR"):
        model.preprocess(np.zeros((2, 2), dtype=np.uint8))


def test_post_process_clips_extreme_logits_before_sigmoid(monkeypatch):
    model = _model()
    outputs = _outputs()
    outputs[1][0, 0, 0] = -1000.0
    monkeypatch.setattr(
        "services.rfdetr.masks2segments_with_boxes",
        lambda mask, box: [np.array([[5, 2], [15, 2], [15, 7]], dtype=np.float32)],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = model.post_process(outputs, _meta())

    assert result.boxes == []
