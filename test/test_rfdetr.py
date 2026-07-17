"""RF-DETR ONNX 推理器单元测试。"""

import cv2
import numpy as np
import pytest
import warnings

from schemas.inference_context import PreprocMeta


def _model(confidence=0.5):
    from services.rfdetr import RFDetrInfer

    model = RFDetrInfer.__new__(RFDetrInfer)
    model._input_model_shape = [1, 3, 2, 2]
    model.nc = 2
    model.confThreshold = confidence
    model.task = "seg"
    model.id2name = {0: "line", 1: "QFU"}
    return model


def _polygon_iou(first, second):
    first = np.asarray(first, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(second, dtype=np.float32).reshape(-1, 2)
    first_area = cv2.contourArea(first)
    second_area = cv2.contourArea(second)
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


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


def test_post_process_polygons_only_skips_full_masks_and_preserves_polygon():
    full_model = _model()
    fast_model = _model()
    fast_model.mask_output = "polygons_only"
    meta = _meta(height=100, width=200)

    full = full_model.post_process(_outputs(), meta)
    fast = fast_model.post_process(_outputs(), meta)

    assert len(full.masks) == 1
    assert fast.masks == []
    assert fast.boxes == full.boxes
    assert fast.class_ids == full.class_ids
    assert fast.scores == pytest.approx(full.scores)
    assert len(fast.mask_polygons) == len(full.mask_polygons) == 1
    assert _polygon_iou(fast.mask_polygons[0], full.mask_polygons[0]) >= 0.98


def test_post_process_rejects_unknown_mask_output():
    model = _model()
    model.mask_output = "invalid"

    with pytest.raises(ValueError, match="mask_output"):
        model.post_process(_outputs(), _meta(height=100, width=200))


def test_polygons_only_fallback_warns_once(monkeypatch):
    model = _model()
    model.mask_output = "polygons_only"
    warnings = []
    monkeypatch.setattr(model, "_polygon_from_box_mask", lambda *args: [])
    monkeypatch.setattr(
        "services.rfdetr.vision_logger.warning",
        lambda message, *args: warnings.append((message, args)),
    )

    first = model.post_process(_outputs(), _meta(height=100, width=200))
    second = model.post_process(_outputs(), _meta(height=100, width=200))

    assert len(first.mask_polygons) == len(second.mask_polygons) == 1
    assert len(warnings) == 1


def test_local_mask_sampling_matches_full_resize_values():
    model = _model()
    raw_mask = np.random.default_rng(7).normal(
        0.0, 1.0, size=(19, 19)
    ).astype(np.float32)
    source_w, source_h = 800, 600
    box = np.array([213.4, 119.2, 517.8, 403.9], dtype=np.float32)

    local = model._resize_mask_roi(raw_mask, box, source_w, source_h)
    full = cv2.resize(
        raw_mask, (source_w, source_h), interpolation=cv2.INTER_LINEAR
    )[119:403, 213:517]

    np.testing.assert_allclose(local, full, atol=5e-4, rtol=0.0)


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
