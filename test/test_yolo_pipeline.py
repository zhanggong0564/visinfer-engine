import numpy as np

from schemas.inference_context import PreprocMeta
from services.yolo_ops import (
    prepare_yolo_input,
    restore_yolo_boxes,
    run_yolo_nms,
)
from services.vision.boxes import scale_boxes
from services.vision.preprocessing import letterbox


def test_prepare_yolo_input_matches_existing_pipeline():
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    expected_image, r, dw, dh = letterbox(
        im=image, auto=False, new_shape=(8, 10)
    )
    expected = np.stack([expected_image])
    expected = expected[..., ::-1].transpose((0, 3, 1, 2))
    expected = np.ascontiguousarray(expected).astype(np.float32) / 255.0

    tensor, meta = prepare_yolo_input(image, (8, 10))

    np.testing.assert_array_equal(tensor, expected)
    assert tensor.shape == (1, 3, 8, 10)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert meta == PreprocMeta(r=r, dw=dw, dh=dh, src_shape=image.shape)


def test_run_yolo_nms_forwards_exact_arguments(monkeypatch):
    captured = {}
    sentinel = [np.array([[1.0, 2.0, 3.0, 4.0, 0.9, 2.0]])]

    def fake_nms(prediction, **kwargs):
        captured["prediction"] = prediction
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "services.yolo_ops.non_max_suppression_v8", fake_nms
    )
    prediction = np.zeros((1, 6, 2), dtype=np.float32)

    result = run_yolo_nms(
        prediction,
        task="rect",
        conf_threshold=0.4,
        iou_threshold=0.6,
        classes=[1, 2],
        agnostic=True,
        nc=3,
    )

    assert result is sentinel
    assert captured == {
        "prediction": prediction,
        "task": "rect",
        "conf_thres": 0.4,
        "iou_thres": 0.6,
        "classes": [1, 2],
        "agnostic": True,
        "multi_label": False,
        "nc": 3,
    }


def test_restore_yolo_boxes_matches_existing_scale_without_mutation():
    detections = np.array(
        [[2.0, 1.0, 8.0, 7.0, 0.9, 1.0]], dtype=np.float32
    )
    original = detections.copy()
    expected = detections.copy()
    expected[:, :4] = scale_boxes(
        (8, 10), expected[:, :4], (3, 5), xywh=False
    )

    restored = restore_yolo_boxes(detections, (8, 10), (3, 5, 3))

    np.testing.assert_allclose(restored, expected)
    np.testing.assert_array_equal(detections, original)


def test_restore_yolo_boxes_preserves_empty_shape():
    detections = np.empty((0, 6), dtype=np.float32)
    restored = restore_yolo_boxes(detections, (8, 10), (3, 5, 3))
    assert restored.shape == (0, 6)
