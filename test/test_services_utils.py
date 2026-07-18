"""Focused vision-operation unit tests."""
import numpy as np
import pytest
from services.vision.nms import (
    box_area,
    numpy_nms,
    non_max_suppression_v8,
)
from services.vision.boxes import (
    xywh2xyxy,
    xywhr2xyxyxyxy,
    scale_boxes,
    clip_boxes,
    sort_boxes,
)
from services.vision.masks import (
    crop_mask,
    process_mask,
    scale_masks,
)
from services.vision.preprocessing import letterbox


class TestBoxArea:
    def test_single_box(self):
        area = box_area(np.array([[0, 0, 10, 10]]))
        assert area[0] == 100.0

    def test_multiple_boxes(self):
        areas = box_area(np.array([[0, 0, 5, 5], [1, 1, 4, 4]]))
        assert areas[0] == 25.0
        assert areas[1] == 9.0


class TestNumpyNms:
    def test_basic_suppression(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 48, 48],  # 与第一个高度重叠
            [100, 100, 150, 150],
        ], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        keep = numpy_nms(boxes, scores, 0.5)
        assert 1 in keep  # 第二高分

    def test_empty_boxes(self):
        keep = numpy_nms(np.empty((0, 4)), np.empty(0), 0.5)
        assert len(keep) == 0

    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = numpy_nms(boxes, scores, 0.5)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_iou_threshold_one_keeps_highest(self):
        """iou_thres=1 时，所有框都保留"""
        boxes = np.array([
            [10, 10, 30, 30],
            [12, 12, 28, 28],
        ], dtype=np.float32)
        scores = np.array([0.5, 0.8], dtype=np.float32)
        keep = numpy_nms(boxes, scores, 1.0)
        assert len(keep) == 2

    def test_handles_large_input_without_full_iou_matrix(self):
        boxes = np.array(
            [[i, 0, i + 10, 10] for i in range(1000)], dtype=np.float32
        )
        scores = np.linspace(1.0, 0.0, len(boxes), dtype=np.float32)

        keep = numpy_nms(boxes, scores, 0.5)

        assert keep.dtype == np.int64
        assert keep[0] == 0


class TestNonMaxSuppressionV8:
    def test_det_mode_empty_result(self):
        pred = np.zeros((1, 84, 8400), dtype=np.float32)
        result = non_max_suppression_v8(pred, task="det", conf_thres=0.25, iou_thres=0.45)
        assert len(result) == 1
        assert result[0].shape[0] == 0  # 无检测结果

    def test_det_mode_with_detection(self):
        pred = np.zeros((1, 84, 1), dtype=np.float32)
        # 设置一个高置信度检测: batch 0, box 0
        pred[0, :4, 0] = [0.5, 0.5, 0.2, 0.2]  # xywh
        pred[0, 4, 0] = 0.9  # conf
        pred[0, 5, 0] = 1.0  # class 0
        result = non_max_suppression_v8(pred, task="det", conf_thres=0.25, iou_thres=0.45)
        assert result[0].shape[0] >= 1


class TestXywh2Xyxy:
    def test_single_box(self):
        x = np.array([[100, 100, 50, 30]], dtype=np.float32)
        y = xywh2xyxy(x)
        assert y[0, 0] == 75.0   # x1 = cx - w/2
        assert y[0, 1] == 85.0   # y1 = cy - h/2
        assert y[0, 2] == 125.0  # x2 = cx + w/2
        assert y[0, 3] == 115.0  # y2 = cy + h/2

    def test_batch(self):
        x = np.array([
            [100, 100, 50, 30],
            [200, 150, 40, 20],
        ], dtype=np.float32)
        y = xywh2xyxy(x)
        assert y.shape == (2, 4)

    def test_preserves_extra_cols(self):
        x = np.array([[100, 100, 50, 30, 0.9, 1.0]], dtype=np.float32)
        y = xywh2xyxy(x)
        assert y[0, 4] == pytest.approx(0.9)
        assert y[0, 5] == pytest.approx(1.0)


class TestXywhr2Xyxyxyxy:
    def test_zero_rotation_square(self):
        center = np.array([[100, 100, 20, 40, 0]], dtype=np.float32)
        corners = xywhr2xyxyxyxy(center)
        assert corners.shape == (1, 4, 2)
        # 旋转0度，应形成矩形
        pts = corners[0]
        assert pts[0, 0] == pytest.approx(110.0)  # cx + w/2

    def test_batch(self):
        center = np.array([
            [100, 100, 20, 40, 0],
            [200, 150, 30, 20, 45],
        ], dtype=np.float32)
        corners = xywhr2xyxyxyxy(center)
        assert corners.shape == (2, 4, 2)


class TestScaleBoxes:
    def test_with_padding(self):
        boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        # img1_shape=(640,640), img0_shape=(480,640)
        scaled = scale_boxes((640, 640), boxes.copy(), (480, 640), padding=True)
        assert scaled.shape == (1, 4)

    def test_without_padding(self):
        boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        scaled = scale_boxes((640, 640), boxes.copy(), (480, 640), padding=False)
        assert scaled.shape == (1, 4)

    def test_with_explicit_ratio_pad(self):
        boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        scaled = scale_boxes(
            (640, 640), boxes.copy(), (480, 640),
            ratio_pad=((0.5, 0.5), (10, 10)), padding=True,
        )
        assert scaled.shape == (1, 4)


class TestClipBoxes:
    def test_boxes_within_bounds(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        clip_boxes(boxes, (100, 100))
        np.testing.assert_array_equal(boxes, np.array([[10, 10, 50, 50]]))

    def test_boxes_out_of_bounds(self):
        boxes = np.array([[-5, -5, 150, 150]], dtype=np.float32)
        clip_boxes(boxes, (100, 100))
        np.testing.assert_array_equal(boxes, np.array([[0, 0, 100, 100]]))


class TestSortBoxes:
    def test_empty_input_returns_empty_boxes_and_indices(self):
        assert sort_boxes([]) == ([], [])

    def test_orders_rows_top_to_bottom_and_each_row_left_to_right(self):
        boxes = [
            [60, 5, 80, 25, 0.9],
            [10, 8, 30, 28, 0.8],
            [50, 50, 70, 70, 0.7],
            [5, 52, 25, 72, 0.6],
        ]

        sorted_boxes, original_indices = sort_boxes(boxes)

        assert original_indices == [1, 0, 3, 2]
        assert sorted_boxes == [boxes[index] for index in original_indices]

    def test_zero_height_boxes_still_sort_left_to_right(self):
        boxes = [[20, 10, 30, 10], [5, 10, 15, 10]]

        sorted_boxes, original_indices = sort_boxes(boxes)

        assert original_indices == [1, 0]
        assert sorted_boxes == [boxes[1], boxes[0]]


class TestLetterbox:
    def test_output_maintains_aspect_ratio(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result, r, dw, dh = letterbox(img, new_shape=(640, 640))
        # 宽 > 高，宽度缩放到 640，高度按比例，auto=True 时 stride 对齐
        assert result.shape[2] == 3
        assert result.shape[1] == 640  # 宽正好 640

    def test_return_int(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result, r, dw, dh = letterbox(img, new_shape=640, return_int=True)
        assert result.dtype == np.uint8


class TestCropMask:
    def test_basic_crop(self):
        masks = np.ones((2, 10, 10), dtype=np.float32)
        boxes = np.array([
            [1, 1, 5, 5],
            [3, 3, 8, 8],
        ], dtype=np.float32)
        boxes = boxes[:, :, None]  # (n, 4, 1)
        result = crop_mask(masks, boxes)
        # broadcast: (2,10,10) * (2,1,10) * (2,10,1) → (2,2,10,10)
        # result[i,i] 对应第 i 个 mask 被第 i 个 box crop
        assert result[0, 0, 2, 2] == 1.0  # inside box0
        assert result[0, 0, 0, 0] == 0.0  # outside box0


class TestScaleMasks:
    def test_single_mask_keeps_channel_dimension(self):
        masks = np.ones((8, 10, 1), dtype=np.uint8)

        result = scale_masks(masks, shape=(20, 16), gain=1.0, dw=0, dh=0)

        assert result.shape == (16, 20, 1)

    def test_rejects_ambiguous_2d_input(self):
        masks = np.ones((8, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="HWC"):
            scale_masks(masks, shape=(20, 16), gain=1.0, dw=0, dh=0)

    def test_process_and_scale_single_mask_keeps_detection_axis(self):
        protos = np.ones((1, 1, 4, 4), dtype=np.float32) * 10
        masks_in = np.ones((1, 1), dtype=np.float32)
        bboxes = np.array([[0, 0, 8, 8]], dtype=np.float32)

        processed = process_mask(protos, masks_in, bboxes, shape=(8, 8))
        scaled = scale_masks(processed, shape=(16, 16), gain=1.0, dw=0, dh=0)

        assert processed.shape == (8, 8, 1)
        assert scaled.transpose(2, 0, 1).shape == (1, 16, 16)
