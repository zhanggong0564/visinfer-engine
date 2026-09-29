from dataclasses import asdict

import numpy as np
import pytest

from services.base import CoordinateSpace, Detection, DetectionResult, Region, xyxy_region
from services.tiled_yolo import (
    TiledPreprocMeta, deduplicate_detections, merge_tile_results, quarter_regions, region_box,
)
from schemas.inference_context import PreprocMeta


def detection(box, score=0.8, class_id=1):
    return Detection(xyxy_region(box), score, class_id, str(class_id))


def test_four_regions_and_total_overlap():
    assert [region_box(region) for region in quarter_regions(4000, 3000)] == [
        (0, 0, 1600, 2100), (0, 1900, 1600, 4000),
        (1400, 0, 3000, 2100), (1400, 1900, 3000, 4000),
    ]


def test_odd_dimensions_and_overlap():
    boxes = [region_box(region) for region in quarter_regions(9, 7, 3)]
    assert boxes == [(0, 0, 5, 6), (0, 3, 5, 9), (2, 0, 7, 6), (2, 3, 7, 9)]


@pytest.mark.parametrize('shape,overlap', [((20, 300), 200), ((1, 10), 0)])
def test_small_image_is_one_crop(shape, overlap):
    height, width = shape
    assert [region_box(region) for region in quarter_regions(height, width, overlap)] == [
        (0, 0, width, height)]


@pytest.mark.parametrize('shape,overlap', [((0, 10), 2), ((10, 0), 2), ((10, 10), -1)])
def test_invalid_crop_parameters(shape, overlap):
    with pytest.raises(ValueError):
        quarter_regions(*shape, overlap)


def test_composite_rank_can_choose_smaller_high_confidence_box():
    large = detection([0, 0, 10, 10], 0.61)
    confident = detection([0, 0, 9, 10], 0.95)
    assert deduplicate_detections([large, confident]) == [confident]


def test_composite_rank_preserves_complete_box_over_small_fragment():
    complete = detection([0, 0, 10, 10], 0.75)
    fragment = detection([0, 0, 3, 10], 0.99)
    assert deduplicate_detections([fragment, complete]) == [complete]


def test_equal_product_uses_confidence():
    large = detection([0, 0, 10, 10], 0.72)
    confident = detection([0, 0, 8, 10], 0.9)
    assert deduplicate_detections([large, confident]) == [confident]


def test_different_classes_and_adjacent_objects_survive():
    rows = [detection([0, 0, 10, 10]), detection([12, 0, 22, 10]),
            detection([0, 0, 10, 10], class_id=2)]
    assert len(deduplicate_detections(rows)) == 3


def test_discarded_box_cannot_chain_suppress_another_target():
    a = detection([0, 0, 10, 10], 0.95)
    b = detection([4, 0, 14, 10], 0.9)
    c = detection([12, 0, 22, 10], 0.8)
    assert deduplicate_detections([a, b, c]) == [a, c]


def test_tile_translation_retains_original_boxes_and_aligned_fields():
    local = detection([1, 2, 5, 8], 0.75)
    result = merge_tile_results([
        (xyxy_region([100, 200, 300, 400]), DetectionResult([local]))])
    final = result.detections[0]
    assert region_box(final.region) == (101, 202, 105, 208)
    assert final.region.space == CoordinateSpace.PIXEL
    assert (final.score, final.class_id, final.class_name) == (0.75, 1, '1')
    assert region_box(local.region) == (1, 2, 5, 8)


def test_tiled_metadata_serialization_keeps_explicit_pixel_regions():
    meta = TiledPreprocMeta((10, 20, 3), (
        (xyxy_region([0, 0, 20, 10]), PreprocMeta(1.0, 0, 0, (10, 20, 3))),))
    serialized = asdict(meta)
    assert serialized['tiles'][0][0]['space'] == CoordinateSpace.PIXEL
    assert serialized['tiles'][0][1]['src_shape'] == (10, 20, 3)


@pytest.mark.parametrize('box', [
    [9.999, 9.999, 20, 20],  # Tiny positive overlap still suppresses.
    [4, 0, 14, 10],
])
def test_any_positive_area_overlap_is_suppressed(box):
    preferred = detection([0, 0, 10, 10], 0.95)
    weaker = detection(box, 0.4)
    assert deduplicate_detections([weaker, preferred]) == [preferred]


@pytest.mark.parametrize('box', [
    [10, 0, 20, 10], [0, 10, 10, 20], [10, 10, 20, 20],
])
def test_edge_or_corner_contact_does_not_suppress(box):
    first = detection([0, 0, 10, 10], 0.95)
    second = detection(box, 0.8)
    assert deduplicate_detections([first, second]) == [first, second]


def test_reported_metal_overlap_keeps_larger_higher_confidence_box():
    preferred = detection(
        [244.08322334289548, 2787.85870552063, 375.4788208007812, 3019.251718521118],
        0.7844530940055847, class_id=2,
    )
    weaker = detection(
        [161.21679067611691, 2842.42130279541, 291.42417669296265, 3033.0217933654785],
        0.4515976309776306, class_id=2,
    )
    assert deduplicate_detections([weaker, preferred]) == [preferred]


def test_overlap_is_checked_after_tile_translation():
    first = detection([0, 0, 10, 10], 0.95)
    second = detection([0, 0, 10, 10], 0.8)
    merged = merge_tile_results([
        (xyxy_region([0, 0, 20, 20]), DetectionResult([first])),
        (xyxy_region([9, 0, 29, 20]), DetectionResult([second])),
    ])
    assert len(merged.detections) == 1
    assert region_box(merged.detections[0].region) == (0, 0, 10, 10)


@pytest.mark.parametrize('det', [
    Detection(Region(((0, 0), (1, 1)), CoordinateSpace.NORMALIZED), 0.8, 1, '1'),
    detection([0, 0, 0, 1]),
    detection([0, 0, 1, 1], float('nan')),
    detection([0, 0, float('inf'), 1]),
    Detection(xyxy_region([0, 0, 1, 1]), 0.8, 1, '1', mask=np.ones((1, 1))),
])
def test_invalid_detection_geometry_or_score(det):
    with pytest.raises(ValueError):
        deduplicate_detections([det])


def test_empty_results():
    assert deduplicate_detections([]) == []
    assert merge_tile_results([]).detections == []
