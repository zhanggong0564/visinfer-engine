import pytest

from schemas.inspection import InspectionVerdict
from services.base import (
    CoordinateSpace,
    Detection,
    OCRToken,
    Region,
    xyxy_region,
)


def test_inspection_verdict_maps_legacy_values():
    assert InspectionVerdict.from_value(True) is InspectionVerdict.PASS
    assert InspectionVerdict.from_value(" false ") is InspectionVerdict.FAIL
    assert InspectionVerdict.from_value("REVIEW") is InspectionVerdict.REVIEW
    assert InspectionVerdict.from_value("unknown") is None
    assert InspectionVerdict.REVIEW.legacy_status == "false"


def test_ocr_token_exposes_explicit_region_and_scores():
    region = xyxy_region([1, 2, 5, 6], CoordinateSpace.NORMALIZED)
    token = OCRToken(
        text="FQ001811",
        region=region,
        recognition_score=0.91,
        detection_score=0.82,
    )

    assert token.polygon == [[1.0, 2.0], [5.0, 2.0], [5.0, 6.0], [1.0, 6.0]]
    assert token.center == (3.0, 4.0)
    assert token.confidence == 0.91
    assert token.region.space is CoordinateSpace.NORMALIZED


def test_detection_uses_common_region():
    detection = Detection(
        region=xyxy_region((1, 2, 3, 4)),
        score=0.8,
        class_id=2,
        class_name="label",
    )

    assert detection.region.center == (2.0, 3.0)
    assert detection.class_name == "label"


def test_empty_region_has_no_center():
    with pytest.raises(ValueError, match="cannot be empty"):
        Region(polygon=()).center
