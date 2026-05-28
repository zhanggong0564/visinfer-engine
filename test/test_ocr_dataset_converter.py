"""tools/convert_ocr_dataset_to_ppocr.py 的单元测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.convert_ocr_dataset_to_ppocr import Sample, find_samples


def _write_labelme(json_path: Path, image_name: str, shapes: list[dict]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "version": "3.3.9",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": 100,
                "imageWidth": 400,
                "description": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _make_shape(description: str = "ABC", difficult: bool = False) -> dict:
    return {
        "label": "text",
        "score": 1.0,
        "points": [[10.0, 10.0], [90.0, 10.0], [90.0, 30.0], [10.0, 30.0]],
        "group_id": 0,
        "description": description,
        "difficult": difficult,
        "shape_type": "polygon",
        "flags": None,
        "attributes": {},
        "kie_linking": [],
    }


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """构造两侧两个工位、共 3 张样本的最小数据集。"""
    root = tmp_path / "dataset"
    img_bytes = b"\x89PNG\r\n\x1a\n"  # 仅作为占位，不会被解码（除非测试明确读图）

    # 交流侧/J46/IMG_1
    j46 = root / "ac" / "J46" / "crop_ocr"
    (j46 / "images").mkdir(parents=True)
    (j46 / "images" / "IMG_1.jpg").write_bytes(img_bytes)
    _write_labelme(j46 / "jsons" / "IMG_1.json", "IMG_1.jpg", [_make_shape("J46-A/PW-1")])

    # 交流侧/J46/IMG_2
    (j46 / "images" / "IMG_2.jpg").write_bytes(img_bytes)
    _write_labelme(j46 / "jsons" / "IMG_2.json", "IMG_2.jpg", [_make_shape("J46-B/PW-2")])

    # 直流侧/T1/IMG_3
    t1 = root / "dc" / "T1" / "crop_ocr"
    (t1 / "images").mkdir(parents=True)
    (t1 / "images" / "IMG_3.jpg").write_bytes(img_bytes)
    _write_labelme(t1 / "jsons" / "IMG_3.json", "IMG_3.jpg", [_make_shape("T1-X/KM-1")])

    return root


def test_find_samples_collects_all(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    assert len(samples) == 3
    stations = sorted(s.station_code for s in samples)
    assert stations == ["J46", "J46", "T1"]
    stems = sorted(s.original_stem for s in samples)
    assert stems == ["IMG_1", "IMG_2", "IMG_3"]


def test_sample_image_path_exists(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    for s in samples:
        assert isinstance(s, Sample)
        assert s.image_path.exists()
        assert s.json_path.exists()
        assert s.station_code in {"J46", "T1"}
