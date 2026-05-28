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


from tools.convert_ocr_dataset_to_ppocr import split_samples, build_det_annotation, det_filename, rec_filename


def test_split_deterministic_with_seed(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    # 3 个样本，val_ratio=0.34 → n_val=1
    train1, val1 = split_samples(samples, val_ratio=0.34, seed=42)
    train2, val2 = split_samples(samples, val_ratio=0.34, seed=42)
    assert [s.original_stem for s in train1] == [s.original_stem for s in train2]
    assert [s.original_stem for s in val1] == [s.original_stem for s in val2]
    assert len(train1) + len(val1) == 3
    assert len(val1) == 1


def test_split_zero_val(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    train, val = split_samples(samples, val_ratio=0.0, seed=42)
    assert len(val) == 0
    assert len(train) == 3


def test_det_filename_no_chinese() -> None:
    name = det_filename("IMG_1", "J46", "jpg")
    assert name == "IMG_1_det_J46.jpg"
    assert all(ord(c) < 128 for c in name), "filename must be pure ASCII"


def test_rec_filename_format() -> None:
    assert rec_filename("IMG_2", "T1") == "IMG_2_rec_T1.png"


def test_build_det_annotation_normal(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    _write_labelme(j, "a.jpg", [_make_shape("HELLO")])
    ann = build_det_annotation(j)
    assert len(ann) == 1
    assert ann[0]["transcription"] == "HELLO"
    assert ann[0]["points"] == [[10, 10], [90, 10], [90, 30], [10, 30]]


def test_build_det_annotation_empty_description_becomes_hash(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    _write_labelme(j, "a.jpg", [_make_shape("")])
    ann = build_det_annotation(j)
    assert ann[0]["transcription"] == "###"


def test_build_det_annotation_difficult_becomes_hash(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    _write_labelme(j, "a.jpg", [_make_shape("text", difficult=True)])
    ann = build_det_annotation(j)
    assert ann[0]["transcription"] == "###"


def test_build_det_annotation_skips_short_points(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    bad = _make_shape("x")
    bad["points"] = [[0.0, 0.0], [10.0, 10.0]]  # 仅 2 个点
    _write_labelme(j, "a.jpg", [bad])
    assert build_det_annotation(j) == []


from tools.convert_ocr_dataset_to_ppocr import write_det_split


def test_write_det_split_creates_images_and_labels(mini_dataset: Path, tmp_path: Path) -> None:
    samples = find_samples(mini_dataset)
    train, val = split_samples(samples, val_ratio=0.34, seed=42)

    det_dir = tmp_path / "out" / "det"
    train_stats = write_det_split(train, det_dir, "train.txt")
    val_stats = write_det_split(val, det_dir, "val.txt")

    # 标签文件 + 图片
    assert (det_dir / "train.txt").exists()
    assert (det_dir / "val.txt").exists()
    images = list((det_dir / "images").iterdir())
    assert len(images) == 3
    for img in images:
        assert "_det_" in img.name
        assert img.suffix in {".jpg", ".jpeg", ".png", ".bmp"}
        assert all(ord(c) < 128 for c in img.name)

    # 标签行格式
    line = (det_dir / "train.txt").read_text(encoding="utf-8").splitlines()[0]
    path_part, ann_part = line.split("\t")
    assert path_part.startswith("images/")
    ann = json.loads(ann_part)
    assert isinstance(ann, list) and "transcription" in ann[0]

    assert train_stats["kept"] + val_stats["kept"] == 3


def test_write_det_split_skips_empty_shape_image(tmp_path: Path) -> None:
    # 构造一个没有有效 shape 的样本
    root = tmp_path / "data"
    p = root / "ac" / "X1" / "crop_ocr"
    (p / "images").mkdir(parents=True)
    (p / "images" / "IMG_E.jpg").write_bytes(b"x")
    _write_labelme(p / "jsons" / "IMG_E.json", "IMG_E.jpg", shapes=[])
    samples = find_samples(root)

    det_dir = tmp_path / "out" / "det"
    stats = write_det_split(samples, det_dir, "train.txt")
    assert stats["kept"] == 0
    assert stats["empty_shape"] == 1
    # 既然全跳过，train.txt 仍应存在但为空
    assert (det_dir / "train.txt").exists()
    assert (det_dir / "train.txt").read_text() == ""


from tools.convert_ocr_dataset_to_ppocr import write_dict


def test_write_dict_unique_sorted(tmp_path: Path) -> None:
    path = tmp_path / "dict.txt"
    n = write_dict(["ABC", "BCD", "A1"], path)
    chars = path.read_text(encoding="utf-8").splitlines()
    assert chars == sorted(set("ABCD1"))
    assert n == len(chars)
