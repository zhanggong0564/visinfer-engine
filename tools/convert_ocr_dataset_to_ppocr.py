"""PPOCRv5 检测 + 识别数据集统一转换工具。

源数据：LabelMe `crop_ocr/{images,jsons}`
产物：<dst>/det/{images,train.txt,val.txt} + <dst>/rec/{images,train.txt,val.txt,dict.txt}

用法：
    python tools/convert_ocr_dataset_to_ppocr.py \
        --src <labelme-root> --dst <out-root> \
        --val-ratio 0.1 --seed 42 --mode all
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    """一个 LabelMe 样本。

    station_code: crop_ocr/ 的父目录名（C1, J46, T1, PE1_A ...）
    original_stem: 原图文件名去掉扩展名
    """

    json_path: Path
    image_path: Path
    station_code: str
    original_stem: str


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def find_samples(src_root: Path) -> list[Sample]:
    """递归扫描 src_root，收集所有 LabelMe 样本。

    匹配规则：`**/crop_ocr/jsons/*.json` + 同 stem 的图片。
    """
    samples: list[Sample] = []
    for json_path in sorted(src_root.rglob("crop_ocr/jsons/*.json")):
        images_dir = json_path.parent.parent / "images"
        stem = json_path.stem
        image_path = None
        for ext in _IMAGE_EXTS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            print(f"[warn] image not found for {json_path}")
            continue
        station_code = json_path.parent.parent.parent.name
        samples.append(
            Sample(
                json_path=json_path,
                image_path=image_path,
                station_code=station_code,
                original_stem=stem,
            )
        )
    return samples


def split_samples(
    samples: list[Sample], val_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """固定 seed 的 shuffle 后按 val_ratio 切分。

    返回 (train_samples, val_samples)。
    """
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio must be in [0,1], got {val_ratio}")
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]
