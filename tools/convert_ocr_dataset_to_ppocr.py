"""PPOCRv5 检测 + 识别数据集统一转换工具。

源数据：LabelMe `crop_ocr/{images,jsons}`
产物：<dst>/det/{images,train.txt,val.txt} + <dst>/rec/{images,train.txt,val.txt,dict.txt}

用法：
    python tools/convert_ocr_dataset_to_ppocr.py \
        --src <labelme-root> --dst <out-root> \
        --val-ratio 0.1 --seed 42 --mode all
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


def det_filename(stem: str, station: str, ext: str) -> str:
    """det 数据集图片名: <stem>_det_<station>.<ext>"""
    return f"{stem}_det_{station}.{ext.lstrip('.')}"


def rec_filename(stem: str, station: str) -> str:
    """rec 数据集图片名: <stem>_rec_<station>.png（强制 PNG 无损）"""
    return f"{stem}_rec_{station}.png"


def build_det_annotation(json_path: Path) -> list[dict]:
    """从 LabelMe JSON 提取 PPOCR det 标注；空 description 或 difficult 写 '###'。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    items: list[dict] = []
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 3:
            continue
        text = shape.get("description")
        if text is None or str(text).strip() == "" or shape.get("difficult"):
            text = "###"
        rounded = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in points]
        items.append({"transcription": str(text), "points": rounded})
    return items


def write_det_split(
    samples: list[Sample], det_dir: Path, split_filename: str
) -> dict[str, int]:
    """把 samples 写为 PPOCR det 格式（images/ 拷贝 + train.txt/val.txt 行）。

    返回统计 dict: {"kept": int, "empty_shape": int}
    """
    images_out = det_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    label_path = det_dir / split_filename

    kept = 0
    empty_shape = 0
    with label_path.open("w", encoding="utf-8") as f:
        for s in samples:
            ann = build_det_annotation(s.json_path)
            if not ann:
                empty_shape += 1
                continue
            ext = s.image_path.suffix.lstrip(".")
            new_name = det_filename(s.original_stem, s.station_code, ext)
            dst_image = images_out / new_name
            if not dst_image.exists():
                shutil.copy2(s.image_path, dst_image)
            f.write(f"images/{new_name}\t{json.dumps(ann, ensure_ascii=False)}\n")
            kept += 1
    return {"kept": kept, "empty_shape": empty_shape}


@dataclass
class RecPipeline:
    """rec 转换需要的 3 个 paddleocr 组件 + 配置 snapshot。"""

    text_det: object  # paddleocr.TextDetection
    text_orient: object  # paddleocr.TextLineOrientationClassification
    crop_by_polys: object  # paddlex CropByPolys
    min_crop_size: int = 4


def build_rec_pipeline(config) -> RecPipeline:
    """根据 PanelLabelConfig 实例化 paddleocr 组件。

    参数 config: 一个具备 text_det_*, orient_model_path,
    text_recognition_model_path（仅作占位，本工具不用 rec 模型）等属性的对象。
    """
    from paddleocr import TextDetection, TextLineOrientationClassification
    from paddlex.inference.pipelines.components import CropByPolys

    text_det = TextDetection(
        model_name="PP-OCRv5_server_det",
        limit_side_len=config.text_det_limit_side_len,
        limit_type=config.text_det_limit_type,
        thresh=config.text_det_thresh,
        box_thresh=config.text_det_box_thresh,
        unclip_ratio=config.text_det_unclip_ratio,
        input_shape=config.text_det_input_shape,
    )
    text_orient = TextLineOrientationClassification(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir=config.orient_model_path,
    )
    crop_by_polys = CropByPolys(det_box_type="quad")
    return RecPipeline(text_det=text_det, text_orient=text_orient, crop_by_polys=crop_by_polys)


def process_rec_sample(sample: Sample, pipeline: RecPipeline) -> tuple[np.ndarray, str] | dict:
    """对单个 sample 跑 rec pipeline，返回 (rotated_crop, transcription)。

    失败/跳过时返回 {"skip_reason": str}。

    Pipeline 与 services/panel_label/panel_label_detect.py:97-149 严格一致：
        TextDetection → max-area polygon → CropByPolys(quad) →
        TextLineOrientationClassification → cv2.rotate(180) if angle==1
    """
    # 短路：description 为空 / difficult 直接跳过
    data = json.loads(sample.json_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes", []) or []
    if not shapes:
        return {"skip_reason": "empty-shape"}
    shape = shapes[0]
    if shape.get("difficult"):
        return {"skip_reason": "difficult"}
    description = shape.get("description")
    if description is None or str(description).strip() == "":
        return {"skip_reason": "empty-desc"}

    image = cv2.imread(str(sample.image_path))
    if image is None:
        return {"skip_reason": "image-unreadable"}

    det_result = pipeline.text_det.predict(image)
    dt_polys = det_result[0]["dt_polys"]
    if dt_polys is None or len(dt_polys) == 0:
        return {"skip_reason": "det-zero"}

    areas = [cv2.contourArea(np.array(p, dtype=np.float32).reshape(-1, 2)) for p in dt_polys]
    best_poly = dt_polys[int(np.argmax(areas))]

    crops = list(pipeline.crop_by_polys(image, [best_poly]))
    if not crops:
        return {"skip_reason": "crop-failed"}
    crop = crops[0]
    if crop is None or crop.shape[0] < pipeline.min_crop_size or crop.shape[1] < pipeline.min_crop_size:
        return {"skip_reason": "too-small"}

    orient_result = pipeline.text_orient.predict([crop])
    angle = int(orient_result[0]["class_ids"][0])
    rotated = cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop

    return rotated, str(description)
