"""PPOCRv5 检测 + 识别数据集统一转换工具。

源数据：LabelMe `crop_ocr/{images,jsons}`
产物：<dst>/det/{images,train.txt,val.txt} + <dst>/rec/{images,train.txt,val.txt,dict.txt}

用法：
    python tools/convert_ocr_dataset_to_ppocr.py \
        --src <labelme-root> --dst <out-root> \
        --val-ratio 0.1 --seed 42 --mode all
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def rec_filename(stem: str, station: str, shape_idx: int) -> str:
    """rec 数据集图片名: <stem>_rec_<station>_<shape_idx>.png（强制 PNG 无损）"""
    return f"{stem}_rec_{station}_{shape_idx}.png"


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
    """rec 转换需要的方向分类组件 + unclip/裁剪配置。"""

    text_orient: object  # paddleocr.TextLineOrientationClassification
    unclip_ratio: float = 2.0
    min_crop_size: int = 4


def build_rec_pipeline(config) -> RecPipeline:
    """根据 PanelLabelConfig 实例化方向分类组件。

    参数 config: 一个具备 orient_model_path, text_det_unclip_ratio 等属性的对象。
    """
    from paddleocr import TextLineOrientationClassification

    text_orient = TextLineOrientationClassification(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir=config.orient_model_path,
    )
    return RecPipeline(
        text_orient=text_orient,
        unclip_ratio=getattr(config, "text_det_unclip_ratio", 2.0),
    )


def _unclip_poly(points: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """对多边形做 unclip 扩展，与 PPOCR DBPostProcess.unclip 逻辑一致。"""
    import pyclipper

    area = cv2.contourArea(points)
    length = cv2.arcLength(points, True)
    distance = area * unclip_ratio / length
    offset = pyclipper.PyclipperOffset()
    # pyclipper 要求路径为 (N,2) 的整数点序列；contourArea/arcLength 可吃 (N,1,2)，此处需展平
    offset.AddPath(points.reshape(-1, 2).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        expanded = np.array(offset.Execute(distance))
    except ValueError:
        expanded = np.array(offset.Execute(distance)[0])
    return expanded


def _get_mini_boxes(contour: np.ndarray) -> tuple[list, float]:
    """取最小外接矩形的 4 个角点，与 PPOCR DBPostProcess.get_mini_boxes 一致。"""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def _get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """透视变换裁剪，与 CropByPolys.get_rotate_crop_image 一致。"""
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    if img_crop_width <= 0 or img_crop_height <= 0:
        return None
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def _crop_by_quad(img: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """对 quad 类型的 4 点做 minAreaRect + 透视裁剪，与 CropByPolys(quad) 一致。"""
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    box_points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if box_points[1][1] > box_points[0][1]:
        index_a, index_d = 0, 1
    else:
        index_a, index_d = 1, 0
    if box_points[3][1] > box_points[2][1]:
        index_b, index_c = 2, 3
    else:
        index_b, index_c = 3, 2

    ordered = [box_points[index_a], box_points[index_b], box_points[index_c], box_points[index_d]]
    return _get_rotate_crop_image(img, np.array(ordered))


def process_rec_sample(sample: Sample, pipeline: RecPipeline) -> list[tuple[np.ndarray, str, int]] | dict:
    """对单个 sample 从 JSON 标注裁剪 rec 图像。

    流程：JSON points → unclip → minAreaRect → 透视裁剪 → 方向分类旋转。
    返回 [(rotated_crop, transcription, shape_idx), ...] 或 {"skip_reason": str}。
    """
    data = json.loads(sample.json_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes", []) or []
    if not shapes:
        return {"skip_reason": "empty-shape"}

    image = cv2.imread(str(sample.image_path))
    if image is None:
        return {"skip_reason": "image-unreadable"}

    results: list[tuple[np.ndarray, str, int]] = []
    all_crops: list[tuple[np.ndarray, int]] = []

    for shape_idx, shape in enumerate(shapes):
        if shape.get("difficult"):
            continue
        description = shape.get("description")
        if description is None or str(description).strip() == "":
            continue

        points = shape.get("points") or []
        if len(points) < 3:
            continue

        pts = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in points], dtype=np.int32)

        # unclip 扩展
        expanded = _unclip_poly(pts.reshape(-1, 1, 2), pipeline.unclip_ratio)
        if len(expanded) == 0:
            continue
        expanded = expanded.reshape(-1, 2)

        # minAreaRect → 4 角点（与 DBPostProcess.boxes_from_bitmap 一致）
        box, sside = _get_mini_boxes(expanded.reshape(-1, 1, 2))
        if sside < 3:
            continue
        box = np.array(box, dtype=np.float32)

        # 透视裁剪（与 CropByPolys quad 模式一致）
        crop = _crop_by_quad(image, box)
        if crop is None or crop.shape[0] < pipeline.min_crop_size or crop.shape[1] < pipeline.min_crop_size:
            continue

        all_crops.append((crop, shape_idx))

    if not all_crops:
        return {"skip_reason": "no-valid-crop"}

    # 批量方向分类
    orient_results = pipeline.text_orient.predict([c for c, _ in all_crops])
    for (crop, shape_idx), orient_res in zip(all_crops, orient_results):
        angle = int(orient_res["class_ids"][0])
        rotated = cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop
        text = str(shapes[shape_idx].get("description", ""))
        results.append((rotated, text, shape_idx))

    return results


def write_rec_split(
    samples: list[Sample], pipeline: RecPipeline, rec_dir: Path, split_filename: str
) -> tuple[dict[str, int], list[str]]:
    """跑 rec pipeline 并写出。

    返回 (stats, kept_transcriptions)。
    stats keys: kept, empty-shape, image-unreadable, no-valid-crop
    """
    images_out = rec_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    label_path = rec_dir / split_filename

    stats: dict[str, int] = {
        "kept": 0,
        "empty-shape": 0,
        "image-unreadable": 0,
        "no-valid-crop": 0,
    }
    transcriptions: list[str] = []

    with label_path.open("w", encoding="utf-8") as f:
        for s in samples:
            result = process_rec_sample(s, pipeline)
            if isinstance(result, dict):
                reason = result["skip_reason"]
                stats[reason] = stats.get(reason, 0) + 1
                continue
            for rotated, text, shape_idx in result:
                new_name = rec_filename(s.original_stem, s.station_code, shape_idx)
                dst = images_out / new_name
                ok = cv2.imwrite(str(dst), rotated)
                if not ok:
                    stats["no-valid-crop"] = stats.get("no-valid-crop", 0) + 1
                    continue
                f.write(f"images/{new_name}\t{text}\n")
                transcriptions.append(text)
                stats["kept"] += 1
    return stats, transcriptions


def write_dict(transcriptions: list[str], dict_path: Path) -> int:
    """收集 transcriptions 里所有 unique 字符，按 sorted 顺序一行一个写入。

    返回字符数。
    """
    chars: set[str] = set()
    for t in transcriptions:
        chars.update(t)
    sorted_chars = sorted(chars)
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(sorted_chars) + "\n" if sorted_chars else ""
    dict_path.write_text(content, encoding="utf-8")
    return len(sorted_chars)


def resplit_det_split(
    samples: list[Sample], det_dir: Path, split_filename: str
) -> dict[str, int]:
    """重写 det/<split_filename>，只为 det/images/ 中已存在的图片写标签行。

    返回 {"kept": int, "missing": int}。missing = 样本在该 split 中但磁盘无对应图。
    """
    images_out = det_dir / "images"
    label_path = det_dir / split_filename
    label_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    missing = 0
    with label_path.open("w", encoding="utf-8") as f:
        for s in samples:
            ann = build_det_annotation(s.json_path)
            if not ann:
                continue
            ext = s.image_path.suffix.lstrip(".")
            new_name = det_filename(s.original_stem, s.station_code, ext)
            if not (images_out / new_name).exists():
                missing += 1
                continue
            f.write(f"images/{new_name}\t{json.dumps(ann, ensure_ascii=False)}\n")
            kept += 1
    return {"kept": kept, "missing": missing}


def resplit_rec_split(
    samples: list[Sample], rec_dir: Path, split_filename: str
) -> tuple[dict[str, int], list[str]]:
    """重写 rec/<split_filename>。text 直接取自 LabelMe description；只引用已存在的 PNG。

    返回 (stats, transcriptions)。stats = {"kept": int, "missing": int}。
    """
    images_out = rec_dir / "images"
    label_path = rec_dir / split_filename
    label_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    missing = 0
    transcriptions: list[str] = []
    with label_path.open("w", encoding="utf-8") as f:
        for s in samples:
            data = json.loads(s.json_path.read_text(encoding="utf-8"))
            shapes = data.get("shapes", []) or []
            for shape_idx, shape in enumerate(shapes):
                if shape.get("difficult"):
                    continue
                description = shape.get("description")
                if description is None or str(description).strip() == "":
                    continue
                text = str(description)
                new_name = rec_filename(s.original_stem, s.station_code, shape_idx)
                if not (images_out / new_name).exists():
                    missing += 1
                    continue
                f.write(f"images/{new_name}\t{text}\n")
                transcriptions.append(text)
                kept += 1
    return {"kept": kept, "missing": missing}, transcriptions


def _print_det_stats(label: str, stats: dict[str, int]) -> None:
    parts = [f"kept={stats['kept']}"]
    if stats["empty_shape"]:
        parts.append(f"empty_shape={stats['empty_shape']}")
    print(f"[{label}] " + " ".join(parts))


def _print_rec_stats(label: str, stats: dict[str, int]) -> None:
    skipped = {k: v for k, v in stats.items() if k != "kept" and v > 0}
    skip_part = " ".join(f"{k}={v}" for k, v in skipped.items())
    print(f"[{label}] kept={stats['kept']}" + (f" (skipped {skip_part})" if skip_part else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", choices=("all", "det-only", "rec-only", "resplit-only"), default="all"
    )
    args = parser.parse_args()

    samples = find_samples(args.src)
    print(f"[scan] found {len(samples)} samples under {args.src}")
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    print(f"[split] train={len(train_samples)} val={len(val_samples)}  seed={args.seed}")

    # resplit-only：仅重写 train/val txt，不跑模型也不拷图
    if args.mode == "resplit-only":
        det_dir = args.dst / "det"
        rec_dir = args.dst / "rec"
        if det_dir.exists():
            ts = resplit_det_split(train_samples, det_dir, "train.txt")
            vs = resplit_det_split(val_samples, det_dir, "val.txt")
            print(f"[det/train] kept={ts['kept']} missing={ts['missing']}")
            print(f"[det/val] kept={vs['kept']} missing={vs['missing']}")
        else:
            print("[resplit] det/ not found, skipping")
        if rec_dir.exists():
            ts, train_texts = resplit_rec_split(train_samples, rec_dir, "train.txt")
            vs, val_texts = resplit_rec_split(val_samples, rec_dir, "val.txt")
            print(f"[rec/train] kept={ts['kept']} missing={ts['missing']}")
            print(f"[rec/val] kept={vs['kept']} missing={vs['missing']}")
            n_chars = write_dict(train_texts + val_texts, rec_dir / "dict.txt")
            print(f"[dict] {n_chars} unique chars → {rec_dir / 'dict.txt'}")
        else:
            print("[resplit] rec/ not found, skipping")
        print(f"[done] output={args.dst}")
        return

    do_det = args.mode in ("all", "det-only")
    do_rec = args.mode in ("all", "rec-only")

    if do_det:
        det_dir = args.dst / "det"
        ts = write_det_split(train_samples, det_dir, "train.txt")
        vs = write_det_split(val_samples, det_dir, "val.txt")
        _print_det_stats("det/train", ts)
        _print_det_stats("det/val", vs)

    if do_rec:
        from vie_plugin_panel_label.config import PanelLabelConfig

        cfg = PanelLabelConfig()
        pipeline = build_rec_pipeline(cfg)
        rec_dir = args.dst / "rec"
        ts, train_texts = write_rec_split(train_samples, pipeline, rec_dir, "train.txt")
        vs, val_texts = write_rec_split(val_samples, pipeline, rec_dir, "val.txt")
        _print_rec_stats("rec/train", ts)
        _print_rec_stats("rec/val", vs)
        n_chars = write_dict(train_texts + val_texts, rec_dir / "dict.txt")
        print(f"[dict] {n_chars} unique chars → {rec_dir / 'dict.txt'}")

    print(f"[done] output={args.dst}")


if __name__ == "__main__":
    main()
