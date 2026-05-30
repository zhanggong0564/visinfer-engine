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


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
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


def rec_filename(stem: str, station: str, tag: str | None = None) -> str:
    """rec 数据集图片名: <stem>_rec_<station>[_<tag>].png（强制 PNG 无损）

    每条 crop_ocr strip 仅一行文字，故文件名不含 shape 序号。
    多外扩比例增广时用 tag（如 e10/e15/e20）区分同一 strip 的不同变体。
    """
    suffix = f"_{tag}" if tag else ""
    return f"{stem}_rec_{station}{suffix}.png"


def _expand_tag(ratio: float, multi: bool) -> str | None:
    """多变体时为外扩比例生成文件名标签（0.15→'e15'）；单变体返回 None（不加标签）。"""
    return f"e{int(round(ratio * 100)):02d}" if multi else None


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


def write_det_split(samples: list[Sample], det_dir: Path, split_filename: str) -> dict[str, int]:
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
    """rec 转换需要的方向分类组件 + 裁剪配置。

    expand_ratios: 裁剪前对标注框向外扩的比例列表（以短边/文字高度为基准，四边等距）。
        人工框贴合文字、偏紧，推理 dt_polys 四周带少量边距，用此项补齐二者差异。
        给多个比例（如 0.10/0.15/0.20）时，每条 strip 按每个比例各裁一份，
        既覆盖推理边距波动、又成倍增广训练数据。典型单值 0.10~0.20；含 0 表示原始紧框。
    """

    text_orient: object  # paddleocr.TextLineOrientationClassification
    min_crop_size: int = 4
    expand_ratios: tuple[float, ...] = (0.15,)
    orient_batch_size: int = 64


def build_rec_pipeline(config, expand_ratios: tuple[float, ...] = (0.15,), orient_batch_size: int = 64) -> RecPipeline:
    """根据 PanelLabelConfig 实例化方向分类组件。

    参数 config: 一个具备 orient_model_path 属性的对象。
    参数 expand_ratios: 裁剪框外扩比例列表（见 RecPipeline.expand_ratios）。
    参数 orient_batch_size: 方向分类批量大小（跨样本统一推理时的分块尺寸）。
    """
    from paddleocr import TextLineOrientationClassification

    text_orient = TextLineOrientationClassification(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir=config.orient_model_path,
    )
    return RecPipeline(
        text_orient=text_orient,
        expand_ratios=tuple(expand_ratios),
        orient_batch_size=orient_batch_size,
    )


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


def _expand_box(box: np.ndarray, ratio: float) -> np.ndarray:
    """沿框自身的宽/高方向，向外各扩 ratio×短边 的距离。

    box: _get_mini_boxes 返回的有序 4 点 [tl, tr, br, bl]（float32）。
    margin 以「短边（文字高度）」为基准、与框宽无关，避免 DB unclip 在宽框上沿
    宽度方向过扩（最初产物四周大留白的病根）。ratio<=0 时原样返回。
    """
    if ratio <= 0:
        return box
    tl, tr, br, bl = box
    u = tr - tl  # 宽方向
    v = bl - tl  # 高方向
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return box
    u = u / nu
    v = v / nv
    d = ratio * min(nu, nv)
    return np.array(
        [
            tl - u * d - v * d,
            tr + u * d - v * d,
            br + u * d + v * d,
            bl - u * d + v * d,
        ],
        dtype=np.float32,
    )


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


def extract_rec_crops(
    sample: Sample, expand_ratios: tuple[float, ...], min_crop_size: int = 4
) -> list[tuple[np.ndarray, str, float]] | dict:
    """从 JSON 标注裁剪 rec 图像（**未做方向校正**，纯几何，无需模型）。

    几何来自人工标注框。人工框贴合文字偏紧，推理 dt_polys 四周带少量边距，故裁剪前
    按 expand_ratios（短边比例、四边等距）做轻量外扩补齐差异；给多个比例时每个比例
    各裁一份用于增广。流程：JSON points → minAreaRect 规整 4 点 → _expand_box 外扩 →
    _crop_by_quad 透视裁剪。

    返回 [(crop, transcription, ratio), ...]（未旋转）或 {"skip_reason": str}。
    """
    data = json.loads(sample.json_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes", []) or []
    if not shapes:
        return {"skip_reason": "empty-shape"}

    # 单行假设：取第一个有效（非 difficult、description 非空、点数足够）的 shape
    shape = None
    for s in shapes:
        if s.get("difficult"):
            continue
        description = s.get("description")
        if description is None or str(description).strip() == "":
            continue
        if len(s.get("points") or []) < 3:
            continue
        shape = s
        break
    if shape is None:
        return {"skip_reason": "empty-desc"}

    image = cv2.imread(str(sample.image_path))
    if image is None:
        return {"skip_reason": "image-unreadable"}

    pts = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in shape["points"]], dtype=np.int32)

    # minAreaRect of raw polygon → 规整的 4 点 quad
    box, sside = _get_mini_boxes(pts.reshape(-1, 1, 2))
    if sside < 3:
        return {"skip_reason": "no-valid-crop"}
    box = np.array(box, dtype=np.float32)

    # 每个外扩比例各裁一份（CropByPolys 透视裁剪，与 OCRPipeline._crop_by_polys quad 一致）
    text = str(shape.get("description", ""))
    crops: list[tuple[np.ndarray, str, float]] = []
    for ratio in expand_ratios:
        crop = _crop_by_quad(image, _expand_box(box, ratio))
        if crop is None or crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
            continue
        crops.append((crop, text, ratio))
    if not crops:
        return {"skip_reason": "no-valid-crop"}
    return crops


def _orient_and_rotate(crops: list[np.ndarray], text_orient, batch_size: int) -> list[np.ndarray]:
    """对裁剪批量做方向分类并按需旋转 180°；按 batch_size 分块以控显存。

    方向分类是逐图独立分类（推理期 BN 用 running stats），故分块不影响单图结果，
    与整体一次推理等价。
    """
    rotated: list[np.ndarray] = []
    for i in range(0, len(crops), max(1, batch_size)):
        chunk = crops[i : i + max(1, batch_size)]
        results = text_orient.predict(chunk)
        for crop, res in zip(chunk, results):
            angle = int(res["class_ids"][0])
            rotated.append(cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop)
    return rotated


def process_rec_sample(sample: Sample, pipeline: RecPipeline) -> list[tuple[np.ndarray, str, float]] | dict:
    """对单个 sample 裁剪并方向校正 rec 图像（每条 strip 一行文字）。

    = extract_rec_crops（几何）+ 方向分类旋转。批量转换走 write_rec_split 的跨样本批处理；
    此函数主要供单样本/测试使用。返回 [(rotated_crop, transcription, ratio), ...] 或 {"skip_reason": str}。
    """
    crops = extract_rec_crops(sample, pipeline.expand_ratios, pipeline.min_crop_size)
    if isinstance(crops, dict):
        return crops
    rotated = _orient_and_rotate([c for c, _, _ in crops], pipeline.text_orient, pipeline.orient_batch_size)
    return [(rot, text, ratio) for rot, (_crop, text, ratio) in zip(rotated, crops)]


def write_rec_split(
    samples: list[Sample], pipeline: RecPipeline, rec_dir: Path, split_filename: str
) -> tuple[dict[str, int], list[str]]:
    """跑 rec pipeline 并写出。

    分三阶段：① 逐样本几何裁剪（extract_rec_crops，无模型）并收集全 split 的裁剪；
    ② 跨样本按 orient_batch_size 分块统一做方向分类（减少 predict 调用、提速）；
    ③ 旋转后写盘 + 写标签行。

    返回 (stats, kept_transcriptions)。
    stats keys: kept, empty-shape, empty-desc, image-unreadable, no-valid-crop
    """
    images_out = rec_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    label_path = rec_dir / split_filename

    stats: dict[str, int] = {
        "kept": 0,
        "empty-shape": 0,
        "empty-desc": 0,
        "image-unreadable": 0,
        "no-valid-crop": 0,
    }
    multi = len(pipeline.expand_ratios) > 1

    # ① 收集全 split 裁剪（未旋转）
    pending: list[tuple[Sample, np.ndarray, str, float]] = []
    for s in samples:
        result = extract_rec_crops(s, pipeline.expand_ratios, pipeline.min_crop_size)
        if isinstance(result, dict):
            reason = result["skip_reason"]
            stats[reason] = stats.get(reason, 0) + 1
            continue
        for crop, text, ratio in result:
            pending.append((s, crop, text, ratio))

    # ② 跨样本批量方向分类 + 旋转
    rotated_all = _orient_and_rotate(
        [crop for _, crop, _, _ in pending], pipeline.text_orient, pipeline.orient_batch_size
    )

    # ③ 写盘 + 标签
    transcriptions: list[str] = []
    with label_path.open("w", encoding="utf-8") as f:
        for (s, _crop, text, ratio), rotated in zip(pending, rotated_all):
            new_name = rec_filename(s.original_stem, s.station_code, _expand_tag(ratio, multi))
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


def resplit_det_split(samples: list[Sample], det_dir: Path, split_filename: str) -> dict[str, int]:
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


def resplit_rec_split(samples: list[Sample], rec_dir: Path, split_filename: str) -> tuple[dict[str, int], list[str]]:
    """重写 rec/<split_filename>。text 直接取自 LabelMe description；只引用已存在的 PNG。

    多外扩比例增广时每条 strip 在磁盘上有多份变体（<stem>_rec_<station>[_e*].png），
    此处按前缀 glob 收集全部变体、每份各写一行。

    返回 (stats, transcriptions)。stats = {"kept": int, "missing": int}（missing 按 strip 计）。
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
            # 单行假设：取第一个有效（非 difficult、description 非空）的 shape
            text = None
            for shape in shapes:
                if shape.get("difficult"):
                    continue
                description = shape.get("description")
                if description is None or str(description).strip() == "":
                    continue
                text = str(description)
                break
            if text is None:
                continue
            # 收集该 strip 的所有变体 PNG（无标签 + e* 标签），remainder 守卫避免站点名前缀误匹配
            prefix = f"{s.original_stem}_rec_{s.station_code}"
            variants = sorted(
                p
                for p in images_out.glob(f"{prefix}*.png")
                if (rem := p.stem[len(prefix) :]) == "" or rem.startswith("_")
            )
            if not variants:
                missing += 1
                continue
            for p in variants:
                f.write(f"images/{p.name}\t{text}\n")
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
    parser.add_argument("--mode", choices=("all", "det-only", "rec-only", "resplit-only"), default="all")
    parser.add_argument(
        "--rec-expand-ratios",
        "-r",
        type=float,
        nargs="+",
        default=[0.15],
        help="rec 裁剪框外扩比例（短边/文字高度为基准，四边等距）；"
        "可给多个值，每条 strip 按每个比例各裁一份做增广（如 0.05 0.15 0.25）；0 表示原始紧框",
    )
    parser.add_argument(
        "--rec-batch-size",
        "-b",
        type=int,
        default=64,
        help="rec 方向分类批量大小（跨样本统一推理时的分块尺寸）",
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
        pipeline = build_rec_pipeline(
            cfg, expand_ratios=tuple(args.rec_expand_ratios), orient_batch_size=args.rec_batch_size
        )
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
