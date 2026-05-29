'''
@Author       : zhanggong
@Date         : 2026-05-19
@FilePath     : export_ocr_dataset.py
@Description  : 批量导出线标 OCR 训练数据 — crop_ocrdet / crop_cls / crop_ocr
                从 LabelMe JSON 读取 line 标签轮廓点，跳过 YOLO 检测，后处理与 OCRPipeline 一致
'''

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.panel_label.utils import Points_to_Mask
from config import settings
from paddleocr import TextDetection, TextLineOrientationClassification, TextRecognition
from paddlex.inference.pipelines.components import CropByPolys

# 所有导入完成后，抑制推理过程的刷屏日志（VisionLogger 单例会在导入时添加 INFO handler）
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="WARNING", colorize=True)


def create_output_dirs(base_dir):
    """在指定目录下创建 crop_ocrdet / crop_cls / crop_ocr 子目录"""
    dirs = {
        "ocrdet": base_dir / "crop_ocr_det",
        "cls": base_dir / "crop_cls",
        "ocr": base_dir / "crop_ocr",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for label in ["0", "1"]:
        (dirs["cls"] / label).mkdir(parents=True, exist_ok=True)
    return dirs


def load_line_polygons_from_json(json_path):
    """从 LabelMe JSON 中读取 label='line' 的多边形轮廓点"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polygons = []
    for shape in data.get("shapes", []):
        if shape.get("label") == "line" and shape.get("shape_type") == "polygon":
            polygons.append(np.array(shape["points"], dtype=np.float32))
    return polygons


def process_image(image_src, stem, points_line, detectors, output_dirs):
    """处理单张图片，导出所有 OCR 中间数据（与 OCRPipeline 流程一致）

    三个输出目录的语义：
    - crop_ocr:     ROI 完整图（mask_rois[i]）
    - crop_cls:     从 ROI 中按 dt_polys 经 CropByPolys 裁出的文本区域（未方向校正），按 0/1 分类存放
    - crop_ocr_det: 从 ROI 中按 dt_polys 经 CropByPolys 裁出的文本区域（方向校正后）
    """
    text_det = detectors["text_det"]
    text_cls = detectors["text_cls"]
    text_rec = detectors["text_rec"]
    crop_by_polys = detectors["crop_by_polys"]

    mask_rois, sorted_idxs = Points_to_Mask(image_src, points_line)

    # Stage 1: Text Detection — 每个 ROI 只保留面积最大的 poly
    all_dt_polys = []
    roi_to_idx = []
    for i, roi in enumerate(mask_rois):
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        det_result = text_det.predict(roi)
        dt_polys = det_result[0].get("dt_polys")
        if dt_polys is None or len(dt_polys) == 0:
            continue
        areas = [cv2.contourArea(np.array(p, dtype=np.float32).reshape(-1, 2)) for p in dt_polys]
        best_poly = dt_polys[int(np.argmax(areas))]
        all_dt_polys.append([best_poly])
        roi_to_idx.append(i)

    # Crop detected text regions（与 OCRPipeline 一致，使用 CropByPolys 透视变换裁剪）
    all_crops = []
    crop_roi_map = []
    for i, dt_polys in enumerate(all_dt_polys):
        roi_idx = roi_to_idx[i]
        roi = mask_rois[roi_idx]
        crops = list(crop_by_polys(roi, dt_polys))
        if crops:
            all_crops.append(crops[0])
            crop_roi_map.append(roi_idx)

    # 保存 crop_ocr：ROI 完整图
    for roi_idx in crop_roi_map:
        crop_name = f"{stem}_{roi_idx}.jpg"
        cv2.imwrite(str(output_dirs["ocr"] / crop_name), mask_rois[roi_idx])

    # Stage 2: Text Line Orientation（批量预测）
    total_crops = 0
    if all_crops:
        orient_results = text_cls.predict(all_crops)
        angles = [int(r["class_ids"][0]) for r in orient_results]

        for crop_idx, (crop, angle) in enumerate(zip(all_crops, angles)):
            roi_idx = crop_roi_map[crop_idx]
            crop_name = f"{stem}_{roi_idx}.jpg"

            # 保存 crop_cls：未方向校正的裁剪，按 0/1 分类存放
            cls_dir = output_dirs["cls"] / str(angle)
            cls_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cls_dir / crop_name), crop)

            # 方向校正
            rotated_crop = cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop

            # 保存 crop_ocr_det：方向校正后的裁剪
            cv2.imwrite(str(output_dirs["ocrdet"] / crop_name), rotated_crop)

            # Stage 3: Text Recognition
            rec_result = text_rec.predict(rotated_crop)
            rec_text = rec_result[0]["rec_text"]
            rec_score = rec_result[0]["rec_score"]
            if isinstance(rec_text, list):
                rec_text = rec_text[0] if rec_text else ""
            label_name = "180_degree" if angle == 1 else "0_degree"
            print(f"  [{crop_name}]: cls={label_name}, rec={rec_text} ({rec_score:.2f})")
            total_crops += 1

    print(f"  [{stem}] exported {total_crops} crops")


def init_detectors():
    """初始化 OCR 模型（参数与 OCRPipeline 一致，无需 YOLO）"""
    cfg = settings.panel_label
    text_det = TextDetection(
        model_name="PP-OCRv5_server_det",
        limit_side_len=cfg.text_det_limit_side_len,
        limit_type=cfg.text_det_limit_type,
        thresh=cfg.text_det_thresh,
        box_thresh=cfg.text_det_box_thresh,
        unclip_ratio=cfg.text_det_unclip_ratio,
    )
    text_cls = TextLineOrientationClassification(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir=cfg.orient_model_path,
    )
    text_rec = TextRecognition(
        model_name="PP-OCRv5_server_rec",
        model_dir=cfg.text_recognition_model_path,
    )
    crop_by_polys = CropByPolys(det_box_type="quad")
    return {
        "text_det": text_det,
        "text_cls": text_cls,
        "text_rec": text_rec,
        "crop_by_polys": crop_by_polys,
    }


def process_single_dir(image_dir, json_dir, output_base, detectors):
    """处理单个目录：从 image_dir 读图片，从 json_dir 读对应 LabelMe JSON"""
    image_paths = sorted(image_dir.rglob("*.jpg")) + sorted(image_dir.rglob("*.JPG"))
    if not image_paths:
        print(f"[跳过] 无 jpg 图片: {image_dir}")
        return

    output_dirs = create_output_dirs(output_base)
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_base}")
    print(f"图片数量: {len(image_paths)}")
    print(f"{'='*60}")

    for image_path in image_paths:
        json_path = json_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            print(f"  [{image_path.name}] 无对应 JSON, skip")
            continue

        points_line = load_line_polygons_from_json(json_path)
        if len(points_line) == 0:
            print(f"  [{image_path.name}] JSON 中无 line 标签, skip")
            continue

        image_src = cv2.imread(str(image_path))
        if image_src is None:
            print(f"  [{image_path.name}] 无法读取, skip")
            continue

        process_image(image_src, image_path.stem, points_line, detectors, output_dirs)

    print(f"{'='*60}")
    print("完成")


def main():
    parser = argparse.ArgumentParser(description="导出面板线标 OCR 训练数据（从 LabelMe JSON 读取轮廓）")
    parser.add_argument("--input-dir", required=True, help="输入根目录（含型号子目录，每个子目录有 images/ + jsons/）")
    parser.add_argument("--output-dir", default="./output/ocr_dataset", help="输出根目录")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"[错误] 输入目录不存在: {input_dir}")
        sys.exit(1)

    output_root = Path(args.output_dir)
    detectors = init_detectors()

    # 检测模式：如果子目录中包含 images/ → 模式 A（型号遍历）
    model_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and (d / "images").is_dir())

    if model_dirs:
        print(f"检测到 {len(model_dirs)} 个型号目录")
        print(f"输入根目录: {input_dir}")
        print(f"输出根目录: {output_root}")
        print(f"{'='*60}")

        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"\n--- 型号: {model_name} ---")
            process_single_dir(
                image_dir=model_dir / "images",
                json_dir=model_dir / "jsons",
                output_base=output_root / model_name,
                detectors=detectors,
            )
    else:
        # 扁平模式：input-dir 下直接有 images/ + jsons/
        json_dir = input_dir / "jsons"
        image_dir = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
        if not json_dir.is_dir():
            print(f"[错误] 找不到 jsons/ 目录: {json_dir}")
            sys.exit(1)
        process_single_dir(
            image_dir=image_dir,
            json_dir=json_dir,
            output_base=output_root,
            detectors=detectors,
        )


if __name__ == "__main__":
    main()
