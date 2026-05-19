'''
@Author       : zhanggong
@Date         : 2026-05-19
@FilePath     : export_ocr_dataset.py
@Description  : 批量导出线标 OCR 训练数据 — crop_ocrdet / crop_cls / crop_ocr
'''

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.panel_label.panel_label_detect import PanelLabelDetect
from services.panel_label.utils import Points_to_Mask
from config import settings
from paddleocr import TextDetection, TextLineOrientationClassification, TextRecognition


def crop_poly_from_image(image, poly):
    """从图像中裁剪多边形区域的最小外接矩形（非旋转）"""
    poly = np.array(poly, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(poly)
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
    return image[y1:y2, x1:x2]


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


def process_image(image_src, stem, detectors, output_dirs):
    """处理单张图片，导出所有 OCR 中间数据"""
    det_model = detectors["yolo"]
    text_det = detectors["text_det"]
    text_cls = detectors["text_cls"]
    text_rec = detectors["text_rec"]

    results = det_model.infer(image_src)
    class_ids = np.array(results.class_ids)
    mask_polygons = np.array(results.mask_polygons, dtype=object)
    points_line = mask_polygons[class_ids == 0] if 0 in class_ids else []

    if len(points_line) == 0:
        print(f"  [{stem}] no line labels detected, skip")
        return

    mask_rois, sorted_idxs = Points_to_Mask(image_src, points_line)

    total_crops = 0

    for roi_idx, mask_roi in enumerate(mask_rois):
        h, w = mask_roi.shape[:2]
        if h < 10 or w < 10:
            continue

        det_results = text_det.predict(mask_roi)
        dt_polys = det_results[0].get("dt_polys")
        if dt_polys is None or len(dt_polys) == 0:
            continue

        for dt_idx, poly in enumerate(dt_polys):
            crop_img = crop_poly_from_image(mask_roi, poly)
            if crop_img.size == 0:
                continue

            # 保存 crop_ocrdet
            crop_name = f"{stem}_{roi_idx}_{dt_idx}.jpg"
            cv2.imwrite(str(output_dirs["ocrdet"] / crop_name), crop_img)

            cls_results = text_cls.predict(crop_img)
            label_name = cls_results[0].get("label_names", ["0_degree"])[0]

            # 保存 crop_cls（按方向分类分文件夹）
            cls_subdir = label_name
            cv2.imwrite(str(output_dirs["cls"] / cls_subdir / crop_name), crop_img)

            # 方向校正
            if label_name == "180_degree":
                crop_img = np.rot90(crop_img, 2)

            # 保存 crop_ocr（识别模型输入）
            cv2.imwrite(str(output_dirs["ocr"] / crop_name), crop_img)

            rec_results = text_rec.predict(crop_img)
            rec_text = rec_results[0].get("rec_texts", ["?"])[0]
            print(f"  [{crop_name}]: cls={label_name}, rec={rec_text}")

            total_crops += 1

    print(f"  [{stem}] exported {total_crops} crops")


def init_detectors(args):
    """初始化所有模型，返回字典"""
    det_model = PanelLabelDetect(
        settings.panel_label.model_path,
        confThreshold=args.conf,
        nmsThreshold=args.nms,
        task="seg",
    )
    text_det = TextDetection(model_name="PP-OCRv5_server_det")
    text_cls = TextLineOrientationClassification(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir=settings.panel_label.orient_model_path,
    )
    text_rec = TextRecognition(
        model_name="PP-OCRv5_server_rec",
        model_dir=settings.panel_label.text_recognition_model_path,
    )
    return {
        "yolo": det_model,
        "text_det": text_det,
        "text_cls": text_cls,
        "text_rec": text_rec,
    }


def process_single_dir(image_dir, output_base, detectors):
    """处理单个目录（模式 B：扁平结构）"""
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
        image_src = cv2.imread(str(image_path))
        if image_src is None:
            print(f"  [{image_path.name}] 无法读取, skip")
            continue
        process_image(image_src, image_path.stem, detectors, output_dirs)

    print(f"{'='*60}")
    print("完成")


def main():
    parser = argparse.ArgumentParser(description="导出面板线标 OCR 训练数据")
    parser.add_argument("--input-dir", required=True, help="输入图片文件夹路径")
    parser.add_argument("--output-dir", default="./output/ocr_dataset", help="输出根目录")
    parser.add_argument("--conf", type=float, default=settings.panel_label.confThreshold, help="YOLO 置信度阈值")
    parser.add_argument("--nms", type=float, default=settings.panel_label.nmsThreshold, help="YOLO NMS 阈值")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"[错误] 输入目录不存在: {input_dir}")
        sys.exit(1)

    output_root = Path(args.output_dir)
    detectors = init_detectors(args)

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
                output_base=output_root / model_name,
                detectors=detectors,
            )
    else:
        process_single_dir(
            image_dir=input_dir,
            output_base=output_root,
            detectors=detectors,
        )


if __name__ == "__main__":
    main()
