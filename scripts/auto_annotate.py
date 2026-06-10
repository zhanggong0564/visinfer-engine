'''
@Description  : 自动推理转标注工具 — 对输入图片运行 PaddleOCR，输出 LabelMe JSON
@Usage        : python tools/auto_annotate.py --input <images_dir>
'''

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

# 将项目根目录加入 path，保证能 import services/config 等
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 纯函数：组装 LabelMe JSON 结构
# ---------------------------------------------------------------------------


def _build_labelme_json(
    shapes: list,
    image_filename: str,
    image_height: int,
    image_width: int,
) -> dict:
    """
    将检测结果组装为 LabelMe v3.3.9 格式的 dict。

    Args:
        shapes: 每个元素是包含 label/score/points/description 的 dict
        image_filename: 图片文件名（仅文件名，不含路径）
        image_height: 图片高度（像素）
        image_width:  图片宽度（像素）

    Returns:
        符合 LabelMe 格式的 dict，可直接 json.dump
    """
    normalized_shapes = []
    for s in shapes:
        normalized_shapes.append(
            {
                "label": s["label"],
                "score": s["score"],
                "points": s["points"],
                "group_id": 0,
                "description": s["description"],
                "difficult": False,
                "shape_type": "polygon",
                "flags": None,
                "attributes": {},
                "kie_linking": [],
            }
        )

    return {
        "version": "3.3.9",
        "flags": {},
        "shapes": normalized_shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "description": "",
    }


from paddleocr import TextDetection, TextLineOrientationClassification, TextRecognition
from paddlex.inference.pipelines.components import CropByPolys

# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------


class AutoAnnotator:
    """
    批量自动标注器：一次加载 PaddleOCR 三阶段模型，对每张图片独立推理，
    将检测框和识别文字转为 LabelMe JSON。
    """

    # 与 PanelLabelConfig 保持一致的默认参数
    _DEFAULT_ORIENT_PATH = "./weights/panel_label/textline_ori_lcnet_v3"
    _DEFAULT_REC_PATH = "./weights/panel_label/text_rec_plane_ppocrv5s_v3"

    def __init__(
        self,
        orient_model_path: str = _DEFAULT_ORIENT_PATH,
        rec_model_path: str = _DEFAULT_REC_PATH,
        score_thresh: float = 0.7,
        # TextDetection 超参（与 PanelLabelConfig 一致）
        text_det_limit_side_len: int = 480,
        text_det_limit_type: str = "max",
        text_det_thresh: float = 0.3,
        text_det_box_thresh: float = 0.3,
        text_det_unclip_ratio: float = 2.0,
        text_det_input_shape: list = None,
        # TextRecognition 超参
        text_rec_input_shape: list = None,
    ):
        self.score_thresh = score_thresh

        self.text_det = TextDetection(
            model_name="PP-OCRv5_server_det",
            limit_side_len=text_det_limit_side_len,
            limit_type=text_det_limit_type,
            thresh=text_det_thresh,
            box_thresh=text_det_box_thresh,
            unclip_ratio=text_det_unclip_ratio,
            input_shape=text_det_input_shape,
        )
        self.text_ori = TextLineOrientationClassification(
            model_name="PP-LCNet_x1_0_textline_ori",
            model_dir=orient_model_path,
        )
        self.text_rec = TextRecognition(
            model_name="PP-OCRv5_server_rec",
            model_dir=rec_model_path,
            input_shape=text_rec_input_shape,
        )
        self._crop = CropByPolys(det_box_type="quad")

    def infer_image(self, image: np.ndarray, image_filename: str) -> dict:
        """
        对单张图片执行 PaddleOCR 三阶段推理，返回 LabelMe JSON dict。

        每张裁剪图片只有一个文本行。若 TextDetection 检测出多个框，则为误检测，
        只保留面积最大的一个（与 OCRPipeline 策略一致）。

        Args:
            image:          BGR 格式 numpy 数组
            image_filename: 仅文件名（如 "img.jpg"），写入 JSON imagePath 字段

        Returns:
            LabelMe 格式 dict（可直接 json.dump），shapes 最多含 1 条记录
        """
        h, w = image.shape[:2]

        # Stage 1: Text Detection
        det_result = self.text_det.predict(image)
        dt_polys = det_result[0]["dt_polys"] if det_result else []

        if dt_polys is None or len(dt_polys) == 0:
            return _build_labelme_json(
                shapes=[],
                image_filename=image_filename,
                image_height=h,
                image_width=w,
            )

        # 每张图片只有一个文本行，检测出多个则为误检测，只保留面积最大的
        if len(dt_polys) > 1:
            areas = [cv2.contourArea(np.array(poly, dtype=np.float32).reshape(-1, 2)) for poly in dt_polys]
            dt_polys = [dt_polys[int(np.argmax(areas))]]

        poly = dt_polys[0]

        # Crop detected text region
        crops = list(self._crop(image, [poly]))
        if not crops:
            return _build_labelme_json(
                shapes=[],
                image_filename=image_filename,
                image_height=h,
                image_width=w,
            )

        crop = crops[0]

        # Stage 2: Text Line Orientation
        orient_results = self.text_ori.predict([crop])
        angle = int(orient_results[0]["class_ids"][0])

        # 旋转修正（angle == 1 → 180°）
        rotated_crop = cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop

        # Stage 3: Text Recognition
        rec_results = self.text_rec.predict([rotated_crop])
        rec = rec_results[0]
        rec_text = rec.get("rec_text", "")
        rec_score = float(rec.get("rec_score", 0.0))

        # 分数低于阈值则置空字符串
        description = rec_text if rec_score >= self.score_thresh else ""

        points = [[float(p[0]), float(p[1])] for p in poly]
        shapes = [
            {
                "label": "text",
                "score": rec_score,
                "points": points,
                "description": description,
            }
        ]

        return _build_labelme_json(
            shapes=shapes,
            image_filename=image_filename,
            image_height=h,
            image_width=w,
        )

    def process_dir(self, input_dir: Path, overwrite: bool = False) -> None:
        """
        批量处理 input_dir 中的所有图片，将 LabelMe JSON 写入同层 jsons/ 目录。

        Args:
            input_dir: 包含图片的目录（如 crop_ocr/images/）
            overwrite: True 时覆盖已存在的 JSON；False（默认）时跳过
        """
        input_dir = Path(input_dir)
        jsons_dir = input_dir.parent / "jsons"
        jsons_dir.mkdir(parents=True, exist_ok=True)

        # 扫描所有图片（大小写不敏感）
        suffixes = {".jpg", ".jpeg", ".png"}
        image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in suffixes)

        if not image_paths:
            print(f"[WARNING] 未找到图片：{input_dir}")
            return

        # 尝试导入 tqdm，不可用时降级为 print
        try:
            from tqdm import tqdm

            iterator = tqdm(image_paths, desc="标注中", unit="img")
        except ImportError:
            iterator = image_paths

        for img_path in iterator:
            json_path = jsons_dir / (img_path.stem + ".json")

            if json_path.exists() and not overwrite:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[WARNING] 无法读取图片，跳过：{img_path.name}")
                continue

            labelme_dict = self.infer_image(image, img_path.name)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(labelme_dict, f, ensure_ascii=False, indent=2)

    def process_root_dir(self, root_dir: Path, overwrite: bool = False) -> None:
        """
        自动遍历 root_dir 下所有产品类型子目录，完成以下操作：
          1. 在 <product>/crop_ocr/ 下创建 images/ 子目录（若不存在）
          2. 将 crop_ocr/ 直接层的图片移入 images/
          3. 对 images/ 运行 OCR 推理，JSON 输出到 crop_ocr/jsons/

        输入目录结构（原始）：
            root_dir/
            ├── B4N4/
            │   └── crop_ocr/
            │       ├── a.jpg
            │       └── b.jpg
            └── FU211-213/
                └── crop_ocr/
                    └── c.jpg

        处理后目录结构：
            root_dir/
            ├── B4N4/
            │   └── crop_ocr/
            │       ├── images/   ← 图片移入这里
            │       └── jsons/    ← JSON 输出到这里
            └── FU211-213/
                └── crop_ocr/
                    ├── images/
                    └── jsons/

        Args:
            root_dir: 数据根目录（如 中压线标数据/）
            overwrite: True 时覆盖已存在的 JSON；False（默认）时跳过
        """
        import shutil

        root_dir = Path(root_dir)
        suffixes = {".jpg", ".jpeg", ".png"}

        crop_ocr_dirs = sorted(p for p in root_dir.glob("*/crop_ocr") if p.is_dir())

        if not crop_ocr_dirs:
            print(f"[WARNING] 未找到任何 <product>/crop_ocr 子目录：{root_dir}")
            return

        print(f"[INFO] 共找到 {len(crop_ocr_dirs)} 个产品类型目录")

        for crop_ocr_dir in crop_ocr_dirs:
            product_type = crop_ocr_dir.parent.name
            images_dir = crop_ocr_dir / "images"
            images_dir.mkdir(exist_ok=True)

            # 将 crop_ocr/ 直接层的图片移入 images/
            loose_images = [p for p in crop_ocr_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
            if loose_images:
                for img in loose_images:
                    shutil.move(str(img), str(images_dir / img.name))
                print(f"[INFO] {product_type}：已移动 {len(loose_images)} 张图片 → images/")

            print(f"\n[INFO] 处理产品类型：{product_type}  →  {images_dir}")
            self.process_dir(images_dir, overwrite=overwrite)
            print(f"[INFO] {product_type} 完成，JSON 已写入：{crop_ocr_dir / 'jsons'}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "自动推理转标注工具：对 images/ 目录中的图片运行 PaddleOCR，输出 LabelMe JSON 到 jsons/\n\n"
            "单目录模式：--input <product>/crop_ocr/images/\n"
            "批量模式：  --root <数据根目录>  （自动遍历所有 <product>/crop_ocr/images/）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--input",
        "-i",
        type=Path,
        help="单目录模式：输入图片目录（如 B4N4/crop_ocr/images/）",
    )
    mode_group.add_argument(
        "--root",
        "-r",
        type=Path,
        help="批量模式：数据根目录（如 中压线标数据/），自动遍历所有 <product>/crop_ocr/images/",
    )

    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.7,
        help="OCR 识别置信度阈值，低于此值的文字置为空字符串（默认 0.7）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="覆盖已存在的 JSON 文件（默认跳过）",
    )
    # 模型路径（高级选项，通常使用默认值）
    parser.add_argument(
        "--orient-model-path",
        default=AutoAnnotator._DEFAULT_ORIENT_PATH,
        help="TextLineOrientationClassification 模型目录",
    )
    parser.add_argument(
        "--rec-model-path",
        default=AutoAnnotator._DEFAULT_REC_PATH,
        help="TextRecognition 模型目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("[INFO] 初始化模型（首次运行可能自动下载）...")
    annotator = AutoAnnotator(
        orient_model_path=args.orient_model_path,
        rec_model_path=args.rec_model_path,
        score_thresh=args.score_thresh,
    )

    if args.root is not None:
        if not args.root.is_dir():
            print(f"[ERROR] 根目录不存在：{args.root}")
            sys.exit(1)
        print(f"[INFO] 批量模式，根目录：{args.root}")
        annotator.process_root_dir(args.root, overwrite=args.overwrite)
        print("\n[INFO] 全部完成！")
    else:
        if not args.input.is_dir():
            print(f"[ERROR] 输入目录不存在：{args.input}")
            sys.exit(1)
        print(f"[INFO] 开始处理：{args.input}")
        annotator.process_dir(args.input, overwrite=args.overwrite)
        print(f"[INFO] 完成！JSON 已写入：{args.input.parent / 'jsons'}")
