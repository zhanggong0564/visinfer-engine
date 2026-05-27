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
        normalized_shapes.append({
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
        })

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
    _DEFAULT_ORIENT_PATH = "./weights/panel_label/PP-LCNet_x1_0_textline_ori_v3"
    _DEFAULT_REC_PATH    = "./weights/panel_label/PP-OCRv5_server_rec_plane_infer_v3"

    def __init__(
        self,
        orient_model_path: str = _DEFAULT_ORIENT_PATH,
        rec_model_path: str    = _DEFAULT_REC_PATH,
        score_thresh: float    = 0.7,
        # TextDetection 超参（与 PanelLabelConfig 一致）
        text_det_limit_side_len: int   = 480,
        text_det_limit_type: str       = "max",
        text_det_thresh: float         = 0.3,
        text_det_box_thresh: float     = 0.3,
        text_det_unclip_ratio: float   = 2.0,
        text_det_input_shape: list     = None,
        # TextRecognition 超参
        text_rec_input_shape: list     = None,
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

        Args:
            image:          BGR 格式 numpy 数组
            image_filename: 仅文件名（如 "img.jpg"），写入 JSON imagePath 字段

        Returns:
            LabelMe 格式 dict（可直接 json.dump）
        """
        h, w = image.shape[:2]

        # Stage 1: Text Detection
        det_result = self.text_det.predict(image)
        dt_polys = det_result[0]["dt_polys"] if det_result else []

        if dt_polys is None or len(dt_polys) == 0:
            return _build_labelme_json(
                shapes=[], image_filename=image_filename,
                image_height=h, image_width=w,
            )

        # Crop detected text regions
        crops = []
        valid_polys = []
        for poly in dt_polys:
            cropped = list(self._crop(image, [poly]))
            if cropped:
                crops.append(cropped[0])
                valid_polys.append(poly)

        if not crops:
            return _build_labelme_json(
                shapes=[], image_filename=image_filename,
                image_height=h, image_width=w,
            )

        # Stage 2: Text Line Orientation
        orient_results = self.text_ori.predict(crops)
        angles = [int(r["class_ids"][0]) for r in orient_results]

        # 旋转修正（angle == 1 → 180°）
        rotated_crops = [
            cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop
            for crop, angle in zip(crops, angles)
        ]

        # Stage 3: Text Recognition
        rec_results = self.text_rec.predict(rotated_crops)

        shapes = []
        for poly, rec in zip(valid_polys, rec_results):
            rec_text  = rec.get("rec_text", "")
            rec_score = float(rec.get("rec_score", 0.0))

            # 分数低于阈值则置空字符串
            description = rec_text if rec_score >= self.score_thresh else ""

            points = [[float(p[0]), float(p[1])] for p in poly]
            shapes.append({
                "label": "text",
                "score": rec_score,
                "points": points,
                "description": description,
            })

        return _build_labelme_json(
            shapes=shapes, image_filename=image_filename,
            image_height=h, image_width=w,
        )

    def process_dir(self, input_dir: Path, overwrite: bool = False) -> None:
        """
        批量处理 input_dir 中的所有图片，将 LabelMe JSON 写入同层 jsons/ 目录。

        Args:
            input_dir: 包含图片的目录（Path 对象）
            overwrite: True 时覆盖已存在的 JSON；False（默认）时跳过
        """
        input_dir = Path(input_dir)
        jsons_dir = input_dir.parent / "jsons"
        jsons_dir.mkdir(parents=True, exist_ok=True)

        # 扫描所有图片（大小写不敏感）
        suffixes = {".jpg", ".jpeg", ".png"}
        image_paths = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in suffixes
        )

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


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="自动推理转标注工具：对 images/ 目录中的图片运行 PaddleOCR，输出 LabelMe JSON 到 jsons/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="输入图片目录（如 PE1_A/crop_ocr/images/）",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.7,
        help="OCR 识别置信度阈值，低于此值的文字置为空字符串",
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

    if not args.input.is_dir():
        print(f"[ERROR] 输入目录不存在：{args.input}")
        sys.exit(1)

    print(f"[INFO] 初始化模型（首次运行可能自动下载）...")
    annotator = AutoAnnotator(
        orient_model_path=args.orient_model_path,
        rec_model_path=args.rec_model_path,
        score_thresh=args.score_thresh,
    )

    print(f"[INFO] 开始处理：{args.input}")
    annotator.process_dir(args.input, overwrite=args.overwrite)
    print(f"[INFO] 完成！JSON 已写入：{args.input.parent / 'jsons'}")
