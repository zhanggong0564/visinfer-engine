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

        if not dt_polys:
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
