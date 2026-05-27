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
