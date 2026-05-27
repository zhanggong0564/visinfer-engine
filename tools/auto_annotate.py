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
