'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:42:41
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-24 07:42:30
@FilePath     : get_crop.py
@Description  :
'''

import sys

sys.path.append("..")
import cv2
from utils import vision_logger
from services import rotate_points
import numpy as np
from vie_plugin_panel_label import OCRPipelineCrop
from vie_plugin_panel_label.config import PanelLabelConfig
from config import settings
from schemas import InputParamsBusiness
import json
from pathlib import Path


if __name__ == '__main__':
    type = "PE1-B"
    crop_dir = f"./demo/data/panel_label/{type}_crop"
    Path(crop_dir).mkdir(parents=True, exist_ok=True)
    image_paths = list(Path(f"./demo/data/panel_label/{type}").glob("*.jpg"))
    positive_num = 0
    total_num = len(image_paths)
    cfg = PanelLabelConfig()
    detector = OCRPipelineCrop(
        cfg.model_path,
        cfg.orient_model_path,
        confThreshold=cfg.confThreshold,
        nmsThreshold=cfg.nmsThreshold,
    )
    for image_path in image_paths:
        image_src = cv2.imread(str(image_path))
        rois = detector.infer(image_src)
        for i, roi in enumerate(rois):
            cv2.imwrite(f"{crop_dir}/{image_path.stem}_{i}.jpg", roi)
