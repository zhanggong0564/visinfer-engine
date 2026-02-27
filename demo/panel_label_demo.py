'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:42:41
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-27 02:53:16
@FilePath     : panel_label_demo.py
@Description  :
'''

import sys

sys.path.append("..")
import cv2
from utils import vision_logger
from services import rotate_points
import numpy as np
from services.panel_label import PanelLabelDetect


if __name__ == '__main__':
    image_path = (
        "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/panel_label/IMG_20260127_150225_262.jpg"
    )
    image_src = cv2.imread(image_path)
    model_path = "./weights/best_v1.onnx"
    confThreshold = 0.6

    detector = PanelLabelDetect(model_path, confThreshold)
    results = detector.infer(image_src)
    results.save_img('res.jpg')
