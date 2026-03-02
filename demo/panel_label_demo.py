'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:42:41
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-02 08:05:24
@FilePath     : panel_label_demo.py
@Description  :
'''

import sys

sys.path.append("..")
import cv2
from utils import vision_logger
from services import rotate_points
import numpy as np
from services.panel_label import OCRPipeline, PanelLabelJudgeApi
from config import settings
from schemas import InputParamsBusiness
import json

if __name__ == '__main__':
    image_path = "./demo/data/panel_label/IMG_20260127_150225_262.jpg"
    image_src = cv2.imread(image_path)
    # model_path = "./weights/panel_label/best_v1.onnx"
    # orient_model_path = "./weights/panel_label/PP-LCNet_x1_0_textline_ori"
    # confThreshold = 0.4

    # detector = OCRPipeline(model_path, orient_model_path, confThreshold)
    # results = detector.infer(image_src)
    # results.save_img(image_src, "./demo/data/panel_label_result.jpg")

    # for result in results:
    #     # result.print()
    #     print(f"rec_texts: {result['rec_texts']}->orientation: {result['textline_orientation_angles']}")
    input_params = InputParamsBusiness(
        image=image_src,
        product_type="QF2",
    )

    detector = PanelLabelJudgeApi(settings.panel_label)
    results = detector.detect(input_params)
    json_str = json.dumps(results.to_dict(), indent=4)

    print(json_str)
