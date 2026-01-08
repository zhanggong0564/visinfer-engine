'''
@Author       : gongzhang4
@Date         : 2026-01-08 01:55:25
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 06:02:46
@FilePath     : lap_suf.py
@Description  :
'''

import sys

sys.path.append("..")

import cv2
import numpy as np
from config import settings
from utils import vision_logger
from services import LapSurfJudgeApi
import json


if __name__ == "__main__":
    image_path = r"/data/zhanggong/workspace/project/move_vsion/mobile_vision/test/images/lQDPJwfWMYLzRv_NC9DND8CwNWP5oG42qP0I3k_Zt4wLAA_4032_3024.jpg"
    image = cv2.imread(image_path)

    infer = LapSurfJudgeApi(settings.lap_surf.model_path, conf_threshold=settings.lap_surf.confThreshold)

    # bboxes, scores, labels, masks = infer(image)
    res = infer.detect(image)
    vision_logger.info(json.dumps(res, ensure_ascii=False, indent=4))
