'''
@Author       : gongzhang4
@Date         : 2026-01-16 03:30:14
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-16 05:55:06
@FilePath     : indicator_light.py
@Description  :
'''

from services import IndicatorLightBusinessAPI
import cv2
from config import settings
import json


if __name__ == '__main__':
    image_path_register = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/demo/1/A0SW0030/register/IMG_20251007_102820_595.jpg"
    image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/demo/1/A0SW0030/IMG_20251007_102900_336.jpg"
    image = cv2.imread(image_path_register)
    image_rec = cv2.imread(image_path)
    confThreshold = 0.5
    infer = IndicatorLightBusinessAPI(
        settings.indicator_light.ModelPath,
        settings.indicator_light.ConfThreshold,
        json_path=settings.indicator_light.JSON_PATH,
        sim_thr=settings.indicator_light.SIM_THR,
    )
    # print(infer.detect(image, type_s='A0SW0030', is_register=True))
    json_str = infer.detect(image_rec, type_s='A0SW0030')
    print(json.dumps(json_str, ensure_ascii=False, indent=2))
