'''
@Author       : gongzhang4
@Date         : 2026-01-16 03:30:14
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 08:51:13
@FilePath     : indicator_light.py
@Description  :
'''

from services import detection_factory
from services.data_base import InputParamsBusiness
import cv2
from config import settings
import json


if __name__ == '__main__':
    image_path_register = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/demo/1/A0SW0030/register/IMG_20251007_102820_595.jpg"
    type_name = "A0SW0030"
    image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/demo/1/A0SW0030/IMG_20251007_102900_336.jpg"
    image = cv2.imread(image_path_register)
    image_rec = cv2.imread(image_path)
    confThreshold = 0.5
    scenarios = detection_factory.get_scenarios("indicator_light")
    # infer = IndicatorLightBusinessAPI(
    #     settings.indicator_light.ModelPath,
    #     settings.indicator_light.ConfThreshold,
    #     json_path=settings.indicator_light.JSON_PATH,
    #     sim_thr=settings.indicator_light.SIM_THR,
    # )
    # print(infer.detect(image, type_s='A0SW0030', is_register=True))
    input = InputParamsBusiness(image=image, product_type=type_name, is_registered=True)
    json_str = scenarios.detect(input)
    print(json.dumps(json_str.to_dict(), ensure_ascii=False, indent=2))
    input = InputParamsBusiness(image=image_rec, product_type=type_name, is_registered=False)
    json_str = scenarios.detect(input)
    print(json.dumps(json_str.to_dict(), ensure_ascii=False, indent=2))
