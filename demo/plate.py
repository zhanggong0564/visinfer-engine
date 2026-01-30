'''
@Author       : gongzhang4
@Date         : 2026-01-13 05:15:10
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-29 12:44:24
@FilePath     : plate.py
@Description  :
'''

import cv2
from services import detection_factory
from services.data_base import InputParamsBusiness

if __name__ == '__main__':
    image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/plate/IMG_7141.JPG"
    image = cv2.imread(image_path)
    # confThreshold = 0.5
    # model = plate_screw_detect(settings.plate_screw.model_path, settings.plate_screw.confThreshold)
    # print(model.infer(image))
    model = detection_factory.get_scenarios("plate_screw")
    inputs = InputParamsBusiness(image=image)
    print(model.detect(inputs))
    # infer = PlateScrewJudgeApi(model_path, confThreshold)
    # print(infer.detect(image))
