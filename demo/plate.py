'''
@Author       : gongzhang4
@Date         : 2026-01-13 05:15:10
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-13 06:50:27
@FilePath     : plate.py
@Description  :
'''

from services import PlateScrewJudgeApi
import cv2

if __name__ == '__main__':
    model_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/weights/mobile_vision_plate_v2.onnx"
    image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/plate/IMG_7141.JPG"
    image = cv2.imread(image_path)
    confThreshold = 0.5
    infer = PlateScrewJudgeApi(model_path, confThreshold)
    print(infer.detect(image))
