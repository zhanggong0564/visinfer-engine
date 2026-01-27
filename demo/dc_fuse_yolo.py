'''
@Author       : gongzhang4
@Date         : 2026-01-23 05:20:26
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 06:57:04
@FilePath     : dc_fuse_yolo.py
@Description  : 检测模型
'''

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from services.dc_fuse import DCFuseDetector
import cv2


if __name__ == '__main__':
    model_path = "./weights/dc_fuse_v5.onnx"
    confThreshold = 0.6

    detector = DCFuseDetector(model_path, confThreshold)
    image_path = "demo/data/dc_fuse/lQDPJwNIOLET3G_NC7jND6CwteZlxtGgiyAJOLfvGUpLAA_4000_3000.jpg"
    image = cv2.imread(image_path)
    import time

    start = time.time()
    for i in range(100):
        results = detector.infer(image)

    print(results)
    end = time.time()
    print(f"推理耗时：{(end - start) / 100}秒")
