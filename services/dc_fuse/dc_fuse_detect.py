'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:48:03
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 02:20:57
@FilePath     : dc_fuse_detect.py
@Description  :
'''

from ..yolo import YoloOnnxInfer


class DCFuseDetector(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, confThreshold, nmsThreshold)
