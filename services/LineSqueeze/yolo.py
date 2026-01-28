'''
@Author       : gongzhang4
@Date         : 2026-01-17 06:31:48
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-28 07:31:16
@FilePath     : yolo.py
@Description  :
'''

import numpy as np
from ..yolo import YoloOnnxInfer
from paddleocr import TextRecognition
from ..data_base import OCRResult, DetectResult


class LineSqueezeDetect(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, 2, confThreshold, nmsThreshold, task)
        self.id2name = {0: "fu_line", 1: "dc_line"}


class LineSqueezePipeline:
    def __init__(self, det_model_path, ocr_model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        self.detector = LineSqueezeDetect(det_model_path, confThreshold, nmsThreshold, task)
        self.ocr = TextRecognition(model_dir=ocr_model_path, model_name='en_PP-OCRv5_mobile_rec')

    def infer(self, image: np.ndarray) -> OCRResult:
        # DetectResult
        results = self.detector.infer(image)

        rois = [image[int(box[1]) + 10 : int(box[3]) - 10, int(box[0]) : int(box[2])] for box in results.boxes]

        res = [res['rec_text'] for res in self.ocr.predict(input=rois) if len(res['rec_text']) > 2]
        return OCRResult(text=res, boxes=results.boxes, class_ids=results.class_ids)

