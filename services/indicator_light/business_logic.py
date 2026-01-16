'''
@Author       : gongzhang4
@Date         : 2026-01-16 02:33:13
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-16 07:00:03
@FilePath     : business_logic.py
@Description  :
'''

from .yolo import IndicatorLightDet, IndicatorLightRecognition
import numpy as np
import json
from utils import vision_logger
import os

'''
1. 先检测出很多roi，对roi进行排序

'''


class IndicatorLightBusinessAPI:
    def __init__(
        self,
        model_path,
        ConfThreshold,
        nmsThreshold=0.5,
        is_cache=True,
        json_path='standard_embeddings.json',
        sim_thr=0.7,
    ):
        self.det = IndicatorLightDet(model_path.det_model_path, ConfThreshold.det, nmsThreshold)
        self.rec = IndicatorLightRecognition(model_path.rec_model_path)
        self.standard_embeddings = {}
        self.is_cache = is_cache
        self.json_path = json_path
        self.sim_thr = sim_thr
        if self.is_cache and os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.standard_embeddings = json.load(f)

    def register(self, embedding, type_s):
        try:
            self.standard_embeddings[type_s] = embedding
            if self.is_cache:
                with open(self.json_path, 'w') as f:
                    json.dump(self.standard_embeddings, f)
        except Exception as e:
            return {"code": 0, "ERROR": str(e), "message": '失败'}

        return {
            "detailList": [{'coordinate': [], "status": False, "scene": "", "accuracy": 0.0}],
            "status": "true",
            "message": "注册成功",
            "error_msg": "",
        }

    def detect(self, img, type_s, is_register=False):
        results = {"error_msg": ""}
        detect_results = []
        self.h, self.w, _ = img.shape
        try:
            sorted_boxes = self.det.infer(img)
            embeddings = []
            for box in sorted_boxes:
                x_min, y_min, x_max, y_max, _ = box
                # roi = image[int(y_min - 10) : int(y_max + 10), int(x_min - 10) : int(x_max + 10)]
                roi = img[
                    max(int(y_min - 10), 0) : min(int(y_max + 10), self.h),
                    max(int(x_min - 10), 0) : min(int(x_max + 10), self.w),
                ]
                embedding = self.rec.infer(roi)
                embeddings.append(embedding.tolist())
            if is_register:
                return self.register(embeddings, type_s)
            standard_embeddings = self.standard_embeddings.get(type_s, None)
            if standard_embeddings is None:
                vision_logger.error(f"未找到类型为 {type_s} 的标准特征，请先注册")
            if len(standard_embeddings) != len(embeddings):
                vision_logger.warning(f"检测到的指示灯数量与注册的标准特征数量不匹配，可能导致比对结果异常")
                return [
                    {
                        "code": 0,
                        "error_msg": f"Number of ROIs does not match the registered standard image {len(standard_embeddings)}!={len(embeddings)}.",
                        "message": "失败",
                        'detailList': [{"status": "false", "scene": "", "coordinate": [], "accuracy": 0.0}],
                    }
                ]

            flag_stutas = True
            for i, (std_embedding, embedding) in enumerate(zip(standard_embeddings, embeddings)):
                status = self.compare_embedding(std_embedding, embedding)
                x1, y1, x2, y2 = sorted_boxes[i][:4]
                coordinate = [
                    x1 / self.w,
                    y1 / self.h,
                    x2 / self.w,
                    y1 / self.h,
                    x2 / self.w,
                    y2 / self.h,
                    x1 / self.w,
                    y2 / self.h,
                ]
                status.update({"coordinate": coordinate})
                detect_results.append(status)
                if status["status"] == False:
                    flag_stutas = False
            results["status"] = "true" if flag_stutas else "false"
            results["message"] = "success" if flag_stutas else "failed"
            results["detailList"] = detect_results
        except Exception as e:
            results['detailList'] = [{"status": "false", "scene": "", "coordinate": [], "accuracy": 0.0}]
            results['error_msg'] = str(e)
            results['status'] = "false"
            results['message'] = "检测失败"

        return results

    def compare_embedding(self, std_embeddings, embeddings):
        if isinstance(std_embeddings, list):
            std_embeddings = np.array(std_embeddings)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        distance = np.dot(std_embeddings, embeddings.T) / (np.linalg.norm(std_embeddings) * np.linalg.norm(embeddings))
        distance = (distance + 1) / 2
        if distance > self.sim_thr:
            return {"status": True, "accuracy": round(distance.item(), 3), "scene": "roi"}
        else:
            return {"status": False, "accuracy": distance.item(), "scene": "roi"}
