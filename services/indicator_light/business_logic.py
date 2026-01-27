'''
@Author       : gongzhang4
@Date         : 2026-01-16 02:33:13
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 10:06:21
@FilePath     : business_logic.py
@Description  :
'''

import numpy as np
import json
from utils import vision_logger
import os
from ..api import detection_factory
from ..base.business_logic_base import BusinessLogicBase
from ..data_base import IndicatorLightEmbedding, MoMResult, DetectionItem
from .indicator_light_det import IndicatorLightDetRec

'''
1. 先检测出很多roi，对roi进行排序

'''


@detection_factory.register("indicator_light")
class IndicatorLightBusinessAPI(BusinessLogicBase):
    def __init__(self, settings):
        super().__init__(settings)
        self.standard_embeddings = {}
        self.is_cache = settings.indicator_light.IS_CACHE
        self.json_path = settings.indicator_light.JSON_PATH
        self.sim_thr = settings.indicator_light.SIM_THR
        if self.is_cache and os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.standard_embeddings = json.load(f)

    def _initialize_model(self, settings):
        try:
            self.detector = IndicatorLightDetRec(
                settings.indicator_light.ModelPath.det_model_path,
                settings.indicator_light.ModelPath.rec_model_path,
                settings.indicator_light.ConfThreshold.det,
            )
        except Exception as e:
            vision_logger.error(f"IndicatorLightBusinessAPI init error: {e}")
            raise e

    def registered_post_process(self, results: IndicatorLightEmbedding, type_s):
        try:
            self.standard_embeddings[type_s] = results.embeddings
            if self.is_cache:
                with open(self.json_path, 'w') as f:
                    json.dump(self.standard_embeddings, f)
        except Exception as e:
            return MoMResult(status=False, error_msg=str(e), message='失败')
        return MoMResult(
            detailList=[
                DetectionItem(coordinate=box, status=True, scene="", accuracy=score)
                for box, score in zip(results.boxes, results.scores)
            ],
            status=True,
            message="注册成功",
            error_msg="",
        )

    def business_logic_post_process(self, results: IndicatorLightEmbedding, type_s: str) -> MoMResult:
        try:
            standard_embeddings = self.standard_embeddings[type_s]
            if standard_embeddings is None:
                vision_logger.error(f"未找到类型为 {type_s} 的标准特征，请先注册")
            if len(standard_embeddings) != len(results.embeddings):
                vision_logger.warning(f"检测到的指示灯数量与注册的标准特征数量不匹配，可能导致比对结果异常")
                return MoMResult(
                    status=False,
                    error_msg=f"Number of ROIs does not match the registered standard image {len(standard_embeddings)}!={len(results.embeddings)}.",
                    message="失败",
                    detailList=[{"status": "false", "scene": "", "coordinate": [], "accuracy": 0.0}],
                )
            flag_status = True
            detect_results = MoMResult()
            for i, (std_embedding, embedding) in enumerate(zip(standard_embeddings, results.embeddings)):
                detectionitem = self.compare_embedding(std_embedding, embedding)
                detectionitem.coordinate = results.boxes[i][:4]
                detect_results.detailList.append(detectionitem)
                if detectionitem.status == False:
                    flag_status = False
            detect_results.status = flag_status
            detect_results.message = "success" if flag_status else "failed"
        except Exception as e:
            return MoMResult(status=False, error_msg=str(e), message='失败')
        return detect_results

    def compare_embedding(self, std_embeddings, embeddings):
        if isinstance(std_embeddings, list):
            std_embeddings = np.array(std_embeddings)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        distance = np.dot(std_embeddings, embeddings.T) / (np.linalg.norm(std_embeddings) * np.linalg.norm(embeddings))
        distance = (distance + 1) / 2
        if distance > self.sim_thr:

            return DetectionItem(
                status=True,
                scene="roi",
                accuracy=round(distance.item(), 3),
            )
        else:
            return DetectionItem(
                status=False,
                scene="roi",
                accuracy=round(distance.item(), 3),
            )
