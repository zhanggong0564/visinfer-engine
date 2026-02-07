'''
@Author       : gongzhang4
@Date         : 2026-01-23 05:37:39
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-07 08:04:21
@FilePath     : business_logic_base.py
@Description  : 业务逻辑基类
'''

import numpy as np
from schemas.data_base import InputParamsBusiness, MoMResult, DetectResult, IndicatorLightEmbedding


class BusinessLogicBase:
    def __init__(self, settings):
        self.settings = settings
        self.detector = None
        self._initialize(settings)

    def _initialize(self, settings):
        self._initialize_model(settings)

    def _initialize_model(self, settings):
        raise NotImplementedError

    def detect(self, InputParams: InputParamsBusiness) -> MoMResult:
        image = InputParams.image
        self.h, self.w, _ = image.shape
        is_registered = InputParams.is_registered
        product_type = InputParams.product_type
        result = self.detector.infer(image)
        if is_registered:
            return self.registered_post_process(result, product_type)
        result = self.business_logic_post_process(result, product_type)
        result = self.result_post_process(result)

        return result

    def business_logic_post_process(self, result: DetectResult, product_type: str) -> MoMResult:
        raise NotImplementedError

    def registered_post_process(self, result: IndicatorLightEmbedding, product_type: str) -> bool:
        raise NotImplementedError

    def result_post_process(self, result: MoMResult) -> MoMResult:
        """结果后处理"""
        detailList = result.detailList
        for item in detailList:
            coordinate = item.coordinate
            ltx, lty, rbx, rby = coordinate
            x1, y1 = ltx, lty
            x2, y2 = rbx, lty
            x3, y3 = rbx, rby
            x4, y4 = ltx, rby
            item.coordinate = [
                x1 / self.w,
                y1 / self.h,
                x2 / self.w,
                y2 / self.h,
                x3 / self.w,
                y3 / self.h,
                x4 / self.w,
                y4 / self.h,
            ]
        return result
