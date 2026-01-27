'''
@Author       : gongzhang4
@Date         : 2026-01-23 05:37:39
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 08:52:22
@FilePath     : business_logic_base.py
@Description  : 业务逻辑基类
'''

import numpy as np
from ..data_base import InputParamsBusiness, MoMResult, DetectResult, IndicatorLightEmbedding


class BusinessLogicBase:
    def __init__(self, settings):
        self.settings = settings
        self.detector = None
        self._initialize(settings)

    def _initialize(self, settings):
        self._initialize_model(settings)

    def _initialize_model(self, settings):
        raise NotImplementedError

    def detect(self, InputParamsBusiness: InputParamsBusiness) -> MoMResult:
        image = InputParamsBusiness.image
        h, w, _ = image.shape
        is_registered = InputParamsBusiness.is_registered
        product_type = InputParamsBusiness.product_type
        result = self.detector.infer(image)
        result = self.business_logic_post_process(result, product_type)
        result = self.result_post_process(result, w, h)

        return result

    def business_logic_post_process(self, result: DetectResult, product_type: str) -> MoMResult:
        raise NotImplementedError

    def registered_post_process(self, result: IndicatorLightEmbedding, product_type: str) -> bool:
        raise NotImplementedError

    def result_post_process(self, result: MoMResult) -> MoMResult:
        raise NotImplementedError
