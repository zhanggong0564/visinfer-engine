'''
@Author       : gongzhang4
@Date         : 2026-01-27 09:14:15
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 09:17:18
@FilePath     : indicator_routers.py
@Description  : 指示灯检测接口
'''

from .base_router import BaseRouter
from schemas import IndicatorRequest
import numpy as np
from services.data_base import InputParamsBusiness


class IndicatorRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type):
        super().__init__(router_name, api_path, summary, description, detector_type)

    def request_schema(self, json_dict):
        return IndicatorRequest(**json_dict)

    def get_inputs(self, request_params: IndicatorRequest, image: np.ndarray):
        main_type = request_params.type
        type = request_params.modelParams.type
        product_type = f"{main_type}-{type}"
        input = InputParamsBusiness(
            image=image, product_type=product_type, is_registered=request_params.modelParams.register
        )
        return input


indicator_router = IndicatorRouter(
    router_name="indicator_router",
    api_path="/indicator_light_detect",
    summary="指示灯检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    detector_type="indicator_light",
)
