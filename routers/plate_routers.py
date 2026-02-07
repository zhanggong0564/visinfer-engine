'''
@Author       : gongzhang4
@Date         : 2026-01-23 09:11:23
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-30 09:32:21
@FilePath     : plate_routers.py
@Description  : 直流熔丝检测口
'''

from .base_router import BaseRouter
from schemas import EmptyRequest
import numpy as np
from schemas.data_base import InputParamsBusiness


class PlateScrewRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type):
        super().__init__(router_name, api_path, summary, description, detector_type)

    def request_schema(self, json_dict):
        return EmptyRequest(**json_dict)

    def get_inputs(self, request_params: EmptyRequest, image: np.ndarray):
        input = InputParamsBusiness(image=image)
        return input


plate_screw_router = PlateScrewRouter(
    router_name="plate_router",
    api_path="/plate_screw_detect",
    summary="螺丝检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    detector_type="plate_screw",
)
