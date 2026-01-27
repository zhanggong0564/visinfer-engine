'''
@Author       : gongzhang4
@Date         : 2026-01-27 10:12:18
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 10:18:22
@FilePath     : lap_surf_routers.py
@Description  : 搭接面检测接口
'''

from .base_router import BaseRouter
from schemas import EmptyRequest
import numpy as np
from services.data_base import InputParamsBusiness


class LapSurfRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type):
        super().__init__(router_name, api_path, summary, description, detector_type)

    def request_schema(self, json_dict):
        return EmptyRequest(**json_dict)

    def get_inputs(self, request_params: EmptyRequest, image: np.ndarray):
        input = InputParamsBusiness(image=image)
        return input


lap_surf_router = LapSurfRouter(
    router_name="lap_surf_router",
    api_path="/lap_surf_detect",
    summary="搭接面检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    detector_type="lap_surf",
)
