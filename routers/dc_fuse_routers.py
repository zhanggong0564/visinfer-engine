'''
@Author       : gongzhang4
@Date         : 2026-01-23 09:11:23
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 06:32:33
@FilePath     : dc_fuse_routers.py
@Description  : 检测融合算法v2
'''

from .base_router import BaseRouter
from schemas import DCFuseRequest
import numpy as np
from services.data_base import InputParamsBusiness


class DCFuseRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type):
        super().__init__(router_name, api_path, summary, description, detector_type)

    def request_schema(self, json_dict):
        return DCFuseRequest(**json_dict)

    def get_inputs(self, request_params: DCFuseRequest, image: np.ndarray):
        input = InputParamsBusiness(image=image, product_type=request_params.modelParams.product_model)
        return input


dc_router = DCFuseRouter(
    router_name="dc_fuse_router",
    api_path="/dcfuse_detect",
    summary="直流熔丝检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    detector_type="dc_fuse",
)
