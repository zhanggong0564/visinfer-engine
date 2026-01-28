'''
@Author       : gongzhang4
@Date         : 2026-01-28 07:06:44
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-28 07:47:44
@FilePath     : line_squeeze_routers.py
@Description  : 线 squeeze 检测接口
'''

from .base_router import BaseRouter
from schemas import LineSqueezeRequest
import numpy as np
from services.data_base import InputParamsBusiness


class LineSqueezeRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type):
        super().__init__(router_name, api_path, summary, description, detector_type)

    def request_schema(self, json_dict):
        return LineSqueezeRequest(**json_dict)

    def get_inputs(self, request_params: LineSqueezeRequest, image: np.ndarray):
        input = InputParamsBusiness(image=image, product_type=request_params.modelParams.product_model)
        return input


line_squeeze_router = LineSqueezeRouter(
    router_name="line_squeeze_router",
    api_path="/line_squeeze_recognition",
    summary="线序识别接口",
    description="根据输入的图像和产品类型，返回识别结果",
    detector_type="LineSqueeze",
)
