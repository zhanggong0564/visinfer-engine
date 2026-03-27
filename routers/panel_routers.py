'''
@Author       : gongzhang4
@Date         : 2026-03-27 11:22:29
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 11:47:42
@FilePath     : panel_routers.py
@Description  : 线标检测接口
'''

from .base_router import BaseRouter
from schemas import EmptyRequest
import numpy as np
from schemas.data_base import InputParamsBusiness


class PanelLabelRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type):
        super().__init__(router_name, api_path, summary, description, detector_type)

    def request_schema(self, json_dict):
        return EmptyRequest(**json_dict)

    def get_inputs(self, request_params: EmptyRequest, image: np.ndarray):
        input = InputParamsBusiness(image=image)
        return input


panel_label_router = PanelLabelRouter(
    router_name="panel_router",
    api_path="/panel_label_detect",
    summary="线标检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    detector_type="panel_label",
)
