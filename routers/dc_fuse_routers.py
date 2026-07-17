'''
@Description : 直流熔丝检测路由（服务内场景形态，被 router_registry 目录扫描自动发现）
'''

import numpy as np

from .base_router import BaseRouter
from schemas.dc_fuse_schemas import DCFuseRequest
from schemas.data_base import InputParamsBusiness
import services.dc_fuse  # noqa: F401  导入即触发 @scenario_registry.register("dc_fuse")


class DCFuseRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return DCFuseRequest(**json_dict)

    @staticmethod
    def _extract_product_type(request_params):
        # 本场景型号字段名为 product_model，重写基类默认的 product_type 提取，
        # 使数据回流按型号分目录而非落到 _unknown_model。
        model_params = getattr(request_params, "modelParams", None)
        return getattr(model_params, "product_model", None) if model_params else None

    def get_inputs(self, request_params: DCFuseRequest, image: np.ndarray):
        product_model = request_params.modelParams.product_model
        return InputParamsBusiness(image=image, product_type=product_model)


dc_fuse_router = DCFuseRouter(
    router_name="dc_fuse_router",
    api_path="/dcfuse_detect",
    summary="直流熔丝检测接口",
    description="根据输入的图像和产品型号，返回直流熔丝检测结果",
    detector_type="dc_fuse",
    tag="直流熔丝检测",
)
