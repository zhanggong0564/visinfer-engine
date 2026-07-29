'''
@Description : 直流熔丝检测请求 schema（服务内场景形态）
'''

from pydantic import BaseModel, Field
from typing import List, Optional

from schemas.common import AICameraModel, VisualReferenceParams


class ModelParams(VisualReferenceParams):
    """modelParams 整体模型（guide_line/example_images 设为可选）。"""

    product_model: str = Field(..., description="产品型号(例如:六路无熔丝盒无磁环)")


AICameraModels = AICameraModel


class DCFuseRequest(BaseModel):
    """请求中 json_data 对应的结构化模型。"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
    AICameraModel: Optional[List[AICameraModels]] = Field(..., description="AICamera模型列表")
