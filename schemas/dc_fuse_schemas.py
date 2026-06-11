'''
@Description : 直流熔丝检测请求 schema（服务内场景形态）
'''

from pydantic import BaseModel, Field
from typing import List, Optional

from schemas.common import GuideLineItem, ExampleImageItem


class ModelParams(BaseModel):
    """modelParams 整体模型（guide_line/example_images 设为可选）。"""

    guide_line: Optional[List[GuideLineItem]] = Field(default_factory=list, description="参考线图片列表")
    example_images: Optional[List[ExampleImageItem]] = Field(default_factory=list, description="示例图片列表")
    product_model: str = Field(..., description="产品型号(例如:六路无熔丝盒无磁环)")


class AICameraModels(BaseModel):
    """AICameraModel 模型。"""

    Id: str = Field(..., description="AICamera模型ID")
    SN: str = Field(..., description="AICamera模型SN")
    ProductName: str = Field(..., description="AICamera模型产品名称")
    Version: int = Field(..., description="AICamera模型版本")
    AIProductTypeName: str = Field(..., description="AICamera模型产品类型")
    AIProductTypeValue: str = Field(..., description="AICamera模型产品类型值")
    ModelFile: str = Field(..., description="AICamera模型文件")
    Remark: Optional[str] = Field(default=None, description="AICamera模型备注")
    CreateBy: Optional[str] = Field(default=None, description="创建人")
    CreateTime: Optional[str] = Field(default=None, description="创建时间")
    UpdateBy: Optional[str] = Field(default=None, description="更新人")
    UpdateTime: Optional[str] = Field(default=None, description="更新时间")
    AIParameterName: Optional[str] = Field(default=None, description="AICamera模型参数名称")
    AIParameterValue: Optional[str] = Field(default=None, description="AICamera模型参数值")
    DictionaryCode: Optional[str] = Field(default=None, description="字典编码")


class DCFuseRequest(BaseModel):
    """请求中 json_data 对应的结构化模型。"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
    AICameraModel: Optional[List[AICameraModels]] = Field(..., description="AICamera模型列表")
