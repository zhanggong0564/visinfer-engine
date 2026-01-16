'''
@Author       : gongzhang4
@Date         : 2026-01-16 05:30:51
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-16 06:32:11
@FilePath     : indicator_schemas.py
@Description  :
'''

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal  # 新增Optional
from .common import GuideLineItem, ExampleImageItem, ResultResponse


class ModelParams(BaseModel):
    """modelParams整体模型（guide_line/example_images设为可选）"""

    guide_line: Optional[List[GuideLineItem]] = Field(default_factory=list, description="参考线图片列表")
    example_images: Optional[List[ExampleImageItem]] = Field(default_factory=list, description="示例图片列表")
    type: int = Field(..., description="产品型号(例如:六路无熔丝盒无磁环)")
    register: bool = Field(..., description="是否为注册模式")


class IndicatorRequest(BaseModel):
    """请求中json_data对应的结构化模型"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
