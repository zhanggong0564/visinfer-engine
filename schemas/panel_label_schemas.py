'''
@Author       : gongzhang4
@Date         : 2026-03-27 12:16:00
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 12:16:02
@FilePath     : panel_label_schemas.py
@Description  :
'''

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal  # 新增Optional
from .common import GuideLineItem, ExampleImageItem


class ModelParams(BaseModel):
    """modelParams整体模型（guide_line/example_images设为可选）"""

    guide_line: Optional[List[GuideLineItem]] = Field(default_factory=list, description="参考线图片列表")
    example_images: Optional[List[ExampleImageItem]] = Field(default_factory=list, description="示例图片列表")
    product_type: str = Field(..., description="产品型号(例如:QF2)")


class PanelLabelRequest(BaseModel):
    """请求中json_data对应的结构化模型"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
