'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:39:58
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 08:39:43
@FilePath     : dc_fuse_schemas.py
@Description  :
'''

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal  # 新增Optional
from .common import GuideLineItem, ExampleImageItem, ResultResponse


class ModelParams(BaseModel):
    """modelParams整体模型（guide_line/example_images设为可选）"""

    guide_line: Optional[List[GuideLineItem]] = Field(default_factory=list, description="参考线图片列表")
    example_images: Optional[List[ExampleImageItem]] = Field(default_factory=list, description="示例图片列表")
    product_model: str = Field(..., description="产品型号(例如:六路无熔丝盒无磁环)")


class DCFuseRequest(BaseModel):
    """请求中json_data对应的结构化模型"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")


class DCFuseResponse(BaseModel):
    """接口顶层返回模型（完全匹配示例）"""

    code: Literal[0, 1] = Field(..., description="0=识别失败，1=识别成功（整数类型）")
    message: str = Field(..., description="返回消息（如“成功”）")
    result: ResultResponse = Field(..., description="检测结果详情对象")
