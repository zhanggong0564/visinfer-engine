'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:48:54
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 06:24:21
@FilePath     : common.py
@Description  :
'''

from pydantic import BaseModel, Field
from typing import List, Literal


class GuideLineItem(BaseModel):
    """modelParams中guide_line的单个元素模型"""

    FileName: str = Field(..., description="参考线图片文件名")
    FilePath: str = Field(..., description="参考线图片访问路径")


class ExampleImageItem(BaseModel):
    """modelParams中example_images的单个元素模型"""

    FileName: str = Field(..., description="示例图片文件名")
    FilePath: str = Field(..., description="示例图片访问路径")


class DetailListItem(BaseModel):
    """result中detailList的单个元素模型(精准匹配示例）"""

    status: bool = Field(..., description="检测状态(false=正常/未检出异常,true=异常）")
    scene: str = Field(..., description="检测类别(screw/metal_piece/brass_plate等)")
    coordinate: List[float] = Field(..., description="检测轮廓点坐标顺序连接")
    accuracy: float = Field(..., description="检测置信度(0-1之间)")


class ResultResponse(BaseModel):
    """顶层返回的result对象模型"""

    detailList: List[DetailListItem] = Field(..., description="检测详情列表")
    status: Literal["true", "false"] = Field(..., description="整体检测状态（false=正常，true=异常）")
    error_msg: str = Field(..., description="错误信息（无错误则为空字符串）")
    message: str = Field(..., description="检测结果描述（如“检测成功”）")


class CommonResponse(BaseModel):
    """接口顶层返回模型（完全匹配示例）"""

    code: Literal[0, 1] = Field(..., description="0=识别失败，1=识别成功（整数类型）")
    message: str = Field(..., description="返回消息（如“成功”）")
    result: ResultResponse = Field(..., description="检测结果详情对象")
