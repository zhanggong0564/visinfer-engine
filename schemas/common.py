'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:48:54
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-03 03:40:29
@FilePath     : common.py
@Description  :
'''

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, List, Literal


class DetectionItemResponse(BaseModel):
    """检测详情项的对外 JSON 响应模型。"""

    model_config = ConfigDict(from_attributes=True)

    status: Literal["true", "false"] = Field(
        default="false",
        description="单项检测状态字符串：true=异常/命中，false=正常/未命中",
    )
    scene: str = Field(default="", description="检测类别或业务场景标识")
    coordinate: List[float] = Field(default_factory=list, description="检测框/轮廓坐标")
    accuracy: float = Field(default=0.0, description="检测置信度，范围 0-1")
    name: str = Field(default="", description="检测目标名称")
    color: str = Field(
        default="#FFFF00",
        description="前端绘制颜色；通常 true 为 #20ff4f，false 为 #FFFF00",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "false"}:
                return normalized
        return value


class GuideLineItem(BaseModel):
    """modelParams中guide_line的单个元素模型"""

    FileName: str = Field(..., description="参考线图片文件名")
    FilePath: str = Field(..., description="参考线图片访问路径")


class ExampleImageItem(BaseModel):
    """modelParams中example_images的单个元素模型"""

    FileName: str = Field(..., description="示例图片文件名")
    FilePath: str = Field(..., description="示例图片访问路径")


# class DetailListItem(BaseModel):
#     """result中detailList的单个元素模型(精准匹配示例）"""

#     status: bool = Field(..., description="检测状态(false=正常/未检出异常,true=异常）")
#     scene: str = Field(..., description="检测类别(screw/metal_piece/brass_plate等)")
#     coordinate: List[float] = Field(..., description="检测轮廓点坐标顺序连接")
#     accuracy: float = Field(..., description="检测置信度(0-1之间)")


class ResultResponse(BaseModel):
    """顶层返回的result对象模型"""

    detailList: List[DetectionItemResponse] = Field(..., description="检测详情列表")
    status: Literal["true", "false"] = Field(..., description="整体检测状态（false=正常，true=异常）")
    error_msg: str = Field(..., description="错误信息（无错误则为空字符串）")
    message: str = Field(..., description="检测结果描述（如“检测成功”）")
    vis_image: str = Field(
        default="",
        description="检测结果可视化图(JPEG base64，带 data:image/jpeg;base64, 前缀，可直接用于 <img src>)；缩图长边≤1280；绘制失败或无原图时为空串",
    )


class CommonResponse(BaseModel):
    """接口顶层返回模型（完全匹配示例）"""

    code: int = Field(..., description="错误码，1=成功，其它见 ErrorCode 定义")
    message: str = Field(..., description="返回消息（如“成功”）")
    result: ResultResponse = Field(..., description="检测结果详情对象")


class EmptyRequest(BaseModel):
    """空返回模型"""
