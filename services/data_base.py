'''
@Author       : gongzhang4
@Date         : 2026-01-21 06:34:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-28 02:12:58
@FilePath     : data_base.py
@Description  :
'''

from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from typing import List


@dataclass
class DetectResult:
    boxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)


@dataclass
class IndicatorLightEmbedding:
    embeddings: List[List[float]] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)


@dataclass
class OCRResult:
    text: List[str] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)


class StatusCode(int, Enum):
    SUCCESS = 0
    FAIL = 1


@dataclass
class DetectionItem:
    status: bool = False
    scene: str = ""
    coordinate: List[float] = field(default_factory=list)
    accuracy: float = 0.0
    name: str = ""

    def to_dict(self):
        return {
            "status": 'true' if self.status else 'false',
            "scene": self.scene,
            "coordinate": self.coordinate,
            "accuracy": self.accuracy,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class MoMResult:
    detailList: List[DetectionItem] = field(default_factory=list)
    status: bool = False
    error_msg: str = ""
    message: str = ""

    def to_dict(self):
        return {
            "detailList": [item.to_dict() for item in self.detailList],
            "status": 'true' if self.status else 'false',
            "error_msg": self.error_msg,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict):
        detailList = [DetectionItem.from_dict(item) for item in data["detailList"]]
        return cls(detailList=detailList, **data)


@dataclass
class InputParamsBusiness:
    image: np.ndarray = field(default_factory=np.ndarray)
    SN: str = ""
    product_type: str = ""
    is_registered: bool = False
