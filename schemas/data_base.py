'''
@Author       : gongzhang4
@Date         : 2026-01-21 06:34:07
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-27 02:55:03
@FilePath     : data_base.py
@Description  :
'''

from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from typing import List
import cv2


@dataclass
class DetectResult:
    boxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    mask_polygons: List[List[float]] = field(default_factory=list)
    ori_img: np.ndarray = field(default_factory=lambda: None)

    def save_img(self, save_path: str):
        '''
        可视化图像，将检测结果绘制在图像上
        '''
        if self.ori_img is not None:
            # 绘制检测框
            for box, score, class_name in zip(self.boxes, self.scores, self.class_names):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(self.ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    self.ori_img,
                    f"{class_name}: {score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            # 绘制掩码
            if len(self.masks) > 0:
                # 初始化 new_mask，确保与 ori_img 的尺寸和通道一致
                new_mask = np.zeros_like(self.ori_img, dtype=np.uint8)

                for mask in self.masks:
                    # 将 mask 转换为二值图像并调整尺寸
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    resized_mask = cv2.resize(mask, (self.ori_img.shape[1], self.ori_img.shape[0]))

                    # 如果 ori_img 是多通道图像，则需要将单通道 mask 转换为多通道
                    if len(new_mask.shape) == 3:  # RGB or BGR image
                        resized_mask = cv2.merge([resized_mask, resized_mask, resized_mask])

                    # 累加 mask
                    new_mask = cv2.add(new_mask, resized_mask)

                # 将 ori_img 和 new_mask 进行加权合成
                self.ori_img = cv2.addWeighted(self.ori_img, 0.5, new_mask, 0.5, 0)

            # 保存结果图像
            cv2.imwrite(save_path, self.ori_img)


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
    scores: List[float] = field(default_factory=list)


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
    color: str = "#20ff4f"  # true #20ff4f false 颜色=#F74E5A

    def to_dict(self):
        return {
            "status": 'true' if self.status else 'false',
            "scene": self.scene,
            "coordinate": self.coordinate,
            "accuracy": self.accuracy,
            "name": self.name,
            "color": self.color if self.status else "#FFFF00",
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
