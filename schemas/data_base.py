'''
@Author       : gongzhang4
@Date         : 2026-01-21 06:34:07
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-04-01 03:40:48
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
        mask_img = self.ori_img.copy()
        vis_img = self.ori_img.copy()
        if self.ori_img is not None:
            if len(self.mask_polygons) > 0:
                # 绘制检测框
                for box, score, class_name in zip(self.boxes, self.scores, self.class_names):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis_img,
                        f"{class_name}: {score:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                for segment in self.mask_polygons:
                    cv2.fillPoly(mask_img, np.int32([segment]), (0, 255, 0))
                vis_img = cv2.addWeighted(vis_img, 0.7, mask_img, 0.3, 0)
            else:
                for box, score, class_name in zip(self.boxes, self.scores, self.class_names):
                    # p1, p2 = (int(box[3][0]), int(box[3][1])), (int(box[2][0]), int(box[2][1]))
                    vis_img = cv2.polylines(
                        vis_img,
                        [np.asarray(box, dtype=int)],
                        True,
                        (0, 255, 0),
                        4,
                    )
                    # cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.putText(
                    #     vis_img,
                    #     f"{class_name}: {score:.2f}",
                    #     (x1, y1 - 5),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     (0, 255, 0),
                    #     2,
                    # )
            # 保存结果图像
            cv2.imwrite(save_path, vis_img)


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


class MessageType(str, Enum):
    SUCCESS = "检测成功"
    FAIL = "检测失败"
    PRODUCT_TYPE_ERROR = "产品类型错误"


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
        detailList = [DetectionItem.from_dict(item) for item in data.pop("detailList")]
        return cls(detailList=detailList, **data)


@dataclass
class InputParamsBusiness:
    image: np.ndarray = field(default_factory=np.ndarray)
    SN: str = ""
    product_type: str = ""
    is_registered: bool = False
    rule: str = "all"
