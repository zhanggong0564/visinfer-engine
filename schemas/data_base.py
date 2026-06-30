'''
@Author       : gongzhang4
@Date         : 2026-01-21 06:34:07
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-04-01 03:40:48
@FilePath     : data_base.py
@Description  :
'''

from dataclasses import dataclass, field
import os
import numpy as np
from enum import Enum
from typing import List, Optional
import cv2

# 可视化绘制常量（统一颜色/线宽，避免散落魔法值）
_DRAW_COLOR = (0, 255, 0)
_BLEND_ALPHA = 0.7  # 检测框图层与掩膜图层的混合权重


@dataclass
class DetectResult:
    boxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    mask_polygons: List[List[float]] = field(default_factory=list)
    ori_img: Optional[np.ndarray] = None

    def save_img(self, save_path: str) -> Optional[np.ndarray]:
        '''
        调试用可视化：把检测结果画到原图副本上并存盘，同时返回渲染后的图像，
        方便在调试器 / Notebook 里直接查看，无需再读回文件。

        - 无原图（ori_img 为 None）时直接返回 None；
        - scores / class_names 缺失也能正常画框（只是不画标签），不影响调试中间结果；
        - 自动创建 save_path 的父目录，写盘失败时抛 IOError 给出明确反馈。
        '''
        if self.ori_img is None:
            return None

        vis_img = self.ori_img.copy()
        # 中间结果可能只有框、没有 scores/class_names，用 None 兜底对齐长度
        scores = self.scores or [None] * len(self.boxes)
        names = self.class_names or [None] * len(self.boxes)

        if self.mask_polygons:
            # 有掩膜：画轴对齐框 + 标签，再叠加半透明掩膜图层
            for box, score, name in zip(self.boxes, scores, names):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), _DRAW_COLOR, 2)
                self._put_label(vis_img, name, score, (x1, y1 - 5))
            mask_img = self.ori_img.copy()
            for segment in self.mask_polygons:
                cv2.fillPoly(mask_img, np.int32([segment]), _DRAW_COLOR)
            vis_img = cv2.addWeighted(vis_img, _BLEND_ALPHA, mask_img, 1 - _BLEND_ALPHA, 0)
        else:
            # 无掩膜：按四边形顶点画闭合多边形 + 标签（reshape 兼容扁平/嵌套两种点格式）
            for box, score, name in zip(self.boxes, scores, names):
                pts = np.asarray(box, dtype=int).reshape(-1, 2)
                cv2.polylines(vis_img, [pts], True, _DRAW_COLOR, 4)
                self._put_label(vis_img, name, score, (int(pts[0][0]), int(pts[0][1]) - 5))

        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not cv2.imwrite(save_path, vis_img):
            raise IOError(f"save_img 写盘失败（路径非法或格式不支持）: {save_path}")
        return vis_img

    @staticmethod
    def _put_label(img: np.ndarray, name, score, org) -> None:
        '''把类别名/置信度画到 org 处；两者都缺省时不画。'''
        if name is None and score is None:
            return
        if name is not None and score is not None:
            text = f"{name}: {score:.2f}"
        elif name is not None:
            text = str(name)
        else:
            text = f"{score:.2f}"
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _DRAW_COLOR, 2)


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
        # 不就地 pop 入参：复制后再拆，避免破坏调用方持有的原 dict
        data = dict(data)
        detailList = [DetectionItem.from_dict(item) for item in data.pop("detailList")]
        return cls(detailList=detailList, **data)


@dataclass
class InputParamsBusiness:
    image: np.ndarray = field(default_factory=lambda: np.array([]))
    SN: str = ""
    product_type: str = ""
    is_registered: bool = False
    rule: str = "all"
    # 注册/比对类场景（如指示灯）的第二输入：注册参考图，默认空数组
    registered: np.ndarray = field(default_factory=lambda: np.array([]))
    # 场景私有的请求级参数透传袋（框架不解释内容，原样交给各场景业务层）。
    # 例：panel_label 用它携带 standard_result / guideline 等随请求下发的判定基准。
    extra: dict = field(default_factory=dict)
