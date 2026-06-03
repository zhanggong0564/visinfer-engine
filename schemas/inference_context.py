'''
@Description : 推理链路每请求上下文与检测器私有缩放元数据
'''

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class InferenceContext:
    """每请求推理上下文：承载原属于单例 self 的每请求状态，贯穿编排链路。"""
    # 输入
    image: np.ndarray
    h: int
    w: int
    product_type: str = ""
    rule: str = "all"
    is_registered: bool = False
    registered: Optional[np.ndarray] = None  # 注册/比对类场景的第二输入（注册参考图）
    # 场景私有的请求级参数透传袋（由 InputParamsBusiness.extra 原样带入，框架不解释）。
    extra: dict = field(default_factory=dict)
    # 阶段产物
    raw_result: Any = None        # detector.infer 的输出
    result: Any = None            # 业务后处理输出 (MoMResult / bool ...)
    # 钩子控制位
    skip_normalize: bool = False  # 注册流程等场景置 True 以跳过归一化


@dataclass
class PreprocMeta:
    """检测器私有的预处理元数据：在 preprocess → post_process 间局部传递，
    取代挂在 self 上的 r/dw/dh/image_src_shape/ori_img。"""
    r: float
    dw: float
    dh: float
    src_shape: Tuple[int, ...]
    ori_img: Optional[np.ndarray] = None
