'''
@Description : 推理链路每请求上下文与检测器私有缩放元数据
'''

from dataclasses import dataclass
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
