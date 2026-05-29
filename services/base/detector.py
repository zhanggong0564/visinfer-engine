'''
@Description : 检测器统一契约
'''

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Detector(Protocol):
    """所有检测器的统一契约：吃一张图，吐一个结果对象。

    结果类型由各场景自定义（DetectResult / PanellabelItem 等），业务层自行解释。
    用 Protocol（结构化子类型），使组合型检测器（如 OCRPipeline）无需强制继承即可满足契约。
    """

    def infer(self, image: np.ndarray) -> Any: ...
