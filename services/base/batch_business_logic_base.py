"""Business-logic template for multi-image scenarios."""

from typing import Any

import numpy as np

from utils import vision_logger
from utils.timing import StageTimer

from .business_logic_base import BusinessLogicBase


class BatchBusinessLogicBase(BusinessLogicBase):
    """Template method for one inference operation over multiple images."""

    NORMALIZE = False

    def inspect(
        self,
        images: list[tuple[str, np.ndarray]],
        request_params: Any,
    ) -> Any:
        timer = StageTimer()
        try:
            with timer.stage("validate_batch"):
                self.validate_batch(images, request_params)
            with timer.stage("inspect_batch"):
                return self.inspect_batch(images, request_params)
        finally:
            vision_logger.info("批量业务阶段耗时 {}", timer.summary(), event="inference.batch_timings")

    def validate_batch(
        self,
        images: list[tuple[str, np.ndarray]],
        request_params: Any,
    ) -> None:
        """Validate cross-image constraints before inference."""

    def inspect_batch(
        self,
        images: list[tuple[str, np.ndarray]],
        request_params: Any,
    ) -> Any:
        raise NotImplementedError("子类必须实现 inspect_batch 方法")

    def detect(self, params) -> Any:
        raise RuntimeError("批量场景应调用 inspect，不支持单图片 detect")

    def business_post_process(self, ctx) -> None:
        raise RuntimeError("批量场景不使用单图片 business_post_process")
