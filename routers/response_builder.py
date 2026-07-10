"""推理结果清洗与公共响应构造。"""

import time
from typing import Any, Callable, Optional

import numpy as np

from schemas import CommonResponse, ERROR_CODE_MESSAGES, ErrorCode
from services.utils.visualize import render_detection_overlay
from utils.async_utils import run_sync


StageRecorder = Callable[[str, float], None]


class ResponseBuilder:
    def __init__(self, vis_enabled: bool, max_side: int, jpeg_quality: int) -> None:
        self.vis_enabled = vis_enabled
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def sanitize_detail_list_names(result_dict: Any) -> None:
        """将扁平或嵌套结果中的 detailList.name 规范为字符串。"""
        if not isinstance(result_dict, dict):
            return
        nested = result_dict.get("result")
        result_data = nested if isinstance(nested, dict) else result_dict
        detail_list = result_data.get("detailList", [])
        if not isinstance(detail_list, list):
            return
        for item in detail_list:
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = item["name"]
            item["name"] = "" if name is None else str(name)

    async def build(
        self,
        image: np.ndarray,
        result_dict: dict,
        inputs: Any,
        stage_recorder: Optional[StageRecorder] = None,
    ) -> CommonResponse:
        self.sanitize_detail_list_names(result_dict)
        response_result = result_dict
        if self.vis_enabled:
            extra = getattr(inputs, "extra", None)
            guideline = extra.get("guideline") if isinstance(extra, dict) else None
            guides = [tuple(guideline)] if guideline else None
            started = time.perf_counter()
            try:
                vis_b64 = await run_sync(
                    render_detection_overlay,
                    image,
                    result_dict.get("detailList", []),
                    guides=guides,
                    max_side=self.max_side,
                    jpeg_quality=self.jpeg_quality,
                )
            finally:
                if stage_recorder:
                    stage_recorder(
                        "vis_render", (time.perf_counter() - started) * 1000
                    )
            response_result = {**result_dict, "vis_image": vis_b64}
        elif stage_recorder:
            stage_recorder("vis_render", 0.0)
        return CommonResponse(
            code=int(ErrorCode.SUCCESS),
            message=ERROR_CODE_MESSAGES[ErrorCode.SUCCESS],
            result=response_result,
        )
