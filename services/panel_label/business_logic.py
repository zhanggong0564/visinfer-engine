'''
@Author       : gongzhang4
@Date         : 2026-03-02 03:48:53
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-02 11:55:29
@FilePath     : business_logic.py
@Description  :
'''

from .panel_label_detect import OCRPipeline, PanellabelItem
from schemas import MoMResult, DetectResult, DetectionItem
from ..api import detection_factory
from ..base import BusinessLogicBase
from utils import vision_logger
from .product_type import PRODUCT_TYPE
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from dataclasses import field


class ErrorType(str, Enum):
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    OK = "ok"


@dataclass
class PanelInfo:
    result: bool = False
    product_type: str = ""
    # 标准ocr结果
    standard_result: list[str] = field(default_factory=list)
    # 观察到的ocr结果
    observed_result: list[str] = field(default_factory=list)
    # 观察到的ocr结果的点坐标
    observed_result_points: list[list[float]] = field(default_factory=list)
    message: str = ErrorType.UNKNOWN.value
    error_indexs: list[int] = field(default_factory=list)
    class_id: List[int] = field(default_factory=list)


@detection_factory.register("panel_label")
class PanelLabelJudgeApi(BusinessLogicBase):

    def __init__(self, settings):
        super().__init__(settings)

    def _initialize_model(self, settings):
        try:
            self.detector = OCRPipeline(
                settings.model_path,
                settings.orient_model_path,
                settings.confThreshold,
                settings.nmsThreshold,
            )
        except Exception as e:
            vision_logger.error(f"initialize model failed, error: {e}")
            raise e

    def business_logic_post_process(self, results: PanellabelItem, product_type: str):
        panel_info = self.analyze(results, product_type)
        mom_result = MoMResult()
        mom_result.status = panel_info.result
        mom_result.message = panel_info.message
        data_list = []
        for i, observed_item in enumerate(panel_info.observed_result):
            status = True
            if i in panel_info.error_indexs:
                status = False
            data_list.append(
                DetectionItem(
                    status=status,
                    scene=panel_info.class_id[i],
                    coordinate=panel_info.observed_result_points[i],
                    accuracy=results.confidence[i],
                    name=observed_item,
                )
            )
        mom_result.detailList = data_list
        return mom_result

    def analyze(self, observed_result: PanellabelItem, product_type: str) -> PanelInfo:
        standard_result = PRODUCT_TYPE[product_type]
        panel_info = PanelInfo(
            standard_result=standard_result,
            observed_result=observed_result.texts,
            observed_result_points=observed_result.Points,
            class_id=observed_result.class_id,
        )
        panel_info.result = True
        panel_info.message = ErrorType.OK.value
        if len(observed_result.texts) != len(standard_result):
            panel_info.message = ErrorType.MISSING.value
            panel_info.result = False
            return panel_info
        for i, item in enumerate(observed_result.texts):
            front3 = item[:3]
            # 取后3位，并转小写字符
            tail3 = item[-2:].lower()
            if front3 != standard_result[i][:3] or tail3 != standard_result[i][-2:].lower():
                panel_info.message = ErrorType.MISMATCH.value
                panel_info.result = False
                panel_info.error_indexs.append(i)
        return panel_info
