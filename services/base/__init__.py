'''
@Author       : gongzhang4
@Date         : 2026-01-27 03:07:47
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 03:08:53
@FilePath     : __init__.py
@Description  :
'''

# 公共值对象先于依赖 API Schema 的编排模块加载，供响应模型直接复用。
from .geometry import CoordinateSpace, Point, Polygon, Region, xyxy_region
from .ocr import OCRResult, OCRToken
from .business_logic_base import BusinessLogicBase
from .batch_business_logic_base import BatchBusinessLogicBase
from .vision_infer import BaseVisionInfer
from .classification_pipeline import (
    BaseClassificationPipeline,
    ClassificationResult,
)
from .ctc_recognition_pipeline import BaseCtcRecognitionPipeline, CtcRecognitionResult
from .detection import Detection, DetectionResult
from .detector import Detector
from .inspection import InspectionVerdict
from .settings import SceneSettings


__all__ = [
    "BaseVisionInfer",
    "BaseClassificationPipeline",
    "BaseCtcRecognitionPipeline",
    "BusinessLogicBase",
    "BatchBusinessLogicBase",
    "ClassificationResult",
    "CtcRecognitionResult",
    "CoordinateSpace",
    "Detection",
    "DetectionResult",
    "Detector",
    "InspectionVerdict",
    "OCRResult",
    "OCRToken",
    "Point",
    "Polygon",
    "Region",
    "SceneSettings",
    "xyxy_region",
]
