'''
@Description : 业务逻辑基类（模板方法 + 可插拔钩子）
'''

from schemas.data_base import InputParamsBusiness
from schemas.inference_context import InferenceContext
from utils import vision_logger
from utils.timing import StageTimer

from .detector import Detector


class BusinessLogicBase:
    """推理编排模板方法基类。

    detect() 固定调用顺序，子类通过重写钩子定制行为：
        build_context → preprocess_hook → detector.infer
        → business_post_process → (should_normalize ? normalize_hook) → finalize_hook
    检测器与本类均无每请求状态，每请求态全部装在 InferenceContext 上，故单例可并发。
    """

    NORMALIZE: bool = True  # 类级开关：场景可置 False 关闭默认坐标归一化

    def __init__(self, settings):
        self.settings = settings
        self.detector: Detector | None = None
        self._initialize(settings)

    def _initialize(self, settings):
        self._initialize_model(settings)

    def _initialize_model(self, settings):
        raise NotImplementedError

    def detect(self, params: InputParamsBusiness):
        timer = StageTimer()
        try:
            detector = self.detector
            if detector is None:
                raise RuntimeError("scenario detector is not initialized")
            with timer.stage("build_context"):
                ctx = self.build_context(params)
            with timer.stage("preprocess_hook"):
                self.preprocess_hook(ctx)
            with timer.stage("detector_infer"):
                ctx.raw_result = detector.infer(ctx.image)
            with timer.stage("business_post_process"):
                self.business_post_process(ctx)
            if self.should_normalize(ctx):
                with timer.stage("normalize_hook"):
                    self.normalize_hook(ctx)
            else:
                timer.record("normalize_hook", 0.0)
            with timer.stage("finalize_hook"):
                self.finalize_hook(ctx)
            return ctx.result
        finally:
            vision_logger.info("业务检测阶段耗时 {}", timer.summary())

    def build_context(self, params: InputParamsBusiness) -> InferenceContext:
        h, w = params.image.shape[:2]
        return InferenceContext(
            image=params.image,
            h=h,
            w=w,
            product_type=params.product_type,
            rule=params.rule,
            is_registered=params.is_registered,
            registered=params.registered,
            extra=params.extra,
        )

    def preprocess_hook(self, ctx: InferenceContext) -> None:
        """图像级预处理钩子，默认 no-op。"""
        pass

    def business_post_process(self, ctx: InferenceContext) -> None:
        """场景业务后处理：读 ctx.raw_result，写 ctx.result。子类必须实现。"""
        raise NotImplementedError

    def should_normalize(self, ctx: InferenceContext) -> bool:
        return self.NORMALIZE and not ctx.skip_normalize

    def normalize_hook(self, ctx: InferenceContext) -> None:
        """默认坐标归一化：按原图宽高把 4/8 值坐标统一成归一化的 8 值多边形。"""
        result = ctx.result
        for item in result.detailList:
            coordinate = item.coordinate
            if len(coordinate) == 4:
                ltx, lty, rbx, rby = coordinate
                x1, y1 = ltx, lty
                x2, y2 = rbx, lty
                x3, y3 = rbx, rby
                x4, y4 = ltx, rby
            elif len(coordinate) == 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
            else:
                # 空坐标（如缺线占位项）或非 4/8 值，无从归一化，原样保留
                continue
            item.coordinate = [
                x1 / ctx.w, y1 / ctx.h,
                x2 / ctx.w, y2 / ctx.h,
                x3 / ctx.w, y3 / ctx.h,
                x4 / ctx.w, y4 / ctx.h,
            ]

    def finalize_hook(self, ctx: InferenceContext) -> None:
        """结果收尾钩子，默认 no-op。"""
        pass

    def close(self) -> None:
        """Release the scenario pipeline if it was initialized."""
        detector = self.detector
        if detector is not None:
            detector.close()
            self.detector = None
