"""推理编排层（Detector 协议 + 模板方法）单元测试"""
import numpy as np
from services.base.detector import Detector


class TestDetectorProtocol:
    def test_runtime_checkable_accepts_infer_object(self):
        class Dummy:
            def infer(self, image):
                return "ok"
        assert isinstance(Dummy(), Detector)

    def test_runtime_checkable_rejects_non_infer_object(self):
        class NoInfer:
            pass
        assert not isinstance(NoInfer(), Detector)


import threading
import time
import numpy as np
from unittest.mock import MagicMock
from schemas.data_base import MoMResult, DetectionItem, InputParamsBusiness
from services.base.business_logic_base import BusinessLogicBase


class _OrderLogic(BusinessLogicBase):
    def _initialize_model(self, settings):
        self.detector = MagicMock()
        self.detector.infer.return_value = MoMResult()
        self.calls = []

    def preprocess_hook(self, ctx):
        self.calls.append("pre")

    def business_post_process(self, ctx):
        self.calls.append("business")
        ctx.result = MoMResult()

    def normalize_hook(self, ctx):
        self.calls.append("normalize")

    def finalize_hook(self, ctx):
        self.calls.append("finalize")


class TestTemplateMethodOrder:
    def test_hook_call_order(self):
        logic = _OrderLogic(MagicMock())
        params = InputParamsBusiness(image=np.zeros((10, 10, 3), dtype=np.uint8))
        logic.detect(params)
        assert logic.calls == ["pre", "business", "normalize", "finalize"]

    def test_normalize_skipped_when_flag_set(self):
        class _Skip(_OrderLogic):
            def business_post_process(self, ctx):
                self.calls.append("business")
                ctx.result = MoMResult()
                ctx.skip_normalize = True
        logic = _Skip(MagicMock())
        params = InputParamsBusiness(image=np.zeros((10, 10, 3), dtype=np.uint8))
        logic.detect(params)
        assert "normalize" not in logic.calls

    def test_normalize_skipped_when_class_flag_false(self):
        class _NoNorm(_OrderLogic):
            NORMALIZE = False
        logic = _NoNorm(MagicMock())
        params = InputParamsBusiness(image=np.zeros((10, 10, 3), dtype=np.uint8))
        logic.detect(params)
        assert "normalize" not in logic.calls


class _ConcurrentLogic(BusinessLogicBase):
    """检测器按输入图像宽度回填坐标；若单例串台则结果会错配。"""

    def _initialize_model(self, settings):
        detector = MagicMock()

        def slow_infer(image):
            time.sleep(0.02)  # 放大竞态窗口
            return image.shape[1]  # 用宽度当"原始结果"
        detector.infer.side_effect = slow_infer
        self.detector = detector

    def business_post_process(self, ctx):
        mom = MoMResult()
        # 期望：业务结果里的宽度 == 本请求 ctx.w == raw_result
        mom.detailList = [DetectionItem(name=str(ctx.raw_result), coordinate=[0, 0, ctx.w, ctx.h])]
        mom.status = ctx.raw_result == ctx.w
        ctx.result = mom


class TestStatelessConcurrency:
    def test_no_cross_talk_under_threads(self):
        logic = _ConcurrentLogic(MagicMock())
        results = {}

        def run(width):
            params = InputParamsBusiness(image=np.zeros((100, width, 3), dtype=np.uint8))
            res = logic.detect(params)
            results[width] = res.status

        widths = [50, 120, 200, 333, 480, 640]
        threads = [threading.Thread(target=run, args=(w,)) for w in widths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 每个请求的业务结果都与自己的宽度一致 → 无串台
        assert all(results[w] is True for w in widths)
