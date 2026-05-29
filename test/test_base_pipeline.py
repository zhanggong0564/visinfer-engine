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
