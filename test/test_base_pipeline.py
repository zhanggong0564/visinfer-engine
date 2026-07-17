"""推理编排层（Detector 协议 + 模板方法）单元测试"""
import inspect

from routers.response_builder import ResponseBuilder
import numpy as np
import pytest
from services.base.detector import Detector
from services.inference import TensorInfo
from services.base.vision_infer import BaseVisionInfer
from schemas.exceptions import ModelInferenceError


EXPECTED_RESULT = object()


class _FakeRunner:
    input_infos = (TensorInfo("images", (1, 3, 32, 32), "tensor(float)"),)
    output_infos = (TensorInfo("output0", (1, 6, 1), "tensor(float)"),)
    providers = ("FakeExecutionProvider",)

    def __init__(self):
        self.inputs = []

    def __bool__(self):
        return False

    def run(self, inputs):
        self.inputs.append(inputs)
        return [np.zeros((1, 6, 1), dtype=np.float32)]

    def close(self):
        self.closed = True


class _StubVisionInfer(BaseVisionInfer):
    def preprocess(self, image):
        meta = type("Meta", (), {})()
        return np.zeros((1, 3, 32, 32), dtype=np.float32), meta

    def post_process(self, outputs, meta):
        return EXPECTED_RESULT


def test_base_vision_infer_uses_injected_runner():
    runner = _FakeRunner()
    model = _StubVisionInfer(runner=runner)

    result = model.infer(np.zeros((32, 32, 3), dtype=np.uint8))

    assert set(runner.inputs[0]) == {"images"}
    assert model.input_names == ["images"]
    assert model.output_names == ["output0"]
    assert model.providers == list(runner.providers)
    assert result is EXPECTED_RESULT


def test_base_vision_infer_preserves_model_inference_error():
    expected = ModelInferenceError("runner failed")
    runner = _FakeRunner()
    runner.run = lambda inputs: (_ for _ in ()).throw(expected)
    model = _StubVisionInfer(runner=runner)

    with pytest.raises(ModelInferenceError) as caught:
        model.infer(np.zeros((32, 32, 3), dtype=np.uint8))

    assert caught.value is expected


def test_base_vision_infer_wraps_unexpected_error():
    runner = _FakeRunner()
    runner.run = lambda inputs: (_ for _ in ()).throw(RuntimeError("backend failed"))
    model = _StubVisionInfer(runner=runner)

    with pytest.raises(ModelInferenceError) as caught:
        model.infer(np.zeros((32, 32, 3), dtype=np.uint8))

    assert caught.value.context["original_error"] == "backend failed"


def test_base_vision_infer_closes_injected_runner_idempotently():
    runner = _FakeRunner()
    model = _StubVisionInfer(runner=runner)

    model.close()
    model.close()

    assert runner.closed is True


class TestDetectorProtocol:
    def test_runtime_checkable_accepts_infer_and_close_object(self):
        class Dummy:
            def infer(self, image):
                return "ok"

            def close(self):
                pass

        assert isinstance(Dummy(), Detector)

    def test_runtime_checkable_rejects_object_without_close(self):
        class InferOnly:
            def infer(self, image):
                return "ok"

        assert not isinstance(InferOnly(), Detector)

    def test_runtime_checkable_rejects_non_infer_object(self):
        class NoInfer:
            pass
        assert not isinstance(NoInfer(), Detector)


def test_model_constructors_do_not_accept_model_paths():
    from services.rfdetr import RFDetrInfer
    from services.yolo import YoloInfer

    assert "model_path" not in inspect.signature(BaseVisionInfer).parameters
    assert "model_path" not in inspect.signature(YoloInfer).parameters
    assert "model_path" not in inspect.signature(RFDetrInfer).parameters


import threading
import time
import numpy as np
from unittest.mock import MagicMock
from schemas import CommonResponse, ErrorCode, ERROR_CODE_MESSAGES
from schemas.data_base import MoMResult, DetectionItem, InputParamsBusiness
from routers.base_router import BaseRouter
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
    def test_detect_rejects_uninitialized_detector(self):
        logic = _OrderLogic(MagicMock())
        logic.detector = None
        params = InputParamsBusiness(image=np.zeros((10, 10, 3), dtype=np.uint8))

        with pytest.raises(RuntimeError, match="not initialized"):
            logic.detect(params)

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


class TestResponseDetailNameSanitize:
    def test_none_detail_name_is_converted_before_common_response_validation(self):
        response_data = {
            "detailList": [
                {"status": "false", "scene": "line", "coordinate": [], "accuracy": 0.9, "name": None},
            ],
            "status": "false",
            "error_msg": "",
            "message": "mismatch",
        }

        ResponseBuilder.sanitize_detail_list_names(response_data)
        response = CommonResponse(
            code=int(ErrorCode.SUCCESS),
            message=ERROR_CODE_MESSAGES[ErrorCode.SUCCESS],
            result=response_data,
        )

        assert response_data["detailList"][0]["name"] == ""
        assert response.model_dump()["result"]["detailList"][0]["name"] == ""
