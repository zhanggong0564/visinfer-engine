from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import onnxruntime as ort
import pytest

from schemas.exceptions import ModelInferenceError
from services.base.inference_runner import InferenceRunner, OnnxRuntimeRunner, TensorInfo


def fake_session(input_shape=(1, 3, 8, 8)):
    session = MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="images", shape=list(input_shape), type="tensor(float)")
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name="output", shape=[1, 6], type="tensor(float)")
    ]
    session.get_providers.return_value = ["CPUExecutionProvider"]
    return session


def make_runner(session, warmup=False):
    with patch(
        "services.base.inference_runner.ort.InferenceSession", return_value=session
    ):
        return OnnxRuntimeRunner("model.onnx", warmup=warmup)


def test_onnx_runner_prefers_cuda_and_keeps_cpu(monkeypatch):
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    session = fake_session()
    session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch(
        "services.base.inference_runner.ort.InferenceSession", return_value=session
    ) as factory:
        runner = OnnxRuntimeRunner("model.onnx", warmup=False)

    # 现在 CUDAExecutionProvider 会被转为 (name, options) 元组
    providers_arg = factory.call_args.kwargs["providers"]
    assert len(providers_arg) == 2
    assert providers_arg[0][0] == "CUDAExecutionProvider"  # 元组的第一项是名字
    assert isinstance(providers_arg[0][1], dict)  # 第二项是选项字典
    assert "cudnn_conv_algo_search" in providers_arg[0][1]
    assert providers_arg[1] == "CPUExecutionProvider"  # CPU 保持字符串
    assert runner.providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


def test_onnx_runner_uses_requested_sequential_execution_mode():
    session = fake_session()

    with patch(
        "services.base.inference_runner.ort.InferenceSession", return_value=session
    ) as factory:
        OnnxRuntimeRunner(
            "model.onnx", warmup=False, execution_mode="sequential"
        )

    assert (
        factory.call_args.kwargs["sess_options"].execution_mode
        == ort.ExecutionMode.ORT_SEQUENTIAL
    )


def test_onnx_runner_can_write_runtime_profile():
    session = fake_session()
    session.end_profiling.return_value = "profile.json"

    with patch(
        "services.base.inference_runner.ort.InferenceSession", return_value=session
    ) as factory:
        runner = OnnxRuntimeRunner(
            "model.onnx", warmup=False, enable_profiling=True
        )

    assert factory.call_args.kwargs["sess_options"].enable_profiling is True
    assert runner.end_profiling() == "profile.json"
    session.end_profiling.assert_called_once_with()


def test_onnx_runner_exposes_tensor_metadata_and_protocol():
    runner = make_runner(fake_session(input_shape=["batch", 3, 8, 8]))

    assert isinstance(runner, InferenceRunner)
    assert runner.input_infos == (
        TensorInfo("images", ("batch", 3, 8, 8), "tensor(float)"),
    )
    assert runner.output_infos == (
        TensorInfo("output", (1, 6), "tensor(float)"),
    )


def test_onnx_runner_warms_up_dynamic_batch_axis():
    session = fake_session(input_shape=["batch", 3, 8, 8])

    make_runner(session, warmup=True)

    inputs = session.run.call_args.args[1]
    assert inputs["images"].shape == (1, 3, 8, 8)
    assert inputs["images"].dtype == np.float32


def test_onnx_runner_skips_warmup_for_non_batch_dynamic_axis():
    session = fake_session(input_shape=["batch", 3, 48, "width"])

    with patch("services.base.inference_runner.vision_logger") as logger:
        make_runner(session, warmup=True)

    session.run.assert_not_called()
    assert "跳过预热" in logger.warning.call_args.args[0]


def test_onnx_runner_runs_all_declared_outputs():
    session = fake_session()
    expected = [np.ones((1, 6), dtype=np.float32)]
    session.run.return_value = expected
    runner = make_runner(session)
    inputs = {"images": np.zeros((1, 3, 8, 8), dtype=np.float32)}

    assert runner.run(inputs) is expected
    session.run.assert_called_once_with(["output"], inputs)


def test_onnx_runner_wraps_execution_error():
    session = fake_session()
    session.run.side_effect = RuntimeError("boom")
    runner = make_runner(session)

    with pytest.raises(ModelInferenceError, match="模型推理失败") as exc_info:
        runner.run({"images": np.zeros((1, 3, 8, 8), np.float32)})

    assert exc_info.value.context["original_error"] == "boom"


def test_onnx_runner_wraps_model_loading_error():
    with patch(
        "services.base.inference_runner.ort.InferenceSession",
        side_effect=RuntimeError("invalid model"),
    ):
        with pytest.raises(ModelInferenceError, match="模型加载失败") as exc_info:
            OnnxRuntimeRunner("broken.onnx", warmup=False)

    assert exc_info.value.context["original_error"] == "invalid model"


def test_onnx_runner_rejects_cpu_fallback_when_cuda_is_required():
    session = fake_session()

    with patch(
        "services.base.inference_runner.ort.InferenceSession",
        return_value=session,
    ):
        with pytest.raises(ModelInferenceError, match="CUDA"):
            OnnxRuntimeRunner(
                "/private/weights/model.onnx",
                warmup=False,
                require_cuda=True,
            )


def test_onnx_runner_accepts_cuda_and_registers_runtime_status():
    session = fake_session()
    session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    registry = MagicMock()

    with patch(
        "services.base.inference_runner.ort.InferenceSession",
        return_value=session,
    ):
        runner = OnnxRuntimeRunner(
            "weights/model.onnx",
            warmup=False,
            require_cuda=True,
            status_registry=registry,
        )

    assert "CUDAExecutionProvider" in runner.providers
    registry.register.assert_called_once_with(
        "weights/model.onnx",
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
    )


def test_onnx_runner_default_policy_allows_cpu_session():
    runner = make_runner(fake_session())

    assert runner.providers == ("CPUExecutionProvider",)
