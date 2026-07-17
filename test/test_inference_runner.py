from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import onnxruntime as ort
import pytest

from schemas.exceptions import ModelInferenceError
from services.inference import (
    InferenceRunner,
    OnnxRuntimeOptions,
    OnnxRuntimeRunner,
    TensorInfo,
)


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
        "services.inference.backends.onnx_runtime.ort.InferenceSession",
        return_value=session,
    ):
        return OnnxRuntimeRunner(
            "model.onnx", OnnxRuntimeOptions(warmup=warmup)
        )


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
        "services.inference.backends.onnx_runtime.ort.InferenceSession",
        return_value=session,
    ) as factory:
        runner = OnnxRuntimeRunner(
            "model.onnx",
            OnnxRuntimeOptions(
                warmup=False,
                cuda_device_id=3,
                cudnn_conv_algo_search="HEURISTIC",
                arena_extend_strategy="kSameAsRequested",
                cuda_mem_limit_gb=2.0,
            ),
        )

    # 现在 CUDAExecutionProvider 会被转为 (name, options) 元组
    providers_arg = factory.call_args.kwargs["providers"]
    assert len(providers_arg) == 2
    assert providers_arg[0][0] == "CUDAExecutionProvider"  # 元组的第一项是名字
    assert isinstance(providers_arg[0][1], dict)  # 第二项是选项字典
    assert "cudnn_conv_algo_search" in providers_arg[0][1]
    assert providers_arg[0][1]["device_id"] == 3
    assert providers_arg[0][1]["gpu_mem_limit"] == 2 * 1024**3
    assert providers_arg[1] == "CPUExecutionProvider"  # CPU 保持字符串
    assert runner.providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


def test_onnx_runner_uses_requested_sequential_execution_mode():
    session = fake_session()

    with patch(
        "services.inference.backends.onnx_runtime.ort.InferenceSession",
        return_value=session,
    ) as factory:
        OnnxRuntimeRunner(
            "model.onnx",
            OnnxRuntimeOptions(warmup=False, execution_mode="sequential"),
        )

    assert (
        factory.call_args.kwargs["sess_options"].execution_mode
        == ort.ExecutionMode.ORT_SEQUENTIAL
    )


def test_onnx_runner_can_write_runtime_profile():
    session = fake_session()
    session.end_profiling.return_value = "profile.json"

    with patch(
        "services.inference.backends.onnx_runtime.ort.InferenceSession",
        return_value=session,
    ) as factory:
        runner = OnnxRuntimeRunner(
            "model.onnx",
            OnnxRuntimeOptions(warmup=False, enable_profiling=True),
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

    with patch("services.inference.backends.onnx_runtime.vision_logger") as logger:
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
        "services.inference.backends.onnx_runtime.ort.InferenceSession",
        side_effect=RuntimeError("invalid model"),
    ):
        with pytest.raises(ModelInferenceError, match="模型加载失败") as exc_info:
            OnnxRuntimeRunner(
                "broken.onnx", OnnxRuntimeOptions(warmup=False)
            )

    assert exc_info.value.context["original_error"] == "invalid model"


def test_onnx_runner_rejects_cpu_fallback_when_cuda_is_required():
    session = fake_session()

    with patch(
        "services.inference.backends.onnx_runtime.ort.InferenceSession",
        return_value=session,
    ):
        with pytest.raises(ModelInferenceError, match="CUDA"):
            OnnxRuntimeRunner(
                "/private/weights/model.onnx",
                OnnxRuntimeOptions(warmup=False, require_cuda=True),
            )


def test_onnx_runner_accepts_cuda_without_registering_runtime_status():
    session = fake_session()
    session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch(
        "services.inference.backends.onnx_runtime.ort.InferenceSession",
        return_value=session,
    ):
        runner = OnnxRuntimeRunner(
            "weights/model.onnx",
            OnnxRuntimeOptions(warmup=False, require_cuda=True),
        )

    assert "CUDAExecutionProvider" in runner.providers


def test_onnx_runner_default_policy_allows_cpu_session():
    runner = make_runner(fake_session())

    assert runner.providers == ("CPUExecutionProvider",)


def test_onnx_runner_close_is_idempotent_and_satisfies_protocol():
    runner = make_runner(fake_session())

    runner.close()
    runner.close()

    assert isinstance(runner, InferenceRunner)
    assert runner._session is None


def test_onnx_options_are_built_explicitly_from_application_settings():
    settings = SimpleNamespace(
        ONNX_REQUIRE_CUDA=True,
        ORT_CUDA_DEVICE_ID=2,
        ORT_CUDNN_CONV_ALGO_SEARCH="DEFAULT",
        ORT_ARENA_EXTEND_STRATEGY="kNextPowerOfTwo",
        ORT_CUDA_MEM_LIMIT_GB=1.5,
    )

    options = OnnxRuntimeOptions.from_settings(settings, warmup=False)

    assert options.require_cuda is True
    assert options.cuda_device_id == 2
    assert options.cudnn_conv_algo_search == "DEFAULT"
    assert options.arena_extend_strategy == "kNextPowerOfTwo"
    assert options.cuda_mem_limit_gb == 1.5
    assert options.warmup is False
