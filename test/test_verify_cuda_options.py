from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import verify_cuda_options


def _cuda_options() -> dict[str, str]:
    return {
        key: str(value)
        for key, value in verify_cuda_options.OnnxRuntimeRunner
        ._cuda_provider_options()
        .items()
    }


def _runner(providers, provider_options):
    session = MagicMock()
    session.get_provider_options.return_value = provider_options
    return SimpleNamespace(providers=providers, _session=session)


def _assert_failure(result: int, output: str) -> None:
    assert result != 0
    assert "验证完成" not in output
    assert "成功" not in output


def test_main_fails_when_model_is_missing(tmp_path, capsys):
    result = verify_cuda_options.main(
        tmp_path / "missing.onnx",
        runner_factory=MagicMock(),
    )

    _assert_failure(result, capsys.readouterr().out)


def test_main_fails_when_model_loading_fails(tmp_path, capsys):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    factory = MagicMock(side_effect=RuntimeError("load failed"))

    result = verify_cuda_options.main(model_path, runner_factory=factory)

    _assert_failure(result, capsys.readouterr().out)
    factory.assert_called_once_with(
        str(model_path),
        warmup=False,
        require_cuda=True,
    )


def test_main_fails_for_cpu_fallback(tmp_path, capsys):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    factory = MagicMock(
        return_value=_runner(
            ("CPUExecutionProvider",),
            {"CPUExecutionProvider": {}},
        )
    )

    result = verify_cuda_options.main(model_path, runner_factory=factory)

    _assert_failure(result, capsys.readouterr().out)
    factory.assert_called_once_with(
        str(model_path),
        warmup=False,
        require_cuda=True,
    )


@pytest.mark.parametrize(
    "provider_options",
    [
        {},
        {"CUDAExecutionProvider": {"arena_extend_strategy": "unexpected"}},
    ],
)
def test_main_fails_when_cuda_options_are_missing_or_mismatched(
    tmp_path,
    capsys,
    provider_options,
):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    factory = MagicMock(
        return_value=_runner(
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
            provider_options,
        )
    )

    result = verify_cuda_options.main(model_path, runner_factory=factory)

    _assert_failure(result, capsys.readouterr().out)


def test_main_succeeds_only_for_cuda_with_matching_options(tmp_path, capsys):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    factory = MagicMock(
        return_value=_runner(
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
            {"CUDAExecutionProvider": _cuda_options()},
        )
    )

    result = verify_cuda_options.main(model_path, runner_factory=factory)

    output = capsys.readouterr().out
    assert result == 0
    assert "验证完成" in output
    factory.assert_called_once_with(
        str(model_path),
        warmup=False,
        require_cuda=True,
    )


def test_main_fails_when_provider_options_cannot_be_read(tmp_path, capsys):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    runner = _runner(("CUDAExecutionProvider",), {})
    runner._session.get_provider_options.side_effect = RuntimeError("unavailable")

    result = verify_cuda_options.main(
        model_path,
        runner_factory=MagicMock(return_value=runner),
    )

    _assert_failure(result, capsys.readouterr().out)
