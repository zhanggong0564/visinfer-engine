from unittest.mock import MagicMock

import pytest

from services.inference import (
    InferenceRunnerGroup,
    OnnxRuntimeOptions,
    RunnerDefinition,
    RunnerSpec,
)


def _definition(role):
    return RunnerDefinition(
        spec=RunnerSpec(
            scenario="mvs",
            onnx_path=f"/weights/{role}/inference.onnx",
            model_role=role,
        ),
        onnx_options=OnnxRuntimeOptions(
            providers=("CPUExecutionProvider",),
        ),
    )


def test_runner_group_creates_ordered_role_mapping(monkeypatch):
    runners = [MagicMock(), MagicMock()]
    calls = []

    def create(spec, options, status_registry=None):
        calls.append((spec.model_role, options.providers, status_registry))
        return runners[len(calls) - 1]

    monkeypatch.setattr(
        "services.inference.group.create_inference_runner",
        create,
    )
    registry = object()
    group = InferenceRunnerGroup(
        [_definition("det"), _definition("rec")],
        status_registry=registry,
    )

    assert list(group) == ["det", "rec"]
    assert group["det"] is runners[0]
    assert calls == [
        ("det", ("CPUExecutionProvider",), registry),
        ("rec", ("CPUExecutionProvider",), registry),
    ]


@pytest.mark.parametrize(
    "definitions",
    [
        [_definition("det"), _definition("det")],
        [
            RunnerDefinition(
                RunnerSpec("mvs", "/weights/a.onnx"),
                OnnxRuntimeOptions(providers=("CPUExecutionProvider",)),
            )
        ],
    ],
)
def test_runner_group_rejects_missing_or_duplicate_roles(definitions):
    with pytest.raises(ValueError, match="model_role"):
        InferenceRunnerGroup(definitions)


def test_runner_group_rejects_empty_definition_list():
    with pytest.raises(ValueError, match="不能为空"):
        InferenceRunnerGroup([])


def test_runner_group_rolls_back_in_reverse_order(monkeypatch):
    events = []

    def runner(role):
        value = MagicMock()
        value.close.side_effect = lambda: events.append(f"close:{role}")
        return value

    created = [runner("det"), runner("rec")]
    calls = iter([*created, RuntimeError("create failed")])

    def create(*args, **kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "services.inference.group.create_inference_runner",
        create,
    )

    with pytest.raises(RuntimeError, match="create failed"):
        InferenceRunnerGroup(
            [_definition("det"), _definition("rec"), _definition("cls")]
        )

    assert events == ["close:rec", "close:det"]


def test_runner_group_close_is_reverse_idempotent_and_continues(monkeypatch):
    events = []
    runners = []
    for role in ("det", "rec"):
        value = MagicMock()
        value.close.side_effect = lambda role=role: events.append(role)
        runners.append(value)
    runners[1].close.side_effect = [
        RuntimeError("close failed"),
    ]
    monkeypatch.setattr(
        "services.inference.group.create_inference_runner",
        MagicMock(side_effect=runners),
    )
    group = InferenceRunnerGroup([_definition("det"), _definition("rec")])

    group.close()
    group.close()

    runners[1].close.assert_called_once_with()
    runners[0].close.assert_called_once_with()
