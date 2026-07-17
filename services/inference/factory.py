"""ONNX inference runner creation and status registration."""

from dataclasses import dataclass

from services.inference.backends import OnnxRuntimeOptions, OnnxRuntimeRunner
from services.inference.contract import InferenceRunner
from services.inference.status import (
    RuntimeStatusRegistry,
    runtime_status_registry,
)

@dataclass(frozen=True)
class RunnerSpec:
    scenario: str
    onnx_path: str


def create_inference_runner(
    spec: RunnerSpec,
    onnx_options: OnnxRuntimeOptions,
    status_registry: RuntimeStatusRegistry | None = None,
) -> InferenceRunner:
    """Create an ONNX Runtime runner and record its runtime status."""

    registry = status_registry or runtime_status_registry
    runner = OnnxRuntimeRunner(spec.onnx_path, onnx_options)
    registry.register(spec.onnx_path, runner.providers, backend="onnx")
    return runner
