"""Backend-independent inference runner contracts and ONNX Runtime adapter."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import onnxruntime as ort

from schemas.exceptions import ModelInferenceError
from utils import vision_logger


@dataclass(frozen=True)
class TensorInfo:
    """Metadata describing one model tensor."""

    name: str
    shape: tuple[object, ...]
    dtype: str


@runtime_checkable
class InferenceRunner(Protocol):
    """Backend-independent model execution contract."""

    @property
    def input_infos(self) -> tuple[TensorInfo, ...]: ...

    @property
    def output_infos(self) -> tuple[TensorInfo, ...]: ...

    @property
    def providers(self) -> tuple[str, ...]: ...

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...


class OnnxRuntimeRunner:
    """Execute an ONNX model through ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        providers: Sequence[str] | None = None,
        warmup: bool = True,
        execution_mode: str = "parallel",
        enable_profiling: bool = False,
    ) -> None:
        selected_providers = (
            list(providers) if providers is not None else self._defaults()
        )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        execution_modes = {
            "parallel": ort.ExecutionMode.ORT_PARALLEL,
            "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
        }
        try:
            session_options.execution_mode = execution_modes[execution_mode]
        except KeyError as exc:
            raise ValueError(
                "execution_mode must be 'parallel' or 'sequential'"
            ) from exc
        session_options.enable_profiling = enable_profiling

        try:
            self._session = ort.InferenceSession(
                model_path,
                providers=selected_providers,
                sess_options=session_options,
            )
            self._input_infos = self._tensor_infos(self._session.get_inputs())
            self._output_infos = self._tensor_infos(self._session.get_outputs())
            self._providers = tuple(self._session.get_providers())
            self._output_names = [info.name for info in self._output_infos]
        except Exception as exc:
            raise ModelInferenceError(
                "模型加载失败", original_error=str(exc)
            ) from exc

        vision_logger.info(f"使用的执行提供程序: {list(self._providers)}")
        if warmup:
            self._warmup()

    @staticmethod
    def _defaults() -> list[str]:
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _tensor_infos(nodes) -> tuple[TensorInfo, ...]:
        return tuple(
            TensorInfo(node.name, tuple(node.shape), node.type) for node in nodes
        )

    @property
    def input_infos(self) -> tuple[TensorInfo, ...]:
        return self._input_infos

    @property
    def output_infos(self) -> tuple[TensorInfo, ...]:
        return self._output_infos

    @property
    def providers(self) -> tuple[str, ...]:
        return self._providers

    def _warmup(self) -> None:
        if not self._input_infos:
            return

        input_info = self._input_infos[0]
        concrete_shape = []
        for index, dimension in enumerate(input_info.shape):
            if isinstance(dimension, int) and dimension > 0:
                concrete_shape.append(dimension)
            elif index == 0:
                concrete_shape.append(1)
            else:
                vision_logger.warning(
                    f"模型输入含动态维度 {input_info.shape}，跳过预热"
                )
                return

        dummy_input = np.zeros(concrete_shape, dtype=np.float32)
        self.run({input_info.name: dummy_input})

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        try:
            return self._session.run(self._output_names, inputs)
        except Exception as exc:
            raise ModelInferenceError(
                "模型推理失败", original_error=str(exc)
            ) from exc

    def end_profiling(self) -> str:
        """Stop ONNX Runtime profiling and return the generated trace path."""
        return self._session.end_profiling()
