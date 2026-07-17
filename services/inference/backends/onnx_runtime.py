"""ONNX Runtime inference backend."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

from schemas.exceptions import ModelInferenceError
from services.inference.contract import TensorInfo
from utils import vision_logger


ort.set_default_logger_severity(3)


@dataclass(frozen=True)
class OnnxRuntimeOptions:
    providers: tuple[str, ...] | None = None
    warmup: bool = True
    execution_mode: Literal["parallel", "sequential"] = "parallel"
    enable_profiling: bool = False
    require_cuda: bool = False
    cuda_device_id: int = 0
    cudnn_conv_algo_search: str = "HEURISTIC"
    arena_extend_strategy: str = "kSameAsRequested"
    cuda_mem_limit_gb: float = 0.0

    @classmethod
    def from_settings(cls, settings, **overrides) -> "OnnxRuntimeOptions":
        options = cls(
            require_cuda=settings.ONNX_REQUIRE_CUDA,
            cuda_device_id=settings.ORT_CUDA_DEVICE_ID,
            cudnn_conv_algo_search=settings.ORT_CUDNN_CONV_ALGO_SEARCH,
            arena_extend_strategy=settings.ORT_ARENA_EXTEND_STRATEGY,
            cuda_mem_limit_gb=settings.ORT_CUDA_MEM_LIMIT_GB,
        )
        return replace(options, **overrides)

    def cuda_provider_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "device_id": self.cuda_device_id,
            "cudnn_conv_algo_search": self.cudnn_conv_algo_search,
            "arena_extend_strategy": self.arena_extend_strategy,
        }
        if self.cuda_mem_limit_gb > 0:
            options["gpu_mem_limit"] = int(
                self.cuda_mem_limit_gb * 1024**3
            )
        return options


class OnnxRuntimeRunner:
    """Execute an ONNX model through ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        options: OnnxRuntimeOptions,
    ) -> None:
        self._closed = False
        self._options = options
        selected_providers = (
            list(options.providers)
            if options.providers is not None
            else self._defaults()
        )
        selected_providers = self._with_cuda_options(selected_providers)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        execution_modes = {
            "parallel": ort.ExecutionMode.ORT_PARALLEL,
            "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
        }
        session_options.execution_mode = execution_modes[options.execution_mode]
        session_options.enable_profiling = options.enable_profiling

        try:
            self._session = ort.InferenceSession(
                model_path,
                providers=selected_providers,
                sess_options=session_options,
            )
            self._input_infos = self._tensor_infos(self._session.get_inputs())
            self._output_infos = self._tensor_infos(self._session.get_outputs())
            self._providers = tuple(self._session.get_providers())
            if (
                options.require_cuda
                and "CUDAExecutionProvider" not in self._providers
            ):
                raise ModelInferenceError(
                    "CUDA 执行提供程序不可用",
                    model=Path(model_path).name,
                    requested_providers=selected_providers,
                    actual_providers=list(self._providers),
                )
            self._output_names = [info.name for info in self._output_infos]
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                "模型加载失败", original_error=str(exc)
            ) from exc

        vision_logger.info(
            f"模型 {Path(model_path).name} 使用的执行提供程序: "
            f"{list(self._providers)}"
        )
        if options.warmup:
            self._warmup()

    def _cuda_provider_options(self) -> dict:
        return self._options.cuda_provider_options()

    def _with_cuda_options(self, providers: list) -> list:
        cuda_options = self._cuda_provider_options()
        return [
            ("CUDAExecutionProvider", cuda_options)
            if item == "CUDAExecutionProvider"
            else item
            for item in providers
        ]

    @staticmethod
    def _defaults() -> list[str]:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
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
        self.run(
            {
                input_info.name: np.zeros(
                    concrete_shape, dtype=np.float32
                )
            }
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        try:
            if self._closed:
                raise RuntimeError("ONNX Runtime runner 已关闭")
            return self._session.run(self._output_names, inputs)
        except Exception as exc:
            if isinstance(exc, ModelInferenceError):
                raise
            raise ModelInferenceError(
                "模型推理失败", original_error=str(exc)
            ) from exc

    def end_profiling(self) -> str:
        return self._session.end_profiling()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session = None
