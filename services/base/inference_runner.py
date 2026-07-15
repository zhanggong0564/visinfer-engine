"""Backend-independent inference runner contracts and ONNX Runtime adapter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import onnxruntime as ort

from config import settings
from schemas.exceptions import ModelInferenceError
from services.base.runtime_status import (
    RuntimeStatusRegistry,
    runtime_status_registry,
)
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
        require_cuda: bool | None = None,
        status_registry: RuntimeStatusRegistry | None = None,
    ) -> None:
        cuda_required = (
            settings.ONNX_REQUIRE_CUDA if require_cuda is None else require_cuda
        )
        registry = (
            runtime_status_registry
            if status_registry is None
            else status_registry
        )
        selected_providers = (
            list(providers) if providers is not None else self._defaults()
        )
        # 统一给 CUDA provider 注入显存/算法策略；对纯字符串和元组都安全
        selected_providers = self._with_cuda_options(selected_providers)
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
            if (
                cuda_required
                and "CUDAExecutionProvider" not in self._providers
            ):
                raise ModelInferenceError(
                    "CUDA 执行提供程序不可用",
                    model=Path(model_path).name,
                    requested_providers=selected_providers,
                    actual_providers=list(self._providers),
                )
            registry.register(model_path, self._providers)
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
        if warmup:
            self._warmup()

    @staticmethod
    def _cuda_provider_options() -> dict:
        """CUDA provider 显存与算法策略。

        关键点(解决大显卡多进程 OOM):
        - cudnn_conv_algo_search=HEURISTIC: 不再按空闲显存穷举卷积算法、
          申请巨型临时 workspace，改为启发式直选。这是 80G 卡多进程 OOM 的主因。
        - arena_extend_strategy=kSameAsRequested: 显存池按需扩张，不再按 2 的幂
          一次抓超大块，降低多进程叠加峰值。
        - gpu_mem_limit: 每进程显存硬上限(0=不限)，多进程共卡时的双保险。
        """
        options: dict = {
            "device_id": settings.ORT_CUDA_DEVICE_ID,
            "cudnn_conv_algo_search": settings.ORT_CUDNN_CONV_ALGO_SEARCH,
            "arena_extend_strategy": settings.ORT_ARENA_EXTEND_STRATEGY,
        }
        limit_gb = settings.ORT_CUDA_MEM_LIMIT_GB
        if limit_gb and limit_gb > 0:
            options["gpu_mem_limit"] = int(limit_gb * 1024 * 1024 * 1024)
        return options

    @classmethod
    def _with_cuda_options(cls, providers: list) -> list:
        """把 provider 列表里裸的 'CUDAExecutionProvider' 字符串替换为
        (name, options) 元组；已是元组或非 CUDA 的原样保留。
        这样即使调用方显式传入字符串 provider，也能享受显存策略。"""
        cuda_opts = cls._cuda_provider_options()
        normalized = []
        for item in providers:
            if item == "CUDAExecutionProvider":
                normalized.append(("CUDAExecutionProvider", cuda_opts))
            else:
                normalized.append(item)
        return normalized

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
