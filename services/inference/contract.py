"""Backend-independent inference contracts."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TensorInfo:
    """Metadata describing one model tensor."""

    name: str
    shape: tuple[object, ...]
    dtype: str


@runtime_checkable
class InferenceRunner(Protocol):
    """Common execution and lifecycle contract for inference backends."""

    @property
    def input_infos(self) -> tuple[TensorInfo, ...]: ...

    @property
    def output_infos(self) -> tuple[TensorInfo, ...]: ...

    @property
    def providers(self) -> tuple[str, ...]: ...

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...

    def close(self) -> None: ...
