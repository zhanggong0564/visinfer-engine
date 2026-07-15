"""Sanitized process-wide ONNX runtime status."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence


@dataclass(frozen=True)
class ModelRuntimeStatus:
    model_path: str
    providers: tuple[str, ...]


class RuntimeStatusRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._models: dict[str, ModelRuntimeStatus] = {}

    def register(self, model_path: str, providers: Sequence[str]) -> None:
        key = str(Path(model_path).resolve())
        status = ModelRuntimeStatus(model_path=key, providers=tuple(providers))
        with self._lock:
            self._models[key] = status

    def public_snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            statuses = sorted(
                self._models.values(), key=lambda item: item.model_path
            )
        return [
            {
                "model": Path(status.model_path).name,
                "providers": list(status.providers),
            }
            for status in statuses
        ]

    def clear(self) -> None:
        with self._lock:
            self._models.clear()


runtime_status_registry = RuntimeStatusRegistry()
