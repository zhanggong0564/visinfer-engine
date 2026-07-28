"""Sanitized process-wide inference runtime status."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence


@dataclass(frozen=True)
class ModelRuntimeStatus:
    model_path: str
    providers: tuple[str, ...]
    backend: str = "onnx"
    scenario: str | None = None
    model_role: str | None = None


class RuntimeStatusRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._models: dict[str, ModelRuntimeStatus] = {}

    def register(
        self,
        model_path: str,
        providers: Sequence[str],
        backend: str = "onnx",
        scenario: str | None = None,
        model_role: str | None = None,
    ) -> None:
        key = str(Path(model_path).resolve())
        status = ModelRuntimeStatus(
            model_path=key,
            providers=tuple(providers),
            backend=backend,
            scenario=scenario,
            model_role=model_role,
        )
        with self._lock:
            self._models[key] = status

    def public_snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            statuses = sorted(
                self._models.values(), key=lambda item: item.model_path
            )
        snapshots = []
        for status in statuses:
            snapshot = {
                "model": Path(status.model_path).name,
                "backend": status.backend,
                "providers": list(status.providers),
            }
            if status.scenario:
                snapshot["scenario"] = status.scenario
            if status.model_role:
                snapshot["role"] = status.model_role
            snapshots.append(snapshot)
        return snapshots

    def clear(self) -> None:
        with self._lock:
            self._models.clear()


runtime_status_registry = RuntimeStatusRegistry()
