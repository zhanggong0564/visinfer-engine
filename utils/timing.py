"""轻量阶段耗时收集工具。"""

import time
from contextlib import contextmanager
from typing import Iterator, List, Tuple


class StageTimer:
    """按顺序记录阶段耗时，输出稳定的日志片段。"""

    def __init__(self) -> None:
        self._total_start = time.perf_counter()
        self._stages: List[Tuple[str, float]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000)

    def record(self, name: str, elapsed_ms: float) -> None:
        self._stages.append((name, elapsed_ms))

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._total_start) * 1000

    def summary(self, *, include_total: bool = True) -> str:
        parts = [f"{name}={elapsed_ms:.1f}ms" for name, elapsed_ms in self._stages]
        if include_total:
            parts.append(f"total={self.total_ms:.1f}ms")
        return " ".join(parts)
