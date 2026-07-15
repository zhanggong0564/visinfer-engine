"""Process-wide asynchronous admission for synchronous detector execution."""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from config import settings
from utils import vision_logger
from utils.async_utils import run_sync, submit_sync


T = TypeVar("T")


@dataclass(frozen=True)
class AdmissionSnapshot:
    max_concurrency: int
    active: int
    waiting: int


class InferenceAdmissionController:
    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be non-negative")
        self._max_concurrency = max_concurrency
        self._semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )
        self._active = 0
        self._waiting = 0

    def snapshot(self) -> AdmissionSnapshot:
        return AdmissionSnapshot(
            max_concurrency=self._max_concurrency,
            active=self._active,
            waiting=self._waiting,
        )

    async def run(
        self,
        detector_type: str,
        func: Callable[..., T],
        /,
        *args,
        **kwargs,
    ) -> T:
        if self._semaphore is None:
            return await run_sync(func, *args, **kwargs)

        wait_started = time.perf_counter()
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1

        self._active += 1
        try:
            future = submit_sync(func, *args, **kwargs)
        except BaseException:
            self._release()
            raise

        future.add_done_callback(lambda _future: self._release())
        wait_ms = (time.perf_counter() - wait_started) * 1000
        vision_logger.info(
            "推理准入 scene={} wait_ms={:.1f} active={} waiting={} limit={}",
            detector_type,
            wait_ms,
            self._active,
            self._waiting,
            self._max_concurrency,
        )
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(self._consume_exception)
            raise

    def _release(self) -> None:
        self._active -= 1
        assert self._semaphore is not None
        self._semaphore.release()

    @staticmethod
    def _consume_exception(future: asyncio.Future) -> None:
        if not future.cancelled():
            future.exception()


inference_admission_controller = InferenceAdmissionController(
    settings.INFERENCE_MAX_CONCURRENCY
)
