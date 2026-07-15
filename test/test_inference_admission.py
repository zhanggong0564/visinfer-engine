import asyncio
import threading
import time

import pytest

from services.inference_admission import InferenceAdmissionController


def _run(coro):
    return asyncio.run(coro)


def test_admission_zero_preserves_unlimited_execution():
    controller = InferenceAdmissionController(0)

    async def _scenario():
        return await controller.run("scene", lambda value: value + 1, 41)

    assert _run(_scenario()) == 42
    assert controller.snapshot().max_concurrency == 0


def test_admission_serializes_four_requests_without_rejecting():
    controller = InferenceAdmissionController(1)
    lock = threading.Lock()
    active = 0
    maximum = 0

    def detect(value):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return value

    async def _scenario():
        return await asyncio.gather(
            *(controller.run("scene", detect, value) for value in range(4))
        )

    assert _run(_scenario()) == [0, 1, 2, 3]
    assert maximum == 1
    assert controller.snapshot().active == 0
    assert controller.snapshot().waiting == 0


def test_cancelling_waiter_does_not_submit_detection():
    controller = InferenceAdmissionController(1)
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def first_detect():
        first_started.set()
        release_first.wait(timeout=1)

    def second_detect():
        second_started.set()

    async def _scenario():
        first = asyncio.create_task(controller.run("scene", first_detect))
        await asyncio.to_thread(first_started.wait, 1)
        second = asyncio.create_task(controller.run("scene", second_detect))
        await asyncio.sleep(0)
        assert controller.snapshot().waiting == 1
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert controller.snapshot().waiting == 0
        assert not second_started.is_set()
        release_first.set()
        await first

    _run(_scenario())


def test_cancelling_active_request_releases_only_after_worker_finishes():
    controller = InferenceAdmissionController(1)
    started = threading.Event()
    release = threading.Event()

    def detect():
        started.set()
        release.wait(timeout=1)

    async def _scenario():
        task = asyncio.create_task(controller.run("scene", detect))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert controller.snapshot().active == 1
        release.set()
        while controller.snapshot().active:
            await asyncio.sleep(0)
        assert controller.snapshot().active == 0

    _run(_scenario())


def test_detection_exception_releases_capacity():
    controller = InferenceAdmissionController(1)

    def fail():
        raise RuntimeError("boom")

    async def _scenario():
        with pytest.raises(RuntimeError, match="boom"):
            await controller.run("scene", fail)
        return await controller.run("scene", lambda: "recovered")

    assert _run(_scenario()) == "recovered"
    assert controller.snapshot().active == 0
