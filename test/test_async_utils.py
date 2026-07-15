import asyncio
import contextvars
import time

import pytest

from utils.async_utils import run_sync, submit_sync


def _new_loop_run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_run_sync_returns_value():
    result = _new_loop_run(run_sync(lambda: 42))
    assert result == 42


def test_run_sync_supports_multiple_sequential_calls():
    async def _run_twice():
        first = await run_sync(lambda: 1)
        second = await run_sync(lambda: 2)
        return first, second

    assert _new_loop_run(_run_twice()) == (1, 2)


def test_run_sync_propagates_exception():
    def _boom():
        raise ValueError("sync error")

    with pytest.raises(ValueError, match="sync error"):
        _new_loop_run(run_sync(_boom))


def test_run_sync_does_not_block_event_loop():
    """工作线程执行期间，事件循环应能推进其他协程。"""
    log = []

    async def _slow():
        await run_sync(time.sleep, 0.05)
        log.append("slow_done")

    async def _fast():
        await asyncio.sleep(0)
        log.append("fast_done")

    async def _both():
        await asyncio.gather(_slow(), _fast())

    _new_loop_run(_both())
    assert log == ["fast_done", "slow_done"]


def test_run_sync_passes_args_and_kwargs():
    result = _new_loop_run(run_sync(lambda a, b=0: a + b, 3, b=4))
    assert result == 7


def test_submit_sync_returns_future_and_value():
    async def _submit():
        future = submit_sync(lambda: 42)
        assert isinstance(future, asyncio.Future)
        return await future

    assert _new_loop_run(_submit()) == 42


def test_submit_sync_propagates_contextvars():
    marker = contextvars.ContextVar("marker", default="missing")

    async def _submit():
        marker.set("request-value")
        return await submit_sync(marker.get)

    assert _new_loop_run(_submit()) == "request-value"
