"""异步服务中卸载同步 CPU/IO 工作的标准库适配。"""

import asyncio
import contextvars
from functools import partial
from typing import Callable, TypeVar


T = TypeVar("T")


def submit_sync(
    func: Callable[..., T], /, *args, **kwargs
) -> asyncio.Future[T]:
    """Submit synchronous work with the caller's context and return its future."""
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    call = partial(context.run, func, *args, **kwargs)
    return loop.run_in_executor(None, call)


async def run_sync(func: Callable[..., T], /, *args, **kwargs) -> T:
    """Execute synchronous work in the default bounded executor."""
    return await submit_sync(func, *args, **kwargs)
