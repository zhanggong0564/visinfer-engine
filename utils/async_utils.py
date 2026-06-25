"""异步服务中卸载同步 CPU/IO 工作的标准库适配。"""

import asyncio
import contextvars
from functools import partial


async def run_sync(func, /, *args, **kwargs):
    """在线程池中执行同步函数，返回 awaitable，零轮询等待。

    走事件循环默认的 ThreadPoolExecutor（有界：默认 min(32, cpu+4)），而非每次
    裸起一个 threading.Thread——单请求会多次 run_sync（解码/推理/落盘/埋点），
    无界建线程在高并发下会把线程数顶爆。contextvars 通过 ctx.run 显式传播，
    保证工作线程看到调用方的上下文。
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    call = partial(context.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, call)
