"""异步服务中卸载同步 CPU/IO 工作的标准库适配。"""

import asyncio
import contextvars
import threading
from functools import partial


async def run_sync(func, /, *args, **kwargs):
    """在线程中执行同步函数，通过 Future 实现零轮询等待。

    用 loop.call_soon_threadsafe 把结果/异常从工作线程回送到事件循环，
    避免 asyncio.sleep 忙等轮询（推理等耗时操作会产生数百次无效唤醒）。
    """
    context = contextvars.copy_context()
    call = partial(func, *args, **kwargs)
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def worker():
        try:
            result = context.run(call)
            loop.call_soon_threadsafe(fut.set_result, result)
        except BaseException as exc:
            loop.call_soon_threadsafe(fut.set_exception, exc)

    threading.Thread(target=worker, daemon=True).start()
    return await fut
