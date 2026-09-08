"""Request-local logging metadata shared with inference and background threads."""

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class RequestLogContext:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    scene: str = "default"
    product_type: str = "-"
    filename: str = "-"
    handler_started: float | None = None
    code: int | None = None
    verdict: str | None = None
    exception_logged: bool = False
    response_sent: bool = False


_current: ContextVar[RequestLogContext | None] = ContextVar("request_log", default=None)


def current_log_context() -> RequestLogContext | None:
    return _current.get()


@contextmanager
def use_log_context(context: RequestLogContext) -> Iterator[RequestLogContext]:
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


@contextmanager
def detection_log_context(scene: str, filename: str) -> Iterator[RequestLogContext]:
    context = current_log_context() or RequestLogContext()
    context.scene = scene
    context.filename = filename
    context.handler_started = time.perf_counter()
    with use_log_context(context):
        yield context


def enrich_log_record(record: dict) -> None:
    context = current_log_context()
    if context is not None:
        for name in ("request_id", "scene", "product_type", "filename"):
            record["extra"][name] = getattr(context, name)
    # 参数错误等普通事件保持单行；exception 的独立堆栈仍由 loguru 输出。
    record["message"] = record["message"].replace("\r", "\\r").replace("\n", "\\n")
    for name in ("scene", "product_type", "filename"):
        value = str(record["extra"].get(name, "-"))
        if len(value) > 256:
            value = f"{value[:256]}...(chars={len(value)})"
        record["extra"][name] = value.replace("\r", "\\r").replace("\n", "\\n")
