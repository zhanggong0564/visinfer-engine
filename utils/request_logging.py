"""Bounded event summaries and streaming HTTP lifecycle logging."""

import json
import time
from typing import Any
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from schemas.inspection import InspectionVerdict
from schemas.error_codes import ErrorCode
from utils import vision_logger
from utils.log_context import RequestLogContext, current_log_context, use_log_context


def log_value(value: Any, *, max_chars: int = 256, max_items: int = 50) -> Any:
    """Bound individual values without silently dropping the rest of a summary."""
    if isinstance(value, str) and len(value) > max_chars:
        return {"value": value[:max_chars], "chars": len(value), "truncated": True}
    if isinstance(value, (list, tuple)):
        items = [
            log_value(item, max_chars=max_chars, max_items=max_items)
            for item in value[:max_items]
        ]
        if len(value) <= max_items:
            return items
        return {"items": items, "total": len(value), "omitted": len(value) - max_items}
    if isinstance(value, dict):
        return {
            key: log_value(item, max_chars=max_chars, max_items=max_items)
            for key, item in value.items()
        }
    return value


def log_json(value: dict) -> str:
    return json.dumps(log_value(value), ensure_ascii=False, separators=(",", ":"), default=str)


def parameter_summary(params: Any) -> dict:
    """Common fields only; scene-specific decision parameters belong to the plugin."""
    def get(obj: Any, name: str, default: Any = None) -> Any:
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    model = get(params, "modelParams", {})
    return {
        "product": get(params, "product"),
        "type": get(params, "type"),
        "product_type": get(model, "product_type"),
        "rule": get(model, "rule"),
        "guide_line_count": len(get(model, "guide_line") or []),
        "example_image_count": len(get(model, "example_images") or []),
    }


def log_detection_result(result: dict, code: int = int(ErrorCode.SUCCESS)) -> None:
    data = result.get("result")
    if not isinstance(data, dict):
        data = result
    verdict = InspectionVerdict.from_value(data.get("verdict", data.get("status")))
    context = current_log_context()
    if context is not None:
        context.code = code
        context.verdict = verdict.value if verdict else None
    details = data.get("detailList") or []
    counts = {item.value: 0 for item in InspectionVerdict}
    for item in details:
        item_verdict = InspectionVerdict.from_value(item.get("verdict", item.get("status")))
        if item_verdict:
            counts[item_verdict.value] += 1
    vision_logger.info("检测完成 {}", log_json({
        "code": code, "verdict": verdict, "reason": data.get("message"),
        "detail_count": len(details), "detail_verdict_counts": counts,
    }), event="detection.completed")


def log_request_error(
    request: Request, exc: Exception, code: ErrorCode, details: Any = None,
) -> dict[str, str]:
    context = (
        current_log_context()
        or request.scope.get("state", {}).get("request_log_context")
        or RequestLogContext()
    )
    if not context.response_sent:
        context.code = int(code)
        context.verdict = None
    endpoint = request.scope.get("endpoint")
    router = getattr(endpoint, "__self__", None)
    context.scene = getattr(router, "detector_type", context.scene)
    with use_log_context(context):
        logger = vision_logger
        if not context.exception_logged and code in (ErrorCode.MODEL_INFERENCE_ERROR, ErrorCode.INTERNAL_ERROR):
            logger = logger.opt(exception=exc)
            context.exception_logged = True
        logger.error("请求异常摘要 {}", log_json({
            "http_status": 200, "code": int(code), "error_type": type(exc).__name__,
            "details": details if details is not None else str(exc),
            "response_already_sent": context.response_sent,
        }), event="background.failed" if context.response_sent else "request.failed")
    return {"X-Request-ID": context.request_id}


class RequestLoggingMiddleware:
    """Observe ASGI receive/send without pre-reading, buffering or parsing the body.

    receive_wait_ms measures awaited receive calls, not pure network transfer time;
    pre_handler_ms includes upload/form parsing and framework scheduling.
    """

    def __init__(self, app: ASGIApp, skip_paths: set[str]) -> None:
        self.app = app
        self.skip_paths = skip_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        context = RequestLogContext()
        scope.setdefault("state", {})["request_log_context"] = context
        received_bytes = 0
        receive_wait_ms = 0.0
        body_finished = None
        response_started = None
        response_finished = None
        status = None

        async def logged_receive() -> Message:
            nonlocal received_bytes, receive_wait_ms, body_finished
            started = time.perf_counter()
            try:
                message = await receive()
            finally:
                receive_wait_ms += (time.perf_counter() - started) * 1000
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if not message.get("more_body", False):
                    body_finished = time.perf_counter()
            return message

        async def logged_send(message: Message) -> None:
            nonlocal status, response_started, response_finished
            if message["type"] == "http.response.start":
                status = message["status"]
                response_started = time.perf_counter()
                headers = [
                    (key, value) for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", context.request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_finished = time.perf_counter()
                context.response_sent = True

        with use_log_context(context):
            try:
                await self.app(scope, logged_receive, logged_send)
            except Exception as exc:
                if response_finished is None:
                    context.code = int(ErrorCode.INTERNAL_ERROR)
                    context.verdict = None
                context.exception_logged = True
                vision_logger.opt(exception=exc).error(
                    "请求执行异常 response_sent={}", response_finished is not None,
                    event="http.exception",
                )
                raise
            finally:
                finished = time.perf_counter()
                quiet_probe = (
                    scope["path"] in self.skip_paths
                    and status is not None
                    and status < 400
                    and not context.exception_logged
                )
                if not quiet_probe:
                    def elapsed(end, start=context.started):
                        return round((end - start) * 1000, 1) if end is not None else None

                    vision_logger.info("HTTP 请求完成 {}", log_json({
                        "method": scope["method"], "path": scope["path"],
                        "http_status": status, "code": context.code,
                        "verdict": context.verdict, "request_bytes": received_bytes,
                        "receive_wait_ms": round(receive_wait_ms, 1),
                        "body_received_ms": elapsed(body_finished),
                        "pre_handler_ms": elapsed(context.handler_started),
                        "response_start_ms": elapsed(response_started),
                        "response_total_ms": elapsed(response_finished),
                        "background_ms": elapsed(finished, response_finished) if response_finished else None,
                        "lifecycle_ms": elapsed(finished),
                        "response_complete": response_finished is not None,
                    }), event="http.completed")
