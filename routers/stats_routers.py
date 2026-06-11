"""接口调用统计查询路由。

裸 APIRouter（非 BaseRouter）：无图片上传、无检测器，仅查询账本。
RouterRegistry._collect_routers_from_module 会自动发现模块级 APIRouter
实例并挂到 /api/v1 前缀。
"""

import re
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from schemas.exceptions import InvalidParamsError
from services.call_stats import call_stats_recorder

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

stats_router = APIRouter()


@stats_router.get(
    "/stats",
    summary="接口调用统计查询",
    description="按场景(detector_type)统计各检测接口的调用次数，"
    "维度为 场景 × 日期 × 结果(ok/ng/error)，支持场景与日期范围（闭区间）过滤",
)
async def get_call_stats(
    scene: Optional[str] = Query(None, description="按场景(detector_type)过滤，如 panel_label"),
    start_date: Optional[str] = Query(None, description="起始日期 YYYY-MM-DD（含当天）"),
    end_date: Optional[str] = Query(None, description="截止日期 YYYY-MM-DD（含当天）"),
):
    for name, value in (("start_date", start_date), ("end_date", end_date)):
        if value is not None and not _DATE_RE.match(value):
            raise InvalidParamsError(f"{name} 格式非法，应为 YYYY-MM-DD")
    # sqlite 读为同步操作，丢线程池避免阻塞事件循环
    result = await run_in_threadpool(
        call_stats_recorder.query, scene=scene, start_date=start_date, end_date=end_date
    )
    return {"code": 1, "message": "查询成功", "result": result}
