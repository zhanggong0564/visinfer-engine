"""OpenAPI 文档定制逻辑。"""

import json
import re

from fastapi import FastAPI, routing
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import core_schema

from schemas.error_codes import ErrorCode, ERROR_CODE_MESSAGES
from schemas.inspection import InspectionVerdict


def _compact_json_example(data: dict) -> str:
    """Swagger 表单字段示例：json_data 是字符串，因此示例也必须是 JSON 字符串。"""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def configure_openapi_docs(app: FastAPI) -> None:
    """让 Swagger UI 展示与项目实际异常/表单契约一致的 OpenAPI schema。"""

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.get("components", {}).get("schemas", {})
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue
                operation.get("responses", {}).pop("422", None)
        # 新版 FastAPI 保留 include_router 上下文，路径前缀和可见性由上下文提供。
        iter_contexts = getattr(routing, "iter_route_contexts", iter)
        for route in iter_contexts(app.routes):
            original_route = getattr(route, "original_route", route)
            if not isinstance(original_route, APIRoute) or not route.include_in_schema:
                continue
            owner = getattr(route.endpoint, "__self__", None)
            model = getattr(owner, "request_document_model", None)
            if model is None:
                continue
            for method in route.methods:
                operation = schema["paths"][route.path_format].get(method.lower())
                if operation is not None:
                    _apply_request_document(operation, components, model, owner)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi


class _RequestJsonSchema(GenerateJsonSchema):
    def get_default_value(self, schema: core_schema.WithDefaultSchema):
        # 只展开无副作用的空容器工厂，不在生成文档时执行任意业务工厂。
        factory = schema.get("default_factory")
        if factory is list or factory is dict:
            return factory()
        return super().get_default_value(schema)


def _register_request_schema(model: type[BaseModel], components: dict) -> str:
    """为插件模型及其嵌套定义加完整模型名前缀，避免 ModelParams 等重名。"""
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", f"{model.__module__}.{model.__qualname__}")
    schema = model.model_json_schema(
        by_alias=True, mode="validation", schema_generator=_RequestJsonSchema,
    )
    definitions = schema.pop("$defs", {})

    def rewrite(value):
        if isinstance(value, dict):
            return {
                key: f"#/components/schemas/{name}__{item.removeprefix('#/$defs/')}"
                if key == "$ref" and item.startswith("#/$defs/") else rewrite(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    components.update({
        f"{name}__{key}": rewrite(value) for key, value in definitions.items()
    })
    components[name] = rewrite(schema)
    return f"#/components/schemas/{name}"


def _field_table(schema: dict, components: dict) -> str:
    rows = [
        "| 字段 | 类型 | 必填（相对所属对象） | 默认值 | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]

    def resolve(node):
        while "$ref" in node:
            node = components[node["$ref"].rsplit("/", 1)[-1]]
        return node

    def type_label(node):
        node = resolve(node)
        if "anyOf" in node:
            return " / ".join(type_label(item) for item in node["anyOf"])
        if node.get("type") == "array":
            if "prefixItems" in node:
                types = dict.fromkeys(type_label(item) for item in node["prefixItems"])
                return f"array<{' / '.join(types)}>（{len(node['prefixItems'])} 项）"
            return f"array<{type_label(node.get('items', {}))}>"
        return node.get("type", "any")

    def walk(node, prefix="", visited=frozenset()):
        ref = node.get("$ref")
        if ref and ref in visited:
            return
        if ref:
            visited = visited | {ref}
        node = resolve(node)
        for variant in node.get("anyOf", []):
            walk(variant, prefix, visited)
        if node.get("type") == "array":
            if isinstance(node.get("items"), dict):
                walk(node["items"], prefix + "[]", visited)
            for index, item in enumerate(node.get("prefixItems", [])):
                walk(item, f"{prefix}[{index}]", visited)
        for key, field in node.get("properties", {}).items():
            path = f"{prefix}.{key}" if prefix else key
            default = (
                json.dumps(field["default"], ensure_ascii=False)
                if "default" in field else "—"
            )
            description = field.get("description", resolve(field).get("description", ""))
            if "enum" in resolve(field):
                description += " 可选值：" + json.dumps(resolve(field)["enum"], ensure_ascii=False)
            cells = [
                f"`{path}`", f"`{type_label(field)}`",
                "是" if key in node.get("required", []) else "否", default, description,
            ]
            rows.append("| " + " | ".join(
                str(cell).replace("|", "\\|").replace("\n", "<br>") for cell in cells
            ) + " |")
            walk(field, path, visited)

    walk(schema)
    return "\n".join(rows) if len(rows) > 2 else "无业务字段，提交 `json_data={}`。"


def _response_examples(verdicts: tuple[InspectionVerdict, ...]) -> dict:
    examples = {}
    for verdict, label in [
        (InspectionVerdict.PASS, "通过"),
        (InspectionVerdict.FAIL, "不通过"),
        (InspectionVerdict.REVIEW, "待复核"),
    ]:
        if verdict not in verdicts:
            continue
        examples[verdict.value] = {
            "summary": label,
            "value": {
                "code": int(ErrorCode.SUCCESS),
                "message": ERROR_CODE_MESSAGES[ErrorCode.SUCCESS],
                "result": {
                    "detailList": [], "status": verdict.legacy_status,
                    "verdict": verdict.value, "error_msg": "",
                    "message": label, "vis_image": "",
                },
            },
        }
    for code in ErrorCode:
        if code is ErrorCode.SUCCESS:
            continue
        message = ERROR_CODE_MESSAGES[code]
        examples[code.name] = {
            "summary": message,
            "value": {
                "code": int(code), "message": message,
                "result": {
                    "detailList": [], "status": "false", "verdict": None,
                    "error_msg": message, "message": message,
                },
            },
        }
    return examples


def _apply_request_document(
    operation: dict, components: dict, model: type[BaseModel], owner,
) -> None:
    multipart = operation.get("requestBody", {}).get("content", {}).get("multipart/form-data")
    if not multipart:
        return
    body = components[multipart["schema"]["$ref"].rsplit("/", 1)[-1]]
    field = body.get("properties", {}).get("json_data")
    if field is None:
        return
    ref = _register_request_schema(model, components)
    field["description"] = "JSON 字符串，字段结构和业务约束见接口说明；作为 multipart/form-data 普通文本字段提交"
    field["x-json-schema"] = {"$ref": ref}
    example = owner.request_document_example
    if example is not None:
        field["example"] = _compact_json_example(example)
    sections = [
        operation.get("description", ""),
        "### 请求参数\n\n使用 `multipart/form-data`：`file` 为必填图片，`json_data` 为必填 JSON 字符串。",
        owner.request_document_notes,
        _field_table({"$ref": ref}, components),
    ]
    if example is not None:
        sections.append(
            "### json_data 示例\n\n```json\n"
            + json.dumps(example, ensure_ascii=False, indent=2) + "\n```"
        )
    verdicts = owner.response_document_verdicts
    labels = {InspectionVerdict.PASS: "通过", InspectionVerdict.FAIL: "不通过",
              InspectionVerdict.REVIEW: "待复核，须独立处理"}
    verdict_description = "、".join(f"`{item.value}`（{labels[item]}）" for item in verdicts)
    sections.append(
        "### 响应说明\n\n检测完成及应用异常处理均返回 HTTP 200。`code=1` 表示推理正常完成，"
        "不代表产品通过；其他业务错误码见响应示例。"
        f"`result.verdict` 为 {verdict_description}，执行错误时为 null。"
        "兼容字段 `status` 是字符串：PASS 对应 `\"true\"`，其余检测结论对应 `\"false\"`。"
        "`detailList` 为检测详情，`vis_image` 为可选的可视化图片。"
        "以下为公共响应结构示例，具体检测详情由场景决定。"
    )
    if InspectionVerdict.REVIEW in verdicts:
        sections.append("不能仅凭 status 将 REVIEW 归为 FAIL。")
    sections.append(owner.response_document_notes)
    operation["description"] = "\n\n".join(section for section in sections if section)
    operation["responses"]["200"]["content"]["application/json"]["examples"] = _response_examples(verdicts)
