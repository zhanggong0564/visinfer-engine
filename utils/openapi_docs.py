"""OpenAPI 文档定制逻辑。"""

import json

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def _compact_json_example(data: dict) -> str:
    """Swagger 表单字段示例：json_data 是字符串，因此示例也必须是 JSON 字符串。"""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


JSON_DATA_EXAMPLES_BY_PATH = {
    "/api/v1/dcfuse_detect": _compact_json_example({
        "product": "直流熔丝",
        "type": "物料号",
        "modelParams": {
            "product_model": "六路无熔丝盒无磁环",
            "guide_line": [],
            "example_images": [],
        },
        "AICameraModel": [],
    }),
    "/api/v1/line_squeeze_recognition": _compact_json_example({
        "product": "线路压缩",
        "type": "物料号",
        "modelParams": {
            "product_model": "五路有熔丝盒有磁环",
        },
    }),
    "/api/v1/panel_label_detect": _compact_json_example({
        "product": "逆变器组件_SG1100UD-V3039_S",
        "type": "A0ST6329",
        "sn": "A2670608545",
        "modelParams": {
            "guide_line": [{
                "FileName": "5、直流侧开关S1S2.png",
                "FilePath": "http://10.172.2.32:9986/AIModelFile/202606/20260622140657_d35pud55.png",
            }],
            "example_images": [{
                "FileName": "屏幕截图 2026-04-22 145231.png",
                "FilePath": "http://10.172.2.32:9986/AIModelFile/202606/20260622140214_edvlny2u.png",
            }],
            "product_type": "S1S2",
            "rule": "front",
            "line_order": "S2-14,S2-13,S1-13,S1-14",
            "guideline_coordinates": "0.154,0.114666666666667,0.771,0.76",
        },
        "AICameraModel": [{
            "Id": "d4315be0f3c645ca86ca0f4a793b9e95",
            "SN": "A2662715398",
            "ProductName": "A0ST6329",
            "Version": 4,
            "AIProductTypeName": "集中式检验组",
            "AIProductTypeValue": "集中式检验组",
            "ModelFile": None,
            "Remark": None,
            "CreateBy": None,
            "CreateTime": "2026-07-01T10:15:46",
            "UpdateBy": None,
            "UpdateTime": "2026-07-01T10:15:46",
            "AIParameterName": "产品类型",
            "AIParameterValue": "五路有熔丝盒无磁环",
            "DictionaryCode": None,
        }],
    }),
    "/api/v1/indicator_light_detect": _compact_json_example({
        "product": "指示灯",
        "type": "物料号",
        "modelParams": {
            "type": 1,
            "register": False,
            "guide_line": [],
            "example_images": [],
        },
        "AICameraModel": [],
    }),
}


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
        for path, path_item in schema.get("paths", {}).items():
            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue
                operation.get("responses", {}).pop("422", None)
                _apply_json_data_example(path, operation, components)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi


def _apply_json_data_example(path: str, operation: dict, components: dict) -> None:
    example = JSON_DATA_EXAMPLES_BY_PATH.get(path)
    request_body = operation.get("requestBody", {})
    multipart = request_body.get("content", {}).get("multipart/form-data")
    body_ref = (multipart or {}).get("schema", {}).get("$ref")
    if not (example and body_ref):
        return
    body_schema = components.get(body_ref.rsplit("/", 1)[-1], {})
    json_data_schema = body_schema.get("properties", {}).get("json_data")
    if json_data_schema is None:
        return
    json_data_schema["description"] = (
        "JSON 字符串，结构见示例；提交时作为 multipart/form-data 的普通文本字段传入"
    )
    json_data_schema["example"] = example
