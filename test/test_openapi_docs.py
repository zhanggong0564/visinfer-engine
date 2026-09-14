"""Swagger 参数与响应契约；无需模型加载或远程参考图。"""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, create_model

from routers.base_router import BaseRouter
from routers.response_builder import ResponseBuilder
from schemas.common import CommonResponse
from utils.openapi_docs import configure_openapi_docs


SCENES = [
    ("dc_fuse", "dc_fuse_router", "/dcfuse_detect"),
    ("indicator_light", "indicator_router", "/indicator_light_detect"),
    ("lap_surf", "lap_surf_router", "/lap_surf_detect"),
    ("line_squeeze", "line_squeeze_router", "/line_squeeze_recognition"),
    ("plate_screw", "plate_screw_router", "/plate_screw_detect"),
]


ALL_DOCUMENTED_SCENES = SCENES + [
    ("panel_label", "panel_label_router", "/panel_label_detect"),
    ("mvs", "mvs_router", "/mvs_inspect"),
]


def _resolve(schema, node):
    if "$ref" in node:
        assert node["$ref"].startswith("#/components/schemas/")
        return schema["components"]["schemas"][node["$ref"].rsplit("/", 1)[-1]]
    return node


def _body(schema, path):
    operation = schema["paths"][path]["post"]
    content = operation["requestBody"]["content"]
    assert set(content) == {"multipart/form-data"}
    return _resolve(schema, content["multipart/form-data"]["schema"])


@pytest.fixture
def scenes_app():
    app = FastAPI()
    routers = {}
    for scene, name, path in ALL_DOCUMENTED_SCENES:
        module = pytest.importorskip(f"vie_plugin_{scene}.plugin", reason="需安装对应 scenes 插件")
        router = getattr(module, name)
        app.include_router(router.get_router(), prefix="/api/v1", tags=[router.tag])
        routers["/api/v1" + path] = router
    configure_openapi_docs(app)
    return app, routers


def test_upload_media_type_keeps_swagger_binary_format(scenes_app, monkeypatch):
    import utils.openapi_docs as docs

    original = docs.get_openapi

    def media_type_schema(**kwargs):
        schema = original(**kwargs)
        for body in schema["components"]["schemas"].values():
            field = body.get("properties", {}).get("file")
            if field is not None:
                field.pop("format", None)
                field["contentMediaType"] = "application/octet-stream"
        return schema

    monkeypatch.setattr(docs, "get_openapi", media_type_schema)
    app, routers = scenes_app
    schema = app.openapi()
    for path in routers:
        field = _body(schema, path)["properties"]["file"]
        assert field["contentMediaType"] == "application/octet-stream"
        assert field["format"] == "binary"


def test_nested_router_documents_use_effective_paths_and_visibility(scenes_app):
    _, routers = scenes_app
    owner = routers["/api/v1/lap_surf_detect"]
    nested = APIRouter()
    nested.include_router(owner.get_router(), prefix="/scene")
    app = FastAPI()
    app.include_router(nested, prefix="/visible")
    app.include_router(nested, prefix="/hidden", include_in_schema=False)
    configure_openapi_docs(app)

    schema = app.openapi()
    path = "/visible/scene/lap_surf_detect"
    assert set(schema["paths"]) == {path}
    assert json.loads(_body(schema, path)["properties"]["json_data"]["example"]) == {}
    assert "HTTP 200" in schema["paths"][path]["post"]["description"]


def test_all_scene_documents_match_parsers_and_form_protocol(scenes_app):
    app, routers = scenes_app
    schema = app.openapi()
    for path, router in routers.items():
        body = _body(schema, path)
        assert set(body["required"]) == {"file", "json_data"}
        assert body["properties"]["file"]["format"] == "binary"
        field = body["properties"]["json_data"]
        assert field["type"] == "string"
        payload = json.loads(field["example"])
        parsed = router.request_schema(payload)
        assert isinstance(parsed, router.request_document_model)
        assert payload == router.request_document_example
        document_model = _resolve(schema, field["x-json-schema"])
        assert set(document_model.get("properties", {})) == set(
            router.request_document_model.model_json_schema(by_alias=True).get("properties", {})
        )
        operation = schema["paths"][path]["post"]
        assert "HTTP 200" in operation["description"]
        assert "422" not in operation["responses"]
        assert "```json" in operation["description"]
        assert router.request_document_notes in operation["description"]
    assert app.openapi() is schema

    def check_refs(value):
        if isinstance(value, dict):
            if "$ref" in value:
                _resolve(schema, value)
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(schema)


def test_nested_models_aliases_defaults_and_names_are_scene_specific(scenes_app):
    schema = scenes_app[0].openapi()
    properties_by_scene = {}
    refs = []
    for scene, _, path in SCENES:
        field = _body(schema, "/api/v1" + path)["properties"]["json_data"]
        request_model = _resolve(schema, field["x-json-schema"])
        params = request_model.get("properties", {}).get("modelParams")
        if params is None:
            continue
        refs.append(params["$ref"])
        properties_by_scene[scene] = _resolve(schema, params)["properties"]
    assert len(set(refs)) == 3
    assert "register" in properties_by_scene["indicator_light"]
    assert "register_mode" not in properties_by_scene["indicator_light"]
    assert "product_model" not in properties_by_scene["indicator_light"]
    assert "type" not in properties_by_scene["dc_fuse"]
    assert properties_by_scene["dc_fuse"]["product_model"]["default"] is None
    assert properties_by_scene["dc_fuse"]["guide_line"]["default"] == []
    description = schema["paths"]["/api/v1/indicator_light_detect"]["post"]["description"]
    assert "`AICameraModel[].ModelFile`" in description
    assert "`modelParams.register`" in description
    line = schema["paths"]["/api/v1/line_squeeze_recognition"]["post"]
    assert line["tags"] == ["线序检测"]
    assert line["summary"] == "线序检测接口"
    assert "线路压缩" not in json.dumps(line, ensure_ascii=False)


def test_response_examples_match_actual_builder_and_error_handler(scenes_app):
    from app import _build_error_response
    from schemas.error_codes import ErrorCode

    schema = scenes_app[0].openapi()
    builder = ResponseBuilder(False, 1280, 85)
    for path in scenes_app[1]:
        response = schema["paths"][path]["post"]["responses"]["200"]
        assert response["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/CommonResponse"
        }
        examples = response["content"]["application/json"]["examples"]
        expected_verdicts = {item.value for item in scenes_app[1][path].response_document_verdicts}
        assert set(examples) & {"PASS", "FAIL", "REVIEW"} == expected_verdicts
        assert {"INVALID_PARAMS", "INTERNAL_ERROR"} <= examples.keys()
        for key, example in examples.items():
            value = example["value"]
            CommonResponse.model_validate(value)
            if value["code"] == int(ErrorCode.SUCCESS):
                actual = asyncio.run(builder.build(
                    np.zeros((2, 2, 3), np.uint8), copy.deepcopy(value["result"]), SimpleNamespace(),
                )).model_dump(mode="json")
                assert actual == value
            else:
                assert _build_error_response(ErrorCode[key], value["result"]["error_msg"]) == value


def test_scenes_response_documents_explain_details_and_real_verdicts(scenes_app):
    schema = scenes_app[0].openapi()
    for _, _, endpoint in SCENES:
        op = schema["paths"]["/api/v1" + endpoint]["post"]
        examples = op["responses"]["200"]["content"]["application/json"]["examples"]
        assert "REVIEW" not in examples
        for verdict in ("PASS", "FAIL"):
            assert examples[verdict]["value"]["result"]["detailList"]
        for field in ("result.detailList[].coordinate", "result.detailList[].accuracy", "result.verdict", "result.vis_image"):
            assert field in op["description"]
        assert "四边形八个数" in op["description"]
        assert "场景明细与判定规则" in op["description"]


def test_swagger_and_form_submission_without_inference(scenes_app, monkeypatch):
    app, routers = scenes_app
    schema = app.openapi()
    with TestClient(app) as client:
        docs = client.get("/docs")
        assert docs.status_code == 200
        assert "/openapi.json" in docs.text
        assert client.get("/openapi.json").json() == schema
        for path, router in routers.items():
            operation = schema["paths"][path]["post"]
            response = operation["responses"]["200"]["content"]["application/json"]["examples"]["PASS"]["value"]
            process = AsyncMock(return_value=response)
            monkeypatch.setattr(router, "_process_detect_request", process)
            example = _body(schema, path)["properties"]["json_data"]["example"]
            result = client.post(path, files={"file": ("sample.png", b"mock-image", "image/png")}, data={"json_data": example})
            assert result.status_code == 200
            assert result.json() == response
            assert process.await_args.args[2] == example


def test_undeclared_router_remains_compatible():
    router = BaseRouter("legacy", "/legacy", "旧接口", "旧说明", "legacy")
    app = FastAPI()
    app.include_router(router.get_router())
    configure_openapi_docs(app)
    schema = app.openapi()
    assert schema["paths"]["/legacy"]["post"]["description"] == "旧说明"
    assert "example" not in _body(schema, "/legacy")["properties"]["json_data"]


def test_application_serves_local_swagger_assets_and_scene_documents():
    from app import app

    # 不进入 lifespan，不预加载真实模型。
    client = TestClient(app)
    try:
        docs = client.get("/docs")
        assert docs.status_code == 200
        for asset in ["swagger-ui-bundle.js", "swagger-ui.css", "favicon-32x32.png"]:
            path = "/static/swagger-ui/" + asset
            assert path in docs.text
            assert client.get(path).status_code == 200
        schema = client.get("/openapi.json").json()
        assert schema == app.openapi()
        for scene, _, path in ALL_DOCUMENTED_SCENES:
            pytest.importorskip(f"vie_plugin_{scene}", reason="需安装对应 scenes 插件")
            assert "example" in _body(schema, "/api/v1" + path)["properties"]["json_data"]
    finally:
        client.close()


def test_generic_documentation_handles_aliases_recursion_and_safe_defaults():
    class Node(BaseModel):
        name: str = Field(alias="Name")
        children: list["Node"] = Field(default_factory=list)

    Node.model_rebuild()
    # 两个不同模块中的同名嵌套模型不能互相覆盖。
    first = create_model("ModelParams", __module__="first", item=(int, ...))
    second = create_model("ModelParams", __module__="second", item=(str, ...))
    models = [create_model("FirstRequest", params=(first, ...)),
              create_model("SecondRequest", params=(second, ...)), Node]
    app = FastAPI()
    for index, model in enumerate(models):
        router = BaseRouter(str(index), f"/route{index}", "请求", "说明", str(index))
        router.request_document_model = model
        router.request_document_example = None
        app.include_router(router.get_router())
    configure_openapi_docs(app)
    schema = app.openapi()
    types = []
    for index in range(2):
        field = _body(schema, f"/route{index}")["properties"]["json_data"]
        request = _resolve(schema, field["x-json-schema"])
        params = _resolve(schema, request["properties"]["params"])
        types.append(params["properties"]["item"]["type"])
    assert types == ["integer", "string"]
    assert "`Name`" in schema["paths"]["/route2"]["post"]["description"]
