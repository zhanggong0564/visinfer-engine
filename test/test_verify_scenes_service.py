import json

import pytest
import requests

from scripts.production import verify_scenes_service as MODULE


class FakeResponse:
    def __init__(self, payload=None, *, error=None, content=b""):
        self.payload = payload
        self.error = error
        self.content = content

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, get_payloads=None, post_payloads=None):
        self.get_payloads = list(get_payloads or [])
        self.post_payloads = list(post_payloads or [])
        self.posts = []

    def get(self, url, timeout):
        return self.get_payloads.pop(0)

    def post(self, url, *, files, data, timeout):
        self.posts.append(json.loads(data["json_data"]))
        return self.post_payloads.pop(0)


def health_payload():
    return {"code": 1, "result": {"status": "healthy"}}


def ready_payload(*, cuda=True, failed=None, models=None):
    if models is None:
        models = [
            {"scenario": scene, "model": model, "providers": ["CUDAExecutionProvider"]}
            for scene, model in MODULE.EXPECTED_MODELS
        ]
    return {
        "code": 1,
        "result": {
            "status": "ready",
            "failed_scenes": failed or [],
            "runtime": {"require_cuda": cuda, "models": models},
        },
    }


def detection_payload(status="false", verdict="FAIL"):
    return {
        "code": 1,
        "result": {
            "status": status,
            "verdict": verdict,
            "error_msg": "10!=6",
            "detailList": [{"status": status, "verdict": verdict}],
            "vis_image": "ignored",
        },
    }


def request_payload():
    return {
        "modelParams": {"type": "2", "register": "true"},
        "type": "A0SW2124",
        "product": "product",
        "sn": "sn",
        "AICameraModel": [{"Id": "id", "Version": 2, "ModelFile": "http://host/a.jpg"}],
    }


def transition_requests():
    before = request_payload()
    after = request_payload()
    before["modelParams"].pop("register")
    after["modelParams"].pop("register")
    before["AICameraModel"][0].update(
        {"ModelFile": "http://host/old.jpg", "UpdateTime": "2026-01-01"}
    )
    after["AICameraModel"][0].update(
        {"ModelFile": "http://host/new.jpg", "UpdateTime": "2026-01-02"}
    )
    return before, after


def test_health_ready_and_openapi_happy_path():
    session = FakeSession(
        get_payloads=[
            FakeResponse(health_payload()),
            FakeResponse(ready_payload()),
            FakeResponse({"paths": {MODULE.INDICATOR_ROUTE: {}}}),
        ]
    )
    verifier = MODULE.ScenesVerifier("http://host/", session=session)

    assert verifier.check_health().detail == "healthy"
    assert verifier.check_readiness().detail == "2 indicator models on CUDA"
    assert verifier.check_openapi().detail == "indicator route"
    assert verifier.base_url == "http://host"


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"code": 0, "result": {"status": "healthy"}}, "code"),
        ({"code": 1, "result": {"status": "bad"}}, "healthy"),
    ],
)
def test_health_rejects_invalid_contract(payload, message):
    verifier = MODULE.ScenesVerifier("http://host", session=FakeSession([FakeResponse(payload)]))
    with pytest.raises(MODULE.VerificationError, match=message):
        verifier.check_health()


@pytest.mark.parametrize(
    "payload,message",
    [
        (ready_payload(cuda=False), "CUDA"),
        (ready_payload(failed=["indicator_light"]), "失败场景"),
        (ready_payload(models=[]), "模型集合"),
        (
            ready_payload(models=[
                {"scenario": scene, "model": model, "providers": ["CPUExecutionProvider"]}
                for scene, model in MODULE.EXPECTED_MODELS
            ]),
            "CUDA provider",
        ),
    ],
)
def test_readiness_rejects_runtime_contract_violations(payload, message):
    verifier = MODULE.ScenesVerifier("http://host", session=FakeSession([FakeResponse(payload)]))
    with pytest.raises(MODULE.VerificationError, match=message):
        verifier.check_readiness()


def test_openapi_reports_missing_routes():
    verifier = MODULE.ScenesVerifier(
        "http://host", session=FakeSession([FakeResponse({"paths": {}})])
    )
    with pytest.raises(MODULE.VerificationError, match="缺少指示灯入口"):
        verifier.check_openapi()


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}, error=requests.HTTPError("500")),
        FakeResponse(ValueError("bad json")),
        FakeResponse([]),
    ],
)
def test_get_json_wraps_transport_and_payload_errors(response):
    verifier = MODULE.ScenesVerifier("http://host", session=FakeSession([response]))
    with pytest.raises(MODULE.VerificationError):
        verifier.check_health()


def test_indicator_variants_never_send_register_true(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    session = FakeSession(post_payloads=[FakeResponse(detection_payload()) for _ in range(6)])
    verifier = MODULE.ScenesVerifier("http://host", session=session)

    result = verifier.check_indicator_register_variants(image, request_payload())

    assert "equivalent" in result.detail
    assert "register" not in session.posts[0]["modelParams"]
    assert session.posts[1]["modelParams"]["register"] == "false"
    assert session.posts[2]["modelParams"]["register"] is False
    assert all(item["modelParams"].get("register") is not True for item in session.posts)
    assert [item["modelParams"]["type"] for item in session.posts] == ["2", "2", "2", 2, 2, 2]


def test_indicator_variants_reject_different_business_result(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    session = FakeSession(
        post_payloads=[FakeResponse(detection_payload())]
        + [FakeResponse(detection_payload(status="true", verdict="PASS"))]
        + [FakeResponse(detection_payload()) for _ in range(4)]
    )
    verifier = MODULE.ScenesVerifier("http://host", session=session)
    with pytest.raises(MODULE.VerificationError, match="结果不一致"):
        verifier.check_indicator_register_variants(image, request_payload())


def test_registration_transition_checks_different_sources_and_stable_new_cache(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    before, after = transition_requests()
    session = FakeSession(
        get_payloads=[
            FakeResponse(content=b"old image"),
            FakeResponse(content=b"new image"),
        ],
        post_payloads=[FakeResponse(detection_payload()) for _ in range(3)],
    )
    verifier = MODULE.ScenesVerifier("http://host", session=session)

    result = verifier.check_registration_transition(
        image, before, after, {"before_detail_count": 1, "after_detail_count": 1}
    )

    assert "cache hit" in result.detail
    assert "register" not in session.posts[0]["modelParams"]
    assert "register" not in session.posts[1]["modelParams"]
    assert session.posts[2]["modelParams"]["register"] is False


@pytest.mark.parametrize(
    "change,message",
    [
        ("material", "物料号"),
        ("id", "注册 ID"),
        ("version", "Version"),
        ("source", "注册来源"),
    ],
)
def test_registration_transition_rejects_incompatible_identity(tmp_path, change, message):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    before, after = transition_requests()
    if change == "material":
        after["type"] = "OTHER"
    elif change == "id":
        after["AICameraModel"][0]["Id"] = "other"
    elif change == "version":
        after["modelParams"]["type"] = "3"
        after["AICameraModel"][0]["Version"] = 3
    else:
        after["AICameraModel"][0]["ModelFile"] = before["AICameraModel"][0]["ModelFile"]
        after["AICameraModel"][0]["UpdateTime"] = before["AICameraModel"][0]["UpdateTime"]
    with pytest.raises(MODULE.VerificationError, match=message):
        MODULE.ScenesVerifier("http://host", session=FakeSession()).check_registration_transition(
            image, before, after
        )


def test_registration_transition_rejects_same_image_content(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    before, after = transition_requests()
    session = FakeSession(
        get_payloads=[FakeResponse(content=b"same"), FakeResponse(content=b"same")]
    )
    with pytest.raises(MODULE.VerificationError, match="内容相同"):
        MODULE.ScenesVerifier("http://host", session=session).check_registration_transition(
            image, before, after
        )


def test_registration_transition_rejects_unstable_new_result(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    before, after = transition_requests()
    session = FakeSession(
        get_payloads=[FakeResponse(content=b"old"), FakeResponse(content=b"new")],
        post_payloads=[
            FakeResponse(detection_payload()),
            FakeResponse(detection_payload()),
            FakeResponse(detection_payload(status="true", verdict="PASS")),
        ],
    )
    with pytest.raises(MODULE.VerificationError, match="结果不稳定"):
        MODULE.ScenesVerifier("http://host", session=session).check_registration_transition(
            image, before, after
        )


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"code": 1001, "message": "missing"}, "检测失败"),
        ({"code": 1, "result": []}, "result"),
        ({"code": 1, "result": {"status": True, "verdict": "PASS", "detailList": []}}, "status"),
        ({"code": 1, "result": {"status": "true", "verdict": "OK", "detailList": []}}, "verdict"),
        ({"code": 1, "result": {"status": "true", "verdict": "PASS", "detailList": {}}}, "detailList"),
    ],
)
def test_detection_contract_rejects_malformed_response(payload, message):
    with pytest.raises(MODULE.VerificationError, match=message):
        MODULE.ScenesVerifier._assert_detection_contract(payload)


@pytest.mark.parametrize(
    "payload,message",
    [
        ({}, "modelParams"),
        ({"modelParams": {}}, "type"),
        ({"modelParams": {"type": "2"}}, "AICameraModel"),
    ],
)
def test_indicator_case_validation_happens_before_network(tmp_path, payload, message):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    with pytest.raises(MODULE.VerificationError, match=message):
        MODULE.ScenesVerifier("http://host", session=FakeSession()).check_indicator_register_variants(
            image, payload
        )


def test_load_case_resolves_relative_image(tmp_path):
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"image": "sample.jpg", "request": request_payload()}))
    image, request = MODULE.load_case(case)
    assert image == tmp_path / "sample.jpg"
    assert request == request_payload()


@pytest.mark.parametrize("payload", [{}, {"image": "x"}, {"image": 1, "request": {}}])
def test_load_case_rejects_invalid_shape(tmp_path, payload):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(payload))
    with pytest.raises(MODULE.VerificationError):
        MODULE.load_case(case)


def test_load_backflow_record_extracts_request_params(tmp_path):
    record = tmp_path / "record.json"
    image = tmp_path / "image.jpg"
    record.write_text(json.dumps({"request_params": request_payload()}))
    assert MODULE.load_backflow_record(record, image) == (image, request_payload())


@pytest.mark.parametrize("content", ["not json", "{}", "[]"])
def test_load_backflow_record_rejects_invalid_content(tmp_path, content):
    record = tmp_path / "record.json"
    record.write_text(content)
    with pytest.raises(MODULE.VerificationError):
        MODULE.load_backflow_record(record, tmp_path / "image.jpg")


def test_load_transition_case_resolves_relative_image(tmp_path):
    before, after = transition_requests()
    case = tmp_path / "transition.json"
    case.write_text(json.dumps({
        "image": "sample.jpg",
        "before_request": before,
        "after_request": after,
        "expected": {"after_detail_count": 1},
    }))
    assert MODULE.load_transition_case(case) == (
        tmp_path / "sample.jpg", before, after, {"after_detail_count": 1}
    )


def test_override_selected_model_url_only_changes_matching_version():
    request = request_payload()
    request["AICameraModel"].insert(0, {
        "Id": "other", "Version": 1, "ModelFile": "http://host/v1.jpg"
    })
    updated = MODULE.override_selected_model_url(request, "http://localhost/new.jpg")
    assert updated["AICameraModel"][0]["ModelFile"] == "http://host/v1.jpg"
    assert updated["AICameraModel"][1]["ModelFile"] == "http://localhost/new.jpg"
    assert request["AICameraModel"][1]["ModelFile"] == "http://host/a.jpg"


def test_main_reports_pass_and_failure(monkeypatch, capsys):
    monkeypatch.setattr(MODULE.ScenesVerifier, "check_health", lambda self: MODULE.CheckResult("h", 1, "ok"))
    monkeypatch.setattr(MODULE.ScenesVerifier, "check_readiness", lambda self: MODULE.CheckResult("r", 1, "ok"))
    monkeypatch.setattr(MODULE.ScenesVerifier, "check_openapi", lambda self: MODULE.CheckResult("o", 1, "ok"))
    assert MODULE.main(["--base-url", "http://host"]) == 0
    assert capsys.readouterr().out.count("PASS") == 3

    def fail(self):
        raise MODULE.VerificationError("boom")

    monkeypatch.setattr(MODULE.ScenesVerifier, "check_health", fail)
    assert MODULE.main(["--base-url", "http://host"]) == 1
    assert "FAIL: boom" in capsys.readouterr().err


def test_main_rejects_incomplete_or_conflicting_case_options(monkeypatch, capsys):
    monkeypatch.setattr(MODULE.ScenesVerifier, "check_health", lambda self: MODULE.CheckResult("h", 1, "ok"))
    monkeypatch.setattr(MODULE.ScenesVerifier, "check_readiness", lambda self: MODULE.CheckResult("r", 1, "ok"))
    monkeypatch.setattr(MODULE.ScenesVerifier, "check_openapi", lambda self: MODULE.CheckResult("o", 1, "ok"))

    assert MODULE.main(["--base-url", "http://host", "--record", "record.json"]) == 1
    assert "必须同时提供" in capsys.readouterr().err
    assert MODULE.main([
        "--base-url", "http://host", "--case", "case.json",
        "--record", "record.json", "--image", "image.jpg",
    ]) == 1
    assert "只能使用一个" in capsys.readouterr().err
    assert MODULE.main([
        "--base-url", "http://host", "--transition-case", "transition.json"
    ]) == 1
    assert "显式确认" in capsys.readouterr().err
    assert MODULE.main([
        "--base-url", "http://host", "--before-record", "before.json"
    ]) == 1
    assert "必须同时提供" in capsys.readouterr().err
