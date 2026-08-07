import base64
import json
from argparse import Namespace

import pytest
import requests

from scripts.production import evaluate_indicator_dataset as MODULE


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.request = None

    def post(self, endpoint, **kwargs):
        self.request = (endpoint, kwargs)
        return self.response


class SequenceSession:
    def __init__(self, responses):
        self.responses = iter(responses)

    def post(self, endpoint, **kwargs):
        return next(self.responses)


def response_payload(verdict="PASS", status="true"):
    return {
        "code": 1,
        "result": {
            "verdict": verdict,
            "status": status,
            "detailList": [{"status": status, "verdict": verdict}],
            "error_msg": "" if verdict == "PASS" else "mismatch",
            "vis_image": "data:image/jpeg;base64," + base64.b64encode(b"jpg").decode(),
        },
    }


def test_discover_cases_groups_registered_and_current_images(tmp_path):
    group = tmp_path / "A0SW2120-2"
    (group / "registered").mkdir(parents=True)
    (group / "current").mkdir()
    (group / "registered" / "r.jpg").write_bytes(b"r")
    (group / "current" / "b.jpg").write_bytes(b"b")
    (group / "current" / "a.jpg").write_bytes(b"a")
    (tmp_path / "invalid-name").mkdir()

    assert MODULE.discover_cases(tmp_path) == [
        ("A0SW2120", "2", group / "registered" / "r.jpg", [
            group / "current" / "a.jpg", group / "current" / "b.jpg"
        ])
    ]


def test_discover_cases_requires_exactly_one_registration_image(tmp_path):
    group = tmp_path / "A0SW2120-2"
    (group / "registered").mkdir(parents=True)
    (group / "current").mkdir()
    with pytest.raises(ValueError, match="注册图数量"):
        MODULE.discover_cases(tmp_path)


def test_build_request_omits_register_and_uses_version():
    request = MODULE.build_request("A0SW2120", "2", "http://host/r.jpg")
    assert request["modelParams"] == {"type": "2"}
    assert request["AICameraModel"][0]["Version"] == 2
    assert request["AICameraModel"][0]["ModelFile"] == "http://host/r.jpg"


def test_evaluate_image_returns_contract_result(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    session = FakeSession(FakeResponse(response_payload()))
    item, payload = MODULE.evaluate_image(
        session, "http://host/detect", image, {"modelParams": {}}, 5
    )
    assert item.verdict == "PASS"
    assert item.detail_count == 1
    assert payload["code"] == 1
    assert session.request[0] == "http://host/detect"


@pytest.mark.parametrize(
    "response,error",
    [
        (FakeResponse({}, requests.HTTPError("500")), "500"),
        (FakeResponse({"code": 1001, "message": "invalid"}), "invalid"),
        (FakeResponse({"code": 1, "result": {"status": True, "verdict": "PASS"}}), "契约"),
    ],
)
def test_evaluate_image_converts_failures_to_error(tmp_path, response, error):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    item, payload = MODULE.evaluate_image(
        FakeSession(response), "http://host/detect", image, {}, 5
    )
    assert item.verdict == "ERROR"
    assert error in item.error
    assert payload == {}


def test_save_failure_writes_response_and_visualization(tmp_path):
    item = MODULE.ImageResult("A0SW2120-2", "image.jpg", "FAIL", "false", 1, 10, "bad")
    payload = response_payload("FAIL", "false")
    MODULE.save_failure(tmp_path, item, payload)
    failure_dir = tmp_path / "failures" / "A0SW2120-2"
    assert (failure_dir / "image.jpg").read_bytes() == b"jpg"
    saved = json.loads((failure_dir / "image.json").read_text())
    assert saved["evaluation"]["verdict"] == "FAIL"
    assert "vis_image" not in saved["response"]["result"]


def test_write_reports_summarizes_each_verdict(tmp_path):
    results = [
        MODULE.ImageResult("A-1", "p.jpg", "PASS", "true", 1, 10),
        MODULE.ImageResult("A-1", "f.jpg", "FAIL", "false", 1, 10),
        MODULE.ImageResult("B-2", "r.jpg", "REVIEW", "false", 1, 10),
        MODULE.ImageResult("B-2", "e.jpg", "ERROR", "false", 0, 10),
    ]
    MODULE.write_reports(tmp_path, results, ["C-1"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total"] == 4
    assert (summary["pass"], summary["fail"], summary["review"], summary["error"]) == (1, 1, 1, 1)
    assert summary["skipped_no_current"] == ["C-1"]
    assert (tmp_path / "results.csv").is_file()


def test_run_evaluation_covers_groups_skips_and_failures(tmp_path):
    dataset = tmp_path / "dataset"
    for name, current_count in (("A0SW2120-1", 0), ("A0SW2120-2", 2)):
        group = dataset / name
        (group / "registered").mkdir(parents=True)
        (group / "current").mkdir()
        (group / "registered" / "registered.jpg").write_bytes(b"registered")
        for index in range(current_count):
            (group / "current" / f"{index}.jpg").write_bytes(b"current")
    output = tmp_path / "output"
    args = Namespace(
        dataset_dir=dataset,
        base_url="http://service",
        registration_base_url="http://files/dataset",
        output_dir=output,
        timeout=5,
    )
    session = SequenceSession([
        FakeResponse(response_payload()),
        FakeResponse(response_payload("FAIL", "false")),
    ])
    assert MODULE.run_evaluation(args, session) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["total"] == 2
    assert summary["skipped_no_current"] == ["A0SW2120-1"]
    assert summary["by_dataset"]["A0SW2120-2"]["pass"] == 1
    assert summary["by_dataset"]["A0SW2120-2"]["fail"] == 1
    assert (output / "failures" / "A0SW2120-2" / "1.jpg").is_file()


def test_run_evaluation_returns_error_for_request_error(tmp_path):
    dataset = tmp_path / "dataset" / "A0SW2120-2"
    (dataset / "registered").mkdir(parents=True)
    (dataset / "current").mkdir()
    (dataset / "registered" / "r.jpg").write_bytes(b"r")
    (dataset / "current" / "c.jpg").write_bytes(b"c")
    args = Namespace(
        dataset_dir=dataset.parent,
        base_url="http://service",
        registration_base_url="http://files",
        output_dir=tmp_path / "output",
        timeout=5,
    )
    session = SequenceSession([FakeResponse({}, requests.HTTPError("down"))])
    assert MODULE.run_evaluation(args, session) == 1
