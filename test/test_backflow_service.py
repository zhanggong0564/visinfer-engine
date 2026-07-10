import json
from pathlib import Path

import pytest

from routers.backflow_service import BackflowService, BackflowTarget


@pytest.fixture
def service(tmp_path):
    def resolve_target(original_filename, fallback_product_type=None):
        return BackflowTarget(
            scene_dir="indicator_light",
            model_dir=fallback_product_type or "_unknown_model",
            save_stem=Path(original_filename).stem,
        )

    return BackflowService(
        detector_type="indicator_light",
        data_dir=str(tmp_path),
        target_resolver=resolve_target,
    )


def test_resolve_paths_keeps_output_under_data_dir(service, tmp_path):
    paths = service.resolve_paths(
        "../../sample.jpg",
        "2026-07-10T10:00:00.000",
        "../TK2",
        "pending",
        ".jpg",
    )
    assert Path(paths["image_path"]).is_relative_to(tmp_path)
    assert ".." not in Path(paths["image_path"]).parts


def test_persist_record_moves_pending_image_and_writes_json(service):
    pending = service.resolve_paths(
        "sample.jpg", "2026-07-10T10:00:00.000", "TK2", "pending", ".jpg"
    )
    Path(pending["image_path"]).parent.mkdir(parents=True)
    Path(pending["image_path"]).write_bytes(b"original-jpeg")

    service.persist_record(
        original_filename="sample.jpg",
        raw_json='{"product_type":"TK2"}',
        result_dict={"status": "true", "detailList": []},
        latency_ms=12.5,
        received_at="2026-07-10T10:00:00.000",
        fallback_product_type="TK2",
        image_extension=".jpg",
    )

    final = service.resolve_paths(
        "sample.jpg", "2026-07-10T10:00:00.000", "TK2", "ok", ".jpg"
    )
    assert Path(final["image_path"]).read_bytes() == b"original-jpeg"
    record = json.loads(Path(final["record_path"]).read_text(encoding="utf-8"))
    assert record["verdict"] == "ok"
    assert record["request_params"] == {"product_type": "TK2"}


def test_image_failure_does_not_prevent_record_write(service, monkeypatch):
    def fail_write(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr("routers.backflow_service.write_bytes_atomically", fail_write)
    service.persist_record(
        original_filename="sample.jpg",
        raw_json="{}",
        result_dict={"status": "false"},
        latency_ms=1.0,
        received_at="2026-07-10T10:00:00.000",
        fallback_product_type="TK2",
        raw_image_bytes=b"fallback",
        image_extension=".jpg",
    )
    paths = service.resolve_paths(
        "sample.jpg", "2026-07-10T10:00:00.000", "TK2", "ng", ".jpg"
    )
    assert Path(paths["record_path"]).exists()


@pytest.mark.parametrize(
    ("status", "expected"),
    [(True, "ok"), ("true", "ok"), (" TRUE ", "ok"), (False, "ng"), (None, "ng")],
)
def test_classify_result_only_accepts_explicit_true(status, expected):
    assert BackflowService.classify_result({"status": status}) == expected


def test_error_record_keeps_pending_image_and_error_verdict(service):
    pending = service.resolve_paths(
        "sample.jpg", "2026-07-10T10:00:00.000", "TK2", "pending", ".jpg"
    )
    Path(pending["image_path"]).parent.mkdir(parents=True)
    Path(pending["image_path"]).write_bytes(b"original-jpeg")

    service.persist_record(
        original_filename="sample.jpg",
        raw_json="not-json",
        result_dict={"error": "boom"},
        latency_ms=1.0,
        received_at="2026-07-10T10:00:00.000",
        fallback_product_type="TK2",
        raw_image_bytes=b"replacement",
        image_extension=".jpg",
    )

    assert Path(pending["image_path"]).read_bytes() == b"original-jpeg"
    record = json.loads(Path(pending["record_path"]).read_text(encoding="utf-8"))
    assert record["verdict"] == "error"
    assert record["request_params"] == "not-json"


def test_detected_extension_overrides_client_suffix(service):
    service.persist_record(
        original_filename="sample.jpg",
        raw_json="{}",
        result_dict={"status": False},
        latency_ms=1.0,
        received_at="2026-07-10T10:00:00.000",
        fallback_product_type="TK2",
        raw_image_bytes=b"original-png",
        image_extension=".png",
    )

    paths = service.resolve_paths(
        "sample.jpg", "2026-07-10T10:00:00.000", "TK2", "ng", ".png"
    )
    assert Path(paths["image_path"]).read_bytes() == b"original-png"
    assert not Path(paths["image_path"]).with_suffix(".jpg").exists()


def test_resolve_paths_rejects_invalid_image_extension(service):
    with pytest.raises(ValueError, match="不支持的回流图片扩展名"):
        service.resolve_paths(
            "sample.jpg", "2026-07-10T10:00:00.000", "TK2", "ng", ".php"
        )
