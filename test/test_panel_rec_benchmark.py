"""Regression checks for the panel-label OCR benchmark setup."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/benchmark/panel_rec_benchmark.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "panel_rec_benchmark", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_compares_matching_v6_recognition_weights():
    benchmark = _load_benchmark_module()

    assert "v6" in benchmark.PADDLE_MODEL.name
    assert benchmark.PADDLE_MODEL.name == benchmark.ONNX_MODEL.stem


def test_profile_summary_groups_node_durations_by_provider(tmp_path):
    benchmark = _load_benchmark_module()
    profile = tmp_path / "ort_profile.json"
    profile.write_text(
        json.dumps(
            [
                {"cat": "Node", "dur": 1500, "args": {"provider": "CUDAExecutionProvider"}},
                {"cat": "Node", "dur": 500, "args": {"provider": "CPUExecutionProvider"}},
                {"cat": "Session", "dur": 9000, "args": {}},
            ]
        ),
        encoding="utf-8",
    )

    assert benchmark.summarize_ort_profile(profile) == {
        "CPUExecutionProvider": {"node_count": 1, "duration_ms": 0.5},
        "CUDAExecutionProvider": {"node_count": 1, "duration_ms": 1.5},
    }


def test_batch_call_mode_keeps_all_images_in_one_predict_call():
    benchmark = _load_benchmark_module()
    images = [object(), object(), object()]

    assert benchmark.prediction_batches(images, "single") == [
        [images[0]],
        [images[1]],
        [images[2]],
    ]
    assert benchmark.prediction_batches(images, "batch") == [images]
