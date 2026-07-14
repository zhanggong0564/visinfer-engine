"""Benchmark panel-label recognition with Paddle and ONNX backends."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT / "zhongyang" / "images"
PADDLE_MODEL = (
    ROOT
    / "weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr"
)
ONNX_MODEL = (
    ROOT / "weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr.onnx"
)
ONNX_METADATA = (
    ROOT
    / "weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr/inference.yml"
)


def percentile(values: list[float]) -> tuple[float, float]:
    p50, p95 = np.percentile(np.asarray(values, dtype=np.float64), [50, 95])
    return float(p50), float(p95)


def load_images(image_dir: Path, limit: int) -> list[np.ndarray]:
    paths = sorted(image_dir.glob("*.png"))[:limit]
    images = [cv2.imread(str(path)) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        raise FileNotFoundError(f"未找到可读取的 PNG 图片: {image_dir}")
    return images


def prediction_batches(
    images: list[np.ndarray], call_mode: str
) -> list[list[np.ndarray]]:
    if call_mode == "single":
        return [[image] for image in images]
    if call_mode == "batch":
        return [images]
    raise ValueError(f"unsupported call mode: {call_mode}")


def summarize_ort_profile(profile_path: Path) -> dict[str, dict[str, float | int]]:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    summary = defaultdict(lambda: {"node_count": 0, "duration_ms": 0.0})
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider", "unknown")
        summary[provider]["node_count"] += 1
        summary[provider]["duration_ms"] += float(event.get("dur", 0)) / 1000
    return {
        provider: {
            "node_count": data["node_count"],
            "duration_ms": round(data["duration_ms"], 3),
        }
        for provider, data in sorted(summary.items())
    }


def build_predictor(
    backend: str,
    onnx_execution_mode: str,
    enable_onnx_profiling: bool,
):
    if backend == "paddle":
        import paddle
        from paddleocr import TextRecognition

        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("当前 Paddle 不是 CUDA 版本")
        if paddle.device.cuda.device_count() < 1:
            raise RuntimeError("Paddle 未检测到 CUDA GPU")
        paddle.set_device("gpu:0")
        print(f"Paddle device: {paddle.device.get_device()}")

        model = TextRecognition(model_dir=str(PADDLE_MODEL))
        return lambda images: list(model.predict(images)), None

    import onnxruntime as ort

    from vie_plugin_panel_label.ocr_models import PanelLabelTextRecognizer
    from services.base import OnnxRuntimeRunner

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    runner = OnnxRuntimeRunner(
        str(ONNX_MODEL),
        providers=providers,
        execution_mode=onnx_execution_mode,
        enable_profiling=enable_onnx_profiling,
    )
    print(
        f"ONNX execution_mode: {onnx_execution_mode}; "
        f"providers: {list(runner.providers)}"
    )
    if "CUDAExecutionProvider" not in runner.providers:
        raise RuntimeError(
            f"ONNX 未启用 CUDA provider，实际 provider={runner.providers}; "
            f"可用 provider={ort.get_available_providers()}"
        )
    model = PanelLabelTextRecognizer(
        str(ONNX_MODEL),
        str(ONNX_METADATA),
        runner=runner,
    )
    return lambda images: model.predict(images), runner


def run(
    backend: str,
    images: list[np.ndarray],
    warmup: int,
    iterations: int,
    onnx_execution_mode: str,
    enable_onnx_profiling: bool,
    call_mode: str,
) -> None:
    predict, runner = build_predictor(
        backend, onnx_execution_mode, enable_onnx_profiling
    )
    batches = prediction_batches(images, call_mode)
    predict(batches[0])
    for _ in range(warmup):
        for batch in batches:
            predict(batch)

    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        for batch in batches:
            predict(batch)
        samples.append((time.perf_counter() - started) * 1000 / len(images))

    p50, p95 = percentile(samples)
    print(
        f"{backend}: call_mode={call_mode} images={len(images)} iterations={iterations} "
        f"p50_ms={p50:.3f} p95_ms={p95:.3f} "
        f"mean_ms={np.mean(samples):.3f}"
    )
    if runner is not None and enable_onnx_profiling:
        profile_path = Path(runner.end_profiling())
        print(f"ORT profile: {profile_path}")
        for provider, data in summarize_ort_profile(profile_path).items():
            print(
                f"ORT nodes provider={provider} "
                f"count={data['node_count']} duration_ms={data['duration_ms']:.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("paddle", "onnx"), required=True)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--call-mode",
        choices=("single", "batch"),
        default="single",
        help="single measures per-crop calls; batch reproduces OCR pipeline calls",
    )
    parser.add_argument(
        "--onnx-execution-mode",
        choices=("parallel", "sequential"),
        default="parallel",
        help="ONNX Runtime graph execution mode; default matches production",
    )
    parser.add_argument(
        "--profile-onnx",
        action="store_true",
        help="write and summarize an ONNX Runtime node-level profile",
    )
    args = parser.parse_args()
    if args.limit < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("limit 必须 >=1，warmup 必须 >=0，iterations 必须 >=1")

    images = load_images(args.image_dir, args.limit)
    run(
        args.backend,
        images,
        args.warmup,
        args.iterations,
        args.onnx_execution_mode,
        args.profile_onnx,
        args.call_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
