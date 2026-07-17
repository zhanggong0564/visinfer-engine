"""比较旧版重编码回流和原始字节原子回流的专项基准。"""

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routers.upload_persistence import (  # noqa: E402
    StagedImageWrite,
    decode_image,
    detect_image_extension,
)
from routers.visualization import render_detection_overlay  # noqa: E402


def percentile(samples, value):
    return float(np.percentile(np.asarray(samples, dtype=np.float64), value))


def make_payload():
    rng = np.random.default_rng(20260710)
    image = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("benchmark JPEG generation failed")
    return buffer.tobytes()


def run_legacy(source_path, image_path):
    start = time.perf_counter()
    payload = Path(source_path).read_bytes()
    image = decode_image(payload)
    if not cv2.imwrite(image_path, image):
        raise RuntimeError("legacy cv2.imwrite failed")
    render_detection_overlay(image, [], max_side=1280, jpeg_quality=85)
    return (time.perf_counter() - start) * 1000


def run_optimized(source_path, image_path, executor):
    start = time.perf_counter()
    payload = Path(source_path).read_bytes()
    detect_image_extension(payload)
    decode_future = executor.submit(decode_image, payload)
    stage_future = executor.submit(StagedImageWrite.write, payload, image_path)
    image = decode_future.result()
    staged = stage_future.result()
    try:
        staged.commit()
    except Exception:
        staged.discard()
        raise
    render_detection_overlay(image, [], max_side=1280, jpeg_quality=85)
    return (time.perf_counter() - start) * 1000


def collect_pairs(legacy_fn, optimized_fn, warmup, iterations):
    for _ in range(warmup):
        legacy_fn()
        optimized_fn()
    legacy, optimized = [], []
    for index in range(iterations):
        if index % 2 == 0:
            legacy.append(legacy_fn())
            optimized.append(optimized_fn())
        else:
            optimized.append(optimized_fn())
            legacy.append(legacy_fn())
    return legacy, optimized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-improvement", type=float, default=15.0)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be >=1 and warmup must be >=0")

    payload = make_payload()
    with tempfile.TemporaryDirectory(prefix="vie-hot-path-") as directory:
        source_path = os.path.join(directory, "source.jpg")
        Path(source_path).write_bytes(payload)
        with ThreadPoolExecutor(max_workers=2) as executor:
            legacy, optimized = collect_pairs(
                lambda: run_legacy(source_path, os.path.join(directory, "legacy.jpg")),
                lambda: run_optimized(source_path, os.path.join(directory, "optimized.jpg"), executor),
                args.warmup,
                args.iterations,
            )

    legacy_p95 = percentile(legacy, 95)
    optimized_p95 = percentile(optimized, 95)
    improvement = (legacy_p95 - optimized_p95) / legacy_p95 * 100.0
    print(f"legacy p50_ms={percentile(legacy, 50):.3f} p95_ms={legacy_p95:.3f}")
    print(f"optimized p50_ms={percentile(optimized, 50):.3f} p95_ms={optimized_p95:.3f} improvement_pct={improvement:.2f}")
    return 0 if improvement >= args.min_improvement else 1


if __name__ == "__main__":
    raise SystemExit(main())
