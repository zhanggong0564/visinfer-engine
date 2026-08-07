#!/usr/bin/env python3
"""Evaluate an indicator-light dataset against a deployed HTTP service."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class ImageResult:
    dataset: str
    image: str
    verdict: str
    status: str
    detail_count: int
    elapsed_ms: float
    error: str = ""


def discover_cases(dataset_dir: Path) -> list[tuple[str, str, Path, list[Path]]]:
    cases = []
    for group_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        try:
            material, version = group_dir.name.rsplit("-", 1)
            int(version)
        except ValueError:
            continue
        registered = sorted((group_dir / "registered").glob("*"))
        current = sorted((group_dir / "current").glob("*"))
        registered = [path for path in registered if path.is_file()]
        current = [path for path in current if path.is_file()]
        if len(registered) != 1:
            raise ValueError(f"{group_dir.name} 注册图数量必须为 1，实际 {len(registered)}")
        cases.append((material, version, registered[0], current))
    return cases


def build_request(material: str, version: str, registration_url: str) -> dict[str, Any]:
    registration_id = f"dataset-{material}-v{version}"
    return {
        "modelParams": {"type": version},
        "type": material,
        "product": f"indicator dataset {material}",
        "sn": "dataset-evaluation",
        "AICameraModel": [
            {
                "Id": registration_id,
                "SN": "dataset-evaluation",
                "ProductName": material,
                "Version": int(version),
                "AIProductTypeName": "指示灯测试集",
                "AIProductTypeValue": "指示灯测试集",
                "ModelFile": registration_url,
                "CreateTime": "2026-08-07T00:00:00",
                "UpdateTime": "2026-08-07T00:00:00",
            }
        ],
    }


def evaluate_image(
    session: requests.Session,
    endpoint: str,
    image_path: Path,
    request_data: dict[str, Any],
    timeout: float,
) -> tuple[ImageResult, dict[str, Any]]:
    started = time.perf_counter()
    try:
        with image_path.open("rb") as image_file:
            response = session.post(
                endpoint,
                files={"file": (image_path.name, image_file, "image/jpeg")},
                data={"json_data": json.dumps(request_data, ensure_ascii=False)},
                timeout=timeout,
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 1 or not isinstance(payload.get("result"), dict):
            raise ValueError(payload.get("message", "响应契约错误"))
        result = payload["result"]
        verdict = str(result.get("verdict", ""))
        status = str(result.get("status", ""))
        if verdict not in {"PASS", "FAIL", "REVIEW"} or status not in {"true", "false"}:
            raise ValueError("响应 status/verdict 契约错误")
        item = ImageResult(
            dataset="",
            image=image_path.name,
            verdict=verdict,
            status=status,
            detail_count=len(result.get("detailList", [])),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=str(result.get("error_msg") or ""),
        )
        return item, payload
    except (OSError, requests.RequestException, ValueError) as exc:
        item = ImageResult(
            dataset="",
            image=image_path.name,
            verdict="ERROR",
            status="false",
            detail_count=0,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )
        return item, {}


def save_failure(output_dir: Path, item: ImageResult, payload: dict[str, Any]) -> None:
    failure_dir = output_dir / "failures" / item.dataset
    failure_dir.mkdir(parents=True, exist_ok=True)
    result = payload.get("result", {})
    vis_image = result.pop("vis_image", "") if isinstance(result, dict) else ""
    (failure_dir / f"{Path(item.image).stem}.json").write_text(
        json.dumps({"evaluation": asdict(item), "response": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if isinstance(vis_image, str) and vis_image:
        encoded = vis_image.split(",", 1)[-1]
        try:
            (failure_dir / f"{Path(item.image).stem}.jpg").write_bytes(
                base64.b64decode(encoded, validate=True)
            )
        except (ValueError, OSError):
            pass


def write_reports(output_dir: Path, results: list[ImageResult], skipped: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=ImageResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)

    by_dataset: dict[str, dict[str, int]] = {}
    for item in results:
        counts = by_dataset.setdefault(
            item.dataset, {"total": 0, "pass": 0, "fail": 0, "review": 0, "error": 0}
        )
        counts["total"] += 1
        counts[item.verdict.lower()] += 1
    summary = {
        "total": len(results),
        "pass": sum(item.verdict == "PASS" for item in results),
        "fail": sum(item.verdict == "FAIL" for item in results),
        "review": sum(item.verdict == "REVIEW" for item in results),
        "error": sum(item.verdict == "ERROR" for item in results),
        "skipped_no_current": skipped,
        "by_dataset": by_dataset,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_evaluation(args: argparse.Namespace, session: requests.Session) -> int:
    cases = discover_cases(args.dataset_dir)
    endpoint = f"{args.base_url.rstrip('/')}/api/v1/indicator_light_detect"
    registration_base = args.registration_base_url.rstrip("/")
    results: list[ImageResult] = []
    skipped: list[str] = []
    for material, version, registered, current_images in cases:
        dataset = f"{material}-{version}"
        if not current_images:
            skipped.append(dataset)
            print(f"SKIP {dataset}: no current images", flush=True)
            continue
        registration_url = f"{registration_base}/{dataset}/registered/{registered.name}"
        request_data = build_request(material, version, registration_url)
        print(f"RUN {dataset}: {len(current_images)} images", flush=True)
        for image_path in current_images:
            item, payload = evaluate_image(
                session, endpoint, image_path, request_data, args.timeout
            )
            item.dataset = dataset
            results.append(item)
            if item.verdict != "PASS":
                save_failure(args.output_dir, item, payload)
            print(f"  {item.verdict:6s} {image_path.name} {item.elapsed_ms:.0f}ms", flush=True)
    write_reports(args.output_dir, results, skipped)
    return 1 if any(item.verdict == "ERROR" for item in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="批量评测指示灯生产接口")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--registration-base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    return run_evaluation(args, requests.Session())


if __name__ == "__main__":
    raise SystemExit(main())
