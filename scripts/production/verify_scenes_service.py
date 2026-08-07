#!/usr/bin/env python3
"""Black-box production checks focused on the indicator-light service."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


INDICATOR_ROUTE = "/api/v1/indicator_light_detect"
EXPECTED_MODELS = {
    ("indicator_light", "rec_v3.onnx"),
    ("indicator_light", "rfdetr-small_v1.1.onnx"),
}


class VerificationError(RuntimeError):
    """Raised when the deployed service violates an expected contract."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    elapsed_ms: float
    detail: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


class ScenesVerifier:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get_json(self, path: str) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            response = self.session.get(
                f"{self.base_url}{path}", timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise VerificationError(f"GET {path} 失败: {exc}") from exc
        _require(isinstance(payload, dict), f"GET {path} 未返回 JSON 对象")
        return payload, (time.perf_counter() - started) * 1000

    def check_health(self) -> CheckResult:
        payload, elapsed = self._get_json("/health")
        _require(payload.get("code") == 1, "/health code 不是 1")
        _require(
            payload.get("result", {}).get("status") == "healthy",
            "/health 未返回 healthy",
        )
        return CheckResult("health", elapsed, "healthy")

    def check_readiness(self) -> CheckResult:
        payload, elapsed = self._get_json("/health/ready")
        result = payload.get("result", {})
        _require(payload.get("code") == 1, "/health/ready code 不是 1")
        _require(result.get("status") == "ready", "服务未 ready")
        _require(not result.get("failed_scenes"), "存在预加载失败场景")

        runtime = result.get("runtime", {})
        _require(runtime.get("require_cuda") is True, "服务未强制要求 CUDA")
        models = runtime.get("models")
        _require(isinstance(models, list), "readiness 缺少模型列表")
        indicator_models = {
            (item.get("scenario"), item.get("model")) for item in models
            if item.get("scenario") == "indicator_light"
        }
        _require(
            indicator_models == EXPECTED_MODELS,
            f"指示灯模型集合不一致: actual={sorted(indicator_models)}",
        )
        for item in models:
            if item.get("scenario") != "indicator_light":
                continue
            _require(
                "CUDAExecutionProvider" in item.get("providers", []),
                f"模型 {item.get('model')} 未使用 CUDA provider",
            )
        return CheckResult("readiness", elapsed, "2 indicator models on CUDA")

    def check_openapi(self) -> CheckResult:
        payload, elapsed = self._get_json("/openapi.json")
        paths = set(payload.get("paths", {}))
        _require(INDICATOR_ROUTE in paths, "OpenAPI 缺少指示灯入口")
        return CheckResult("openapi", elapsed, "indicator route")

    def _post_indicator(
        self, image_path: Path, request_data: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            with image_path.open("rb") as image_file:
                response = self.session.post(
                    f"{self.base_url}/api/v1/indicator_light_detect",
                    files={"file": (image_path.name, image_file, "image/jpeg")},
                    data={
                        "json_data": json.dumps(
                            request_data, ensure_ascii=False, separators=(",", ":")
                        )
                    },
                    timeout=self.timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except (OSError, requests.RequestException, ValueError) as exc:
            raise VerificationError(f"指示灯请求失败: {exc}") from exc
        _require(isinstance(payload, dict), "指示灯接口未返回 JSON 对象")
        return payload, (time.perf_counter() - started) * 1000

    @staticmethod
    def _assert_detection_contract(payload: dict[str, Any]) -> None:
        _require(payload.get("code") == 1, f"检测失败: {payload}")
        result = payload.get("result")
        _require(isinstance(result, dict), "响应缺少 result 对象")
        _require(result.get("status") in {"true", "false"}, "status 契约错误")
        _require(result.get("verdict") in {"PASS", "FAIL", "REVIEW"}, "verdict 契约错误")
        _require(isinstance(result.get("detailList"), list), "detailList 契约错误")
        for item in result["detailList"]:
            _require(item.get("status") in {"true", "false"}, "明细 status 契约错误")
            _require(item.get("verdict") in {"PASS", "FAIL", "REVIEW"}, "明细 verdict 契约错误")

    @staticmethod
    def _business_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
        result = payload["result"]
        details = result.get("detailList", [])
        return (
            result.get("status"),
            result.get("verdict"),
            result.get("error_msg"),
            tuple((item.get("status"), item.get("verdict")) for item in details),
        )

    def check_indicator_register_variants(
        self, image_path: Path, request_data: dict[str, Any]
    ) -> CheckResult:
        _require(image_path.is_file(), f"测试图片不存在: {image_path}")
        model_params = request_data.get("modelParams")
        _require(isinstance(model_params, dict), "请求缺少 modelParams")
        _require(model_params.get("type") is not None, "modelParams 缺少 type")
        _require(bool(request_data.get("AICameraModel")), "请求缺少 AICameraModel")

        variants: list[tuple[str, dict[str, Any]]] = []
        canonical_type = int(model_params["type"])
        for type_name, type_value in (
            ("type string", str(canonical_type)),
            ("type integer", canonical_type),
        ):
            for register_name, register_value in (
                ("register omitted", None),
                ("register false string", "false"),
                ("register false boolean", False),
            ):
                variant = copy.deepcopy(request_data)
                variant["modelParams"]["type"] = type_value
                if register_value is None:
                    variant["modelParams"].pop("register", None)
                else:
                    variant["modelParams"]["register"] = register_value
                variants.append((f"{type_name}, {register_name}", variant))

        signatures = []
        total_elapsed = 0.0
        for name, variant in variants:
            payload, elapsed = self._post_indicator(image_path, variant)
            self._assert_detection_contract(payload)
            signatures.append((name, self._business_signature(payload)))
            total_elapsed += elapsed
        baseline = signatures[0][1]
        for name, signature in signatures[1:]:
            _require(signature == baseline, f"{name} 与省略 register 的结果不一致")
        return CheckResult(
            "indicator register variants",
            total_elapsed,
            "type string/integer and register omitted/string/boolean false equivalent",
        )

    @staticmethod
    def _selected_model(request_data: dict[str, Any]) -> dict[str, Any]:
        model_params = request_data.get("modelParams", {})
        try:
            version = int(model_params["type"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError("切换请求缺少合法的 modelParams.type") from exc
        selected = None
        for model in request_data.get("AICameraModel") or []:
            if model.get("Version") == version:
                selected = model
        _require(selected is not None, f"切换请求未找到 Version={version} 注册图")
        return selected

    def _source_sha256(self, url: str) -> str:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            content = response.content
        except requests.RequestException as exc:
            raise VerificationError(f"下载注册源失败: {exc}") from exc
        _require(bool(content), f"注册源为空: {url}")
        return hashlib.sha256(content).hexdigest()

    def check_registration_transition(
        self,
        image_path: Path,
        before_request: dict[str, Any],
        after_request: dict[str, Any],
        expected: dict[str, Any] | None = None,
    ) -> CheckResult:
        """Switch one registration identity from an old image to a new image."""
        _require(image_path.is_file(), f"测试图片不存在: {image_path}")
        before_model = self._selected_model(before_request)
        after_model = self._selected_model(after_request)
        _require(before_request.get("type") == after_request.get("type"), "切换前后物料号不同")
        _require(before_model.get("Id") == after_model.get("Id"), "切换前后注册 ID 不同")
        _require(before_model.get("Version") == after_model.get("Version"), "切换前后 Version 不同")
        before_url = before_model.get("ModelFile")
        after_url = after_model.get("ModelFile")
        _require(isinstance(before_url, str) and before_url, "旧注册图 URL 缺失")
        _require(isinstance(after_url, str) and after_url, "新注册图 URL 缺失")
        _require(
            before_url != after_url or before_model.get("UpdateTime") != after_model.get("UpdateTime"),
            "切换前后注册来源没有变化",
        )
        _require(
            self._source_sha256(before_url) != self._source_sha256(after_url),
            "切换前后注册图片内容相同",
        )

        requests_to_send = [
            ("old source", copy.deepcopy(before_request)),
            ("new source", copy.deepcopy(after_request)),
            ("new source cache hit", copy.deepcopy(after_request)),
        ]
        requests_to_send[0][1]["modelParams"].pop("register", None)
        requests_to_send[1][1]["modelParams"].pop("register", None)
        requests_to_send[2][1]["modelParams"]["register"] = False

        signatures = []
        detail_counts = []
        total_elapsed = 0.0
        for name, request_data in requests_to_send:
            _require(request_data["modelParams"].get("register") is not True, "禁止强制注册")
            payload, elapsed = self._post_indicator(image_path, request_data)
            self._assert_detection_contract(payload)
            signatures.append((name, self._business_signature(payload)))
            detail_counts.append(len(payload["result"]["detailList"]))
            total_elapsed += elapsed

        _require(signatures[1][1] == signatures[2][1], "新注册图刷新后再次请求结果不稳定")
        expected = expected or {}
        if "before_detail_count" in expected:
            _require(detail_counts[0] == expected["before_detail_count"], "旧注册图明细数量不符合预期")
        if "after_detail_count" in expected:
            _require(detail_counts[1] == expected["after_detail_count"], "新注册图明细数量不符合预期")
        if expected.get("results_differ") is True:
            _require(signatures[0][1] != signatures[1][1], "切换注册图前后业务结果没有变化")
        return CheckResult(
            "indicator registration transition",
            total_elapsed,
            "old source -> new source -> new source cache hit",
        )


def load_case(path: Path) -> tuple[Path, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"读取用例失败: {exc}") from exc
    _require(isinstance(payload, dict), "用例文件必须是 JSON 对象")
    image_value = payload.get("image")
    request_data = payload.get("request")
    _require(isinstance(image_value, str) and image_value, "用例缺少 image")
    _require(isinstance(request_data, dict), "用例缺少 request 对象")
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = path.parent / image_path
    return image_path, request_data


def load_backflow_record(
    record_path: Path, image_path: Path
) -> tuple[Path, dict[str, Any]]:
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"读取回流记录失败: {exc}") from exc
    request_data = payload.get("request_params") if isinstance(payload, dict) else None
    _require(isinstance(request_data, dict), "回流记录缺少 request_params")
    return image_path, request_data


def load_transition_case(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"读取注册图切换用例失败: {exc}") from exc
    _require(isinstance(payload, dict), "注册图切换用例必须是 JSON 对象")
    image_value = payload.get("image")
    before_request = payload.get("before_request")
    after_request = payload.get("after_request")
    expected = payload.get("expected", {})
    _require(isinstance(image_value, str) and image_value, "切换用例缺少 image")
    _require(isinstance(before_request, dict), "切换用例缺少 before_request")
    _require(isinstance(after_request, dict), "切换用例缺少 after_request")
    _require(isinstance(expected, dict), "切换用例 expected 必须是对象")
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = path.parent / image_path
    return image_path, before_request, after_request, expected


def override_selected_model_url(
    request_data: dict[str, Any], model_url: str | None
) -> dict[str, Any]:
    updated = copy.deepcopy(request_data)
    if model_url:
        ScenesVerifier._selected_model(updated)["ModelFile"] = model_url
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证生产指示灯服务契约")
    parser.add_argument("--base-url", required=True, help="例如 http://192.168.100.183:3005")
    parser.add_argument("--case", type=Path, help="可选的指示灯真实请求用例 JSON")
    parser.add_argument("--record", type=Path, help="可选的生产回流 record JSON")
    parser.add_argument("--image", type=Path, help="与 --record 配套的原始图片")
    parser.add_argument("--transition-case", type=Path, help="同型号不同注册图切换用例")
    parser.add_argument("--before-record", type=Path, help="注册图切换前的回流 record JSON")
    parser.add_argument("--after-record", type=Path, help="注册图切换后的回流 record JSON")
    parser.add_argument("--before-model-url", help="覆盖旧记录的注册图 URL")
    parser.add_argument("--after-model-url", help="覆盖新记录的注册图 URL")
    parser.add_argument(
        "--allow-registration-mutation",
        action="store_true",
        help="确认允许切换目标服务的注册向量和归档图",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verifier = ScenesVerifier(args.base_url, timeout=args.timeout)
    try:
        results = [
            verifier.check_health(),
            verifier.check_readiness(),
            verifier.check_openapi(),
        ]
        transition_records = bool(args.before_record or args.after_record)
        selected_modes = sum(
            bool(item) for item in (args.case, args.record, args.transition_case, transition_records)
        )
        _require(selected_modes <= 1, "--case、--record 与 --transition-case 只能使用一个")
        _require(not args.record or bool(args.image), "--record 必须同时提供 --image")
        _require(
            bool(args.before_record) == bool(args.after_record),
            "--before-record 与 --after-record 必须同时提供",
        )
        _require(
            not transition_records or bool(args.image),
            "注册图切换记录测试必须提供 --image",
        )
        if args.case:
            image_path, request_data = load_case(args.case)
            results.append(
                verifier.check_indicator_register_variants(image_path, request_data)
            )
        elif args.record and args.image:
            image_path, request_data = load_backflow_record(args.record, args.image)
            results.append(
                verifier.check_indicator_register_variants(image_path, request_data)
            )
        elif args.transition_case:
            _require(args.allow_registration_mutation, "注册图切换测试必须显式确认 --allow-registration-mutation")
            image_path, before_request, after_request, expected = load_transition_case(
                args.transition_case
            )
            results.append(
                verifier.check_registration_transition(
                    image_path, before_request, after_request, expected
                )
            )
        elif args.before_record and args.after_record and args.image:
            _require(args.allow_registration_mutation, "注册图切换测试必须显式确认 --allow-registration-mutation")
            image_path, before_request = load_backflow_record(args.before_record, args.image)
            _, after_request = load_backflow_record(args.after_record, args.image)
            before_request = override_selected_model_url(
                before_request, args.before_model_url
            )
            after_request = override_selected_model_url(
                after_request, args.after_model_url
            )
            results.append(
                verifier.check_registration_transition(
                    image_path, before_request, after_request
                )
            )
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(f"PASS {result.name}: {result.detail} ({result.elapsed_ms:.1f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
