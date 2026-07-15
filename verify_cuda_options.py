#!/usr/bin/env python
"""验证 ONNX Runtime CUDA provider 选项是否真实生效。"""

from pathlib import Path
from typing import Callable

from services.base.inference_runner import OnnxRuntimeRunner


DEFAULT_MODEL_PATH = Path(
    "weights/panel_label/text_rec/"
    "PP-OCRv5_server_rec_merged_v6_diff_lr.onnx"
)


def _options_match(actual: dict, expected: dict) -> bool:
    """Return whether all configured CUDA options match runtime values."""
    return all(
        str(actual.get(key)) == str(value)
        for key, value in expected.items()
    )


def main(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    runner_factory: Callable[..., OnnxRuntimeRunner] = OnnxRuntimeRunner,
) -> int:
    """验证真实 CUDA Session 及其 provider options，返回进程退出码。"""
    path = Path(model_path)
    print("=" * 60)
    print("ONNX Runtime CUDA Provider 配置验证")
    print("=" * 60)

    if not path.exists():
        print(f"✗ 测试模型不存在: {path}")
        return 1

    try:
        runner = runner_factory(
            str(path),
            warmup=False,
            require_cuda=True,
        )
    except Exception as exc:
        print(f"✗ 模型加载失败: {exc}")
        return 1

    if "CUDAExecutionProvider" not in runner.providers:
        print(f"✗ 实际 providers 未包含 CUDA: {runner.providers}")
        return 1

    try:
        provider_options = runner._session.get_provider_options()
    except Exception as exc:
        print(f"✗ 无法读取 Session provider options: {exc}")
        return 1

    actual_options = provider_options.get("CUDAExecutionProvider")
    if not isinstance(actual_options, dict):
        print("✗ Session 未返回 CUDA provider options")
        return 1

    expected_options = OnnxRuntimeRunner._cuda_provider_options()
    if not _options_match(actual_options, expected_options):
        print(f"✗ CUDA provider options 不匹配: {actual_options}")
        print(f"  预期配置: {expected_options}")
        return 1

    print(f"✓ 模型: {path}")
    print(f"✓ 实际 providers: {runner.providers}")
    print(f"✓ CUDA provider options: {actual_options}")
    print("验证完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
