#!/usr/bin/env python
"""验证 ONNX Runtime CUDA provider 选项是否生效"""

import sys
from pathlib import Path

# 添加项目根到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.base.inference_runner import OnnxRuntimeRunner
from config.config import settings

print("=" * 60)
print("ONNX Runtime CUDA Provider 配置验证")
print("=" * 60)

print("\n1. 配置文件读取:")
print(f"   ORT_CUDA_DEVICE_ID: {settings.ORT_CUDA_DEVICE_ID}")
print(f"   ORT_CUDNN_CONV_ALGO_SEARCH: {settings.ORT_CUDNN_CONV_ALGO_SEARCH}")
print(f"   ORT_ARENA_EXTEND_STRATEGY: {settings.ORT_ARENA_EXTEND_STRATEGY}")
print(f"   ORT_CUDA_MEM_LIMIT_GB: {settings.ORT_CUDA_MEM_LIMIT_GB}")

print("\n2. 构造 CUDA provider 选项:")
cuda_opts = OnnxRuntimeRunner._cuda_provider_options()
print(f"   {cuda_opts}")

print("\n3. Provider 列表注入测试:")
providers_before = ["CUDAExecutionProvider", "CPUExecutionProvider"]
print(f"   注入前: {providers_before}")
providers_after = OnnxRuntimeRunner._with_cuda_options(providers_before)
print(f"   注入后: {providers_after}")

print("\n4. 实际模型加载验证(如果模型存在):")
test_model = "weights/panel_label/text_rec/PP-OCRv5_server_rec_merged_v6_diff_lr.onnx"
if Path(test_model).exists():
    try:
        runner = OnnxRuntimeRunner(test_model, warmup=False)
        print(f"   ✓ 模型加载成功: {test_model}")
        print(f"   ✓ 使用的 providers: {runner.providers}")
        # 尝试获取实际的 provider options(需要 ONNX Runtime 支持)
        try:
            session = runner._session
            if hasattr(session, 'get_provider_options'):
                opts = session.get_provider_options()
                if 'CUDAExecutionProvider' in opts:
                    cuda_actual = opts['CUDAExecutionProvider']
                    print(f"   ✓ CUDA 实际选项: {cuda_actual}")
        except Exception as e:
            print(f"   ⚠ 无法读取 session provider options: {e}")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
else:
    print(f"   ⚠ 测试模型不存在,跳过: {test_model}")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)
