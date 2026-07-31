#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIE_CONTRACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
ORT_WHEEL="whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
ENVIRONMENT_FILES=(
  Dockerfile.base
  Dockerfile.runtime
  requirements.txt
  requirements.scenes.txt
  "$ORT_WHEEL"
)

for path in "${ENVIRONMENT_FILES[@]}"; do
  test -f "$path" || {
    echo "镜像环境契约输入不存在: $path" >&2
    exit 1
  }
done

export LC_ALL=C
{
  printf 'cuda_base_image=%s\n' "$CUDA_BASE_IMAGE"
  for path in "${ENVIRONMENT_FILES[@]}"; do
    printf 'path=%s\nsha256=%s\n' \
      "$path" "$(sha256sum "$path" | awk '{print $1}')"
  done
} | sha256sum | awk '{print $1}'
