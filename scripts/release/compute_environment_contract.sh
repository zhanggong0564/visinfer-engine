#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIE_CONTRACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

LEGACY_IMAGE_NAMES=0
case "${1:-}" in
  "") ;;
  --legacy-image-names) LEGACY_IMAGE_NAMES=1 ;;
  *) echo "未知环境契约参数: $1" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "环境契约参数过多" >&2; exit 2; }

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

# Only the two known default image aliases may vary. FROM/RUN/COPY,
# CUDA, requirements and the ORT wheel remain part of the full fingerprint.
if [ "$LEGACY_IMAGE_NAMES" -eq 1 ]; then
  case "$(sed -n '/^ARG BASE_IMAGE=/p' Dockerfile.runtime)" in
    ARG\ BASE_IMAGE=mobile_vision/runtime-base:latest|ARG\ BASE_IMAGE=mobile_vision:base) ;;
    *) echo "不支持的运行基础镜像名称" >&2; exit 2 ;;
  esac
  case "$(sed -n '/^ARG BUILDER_IMAGE=/p' Dockerfile.runtime)" in
    ARG\ BUILDER_IMAGE=mobile_vision/build-base:latest|ARG\ BUILDER_IMAGE=mobile_vision:base-builder) ;;
    *) echo "不支持的构建基础镜像名称" >&2; exit 2 ;;
  esac
fi

file_sha256() {
  if [ "$LEGACY_IMAGE_NAMES" -eq 1 ] && [ "$1" = "Dockerfile.runtime" ]; then
    sed -e 's@^ARG BASE_IMAGE=mobile_vision/runtime-base:latest$@ARG BASE_IMAGE=mobile_vision:base@' \
        -e 's@^ARG BUILDER_IMAGE=mobile_vision/build-base:latest$@ARG BUILDER_IMAGE=mobile_vision:base-builder@' \
        "$1" | sha256sum | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

export LC_ALL=C
{
  printf 'cuda_base_image=%s\n' "$CUDA_BASE_IMAGE"
  for path in "${ENVIRONMENT_FILES[@]}"; do
    printf 'path=%s\nsha256=%s\n' \
      "$path" "$(file_sha256 "$path")"
  done
} | sha256sum | awk '{print $1}'
