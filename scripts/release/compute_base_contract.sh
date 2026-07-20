#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
ORT_WHEEL="whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
BASE_FILES=(
  .dockerignore
  Dockerfile.base
  pyproject.toml
  setup.py
  requirements.txt
  requirements.scenes.txt
  scripts/release/build_wheels.py
  scripts/release/compute_base_contract.sh
  "$ORT_WHEEL"
)
FRAMEWORK_DIRS=(services schemas routers utils config)

for path in "${BASE_FILES[@]}" "${FRAMEWORK_DIRS[@]}"; do
  test -e "$path" || {
    echo "基础镜像合同输入不存在: $path" >&2
    exit 1
  }
done

hash_file() {
  local path="$1"
  printf 'path=%s\nsha256=%s\n' \
    "$path" "$(sha256sum "$path" | awk '{print $1}')"
}

export LC_ALL=C
{
  printf 'cuda_base_image=%s\n' "$CUDA_BASE_IMAGE"
  for path in "${BASE_FILES[@]}"; do
    hash_file "$path"
  done
  find "${FRAMEWORK_DIRS[@]}" -type f \
    ! -path '*/__pycache__/*' \
    ! -path '*/logs/*' \
    ! -name '*.pyc' \
    ! -name '*.pyo' \
    ! -name '*.so' \
    -print0 \
    | sort -z \
    | while IFS= read -r -d '' path; do
        hash_file "$path"
      done
} | sha256sum | awk '{print $1}'
