#!/usr/bin/env bash
# =========================================================
# 构建共享镜像 mobile_vision:base-builder 和 mobile_vision:base。
#
# base-builder = CUDA + 构建工具链 + 全部 pip 依赖 + framework。
# base = CUDA 运行库 + 全部 pip 依赖 + framework，不包含编译工具链。
# 场景插件使用 base-builder 编译，panel-label/scenes runtime 继承 base。
#
# 何时重建：CUDA 底座、requirements、ONNX Runtime 或 framework 变化时。
#   场景插件变化不需要重建 base，只重建对应 runtime。
#
# 用法（仓库根目录）：
#   bash scripts/release/build_base.sh
#   BASE_TAG=mobile_vision:base-20260626 \
#   BASE_BUILDER_TAG=mobile_vision:base-builder-20260626 \
#     bash scripts/release/build_base.sh
#
# 构建完 base 后，通过 build_docker_release.sh 构建具体场景 runtime。
# =========================================================
set -euo pipefail

# 脚本位于 scripts/release/，距仓库根两级
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BASE_TAG="${BASE_TAG:-${TAG:-mobile_vision:base}}"
BASE_BUILDER_TAG="${BASE_BUILDER_TAG:-mobile_vision:base-builder}"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
CONDA_ENV="${CONDA_ENV:-mobile_vision}"
ORT_WHEEL="whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
ORT_WHEEL_SHA256="a5b4e1641db48752118dda353b8614c6d6570344062b58faea70b5350c41cf68"
test -f "$ORT_WHEEL" || { echo "缺少 ONNX Runtime wheel: $ORT_WHEEL" >&2; exit 1; }
test "$(sha256sum "$ORT_WHEEL" | awk '{print $1}')" = "$ORT_WHEEL_SHA256" || {
  echo "ONNX Runtime wheel 不是 CUDA 12/cuDNN 9 官方构建: $ORT_WHEEL" >&2
  exit 1
}
REQUIREMENTS_SHA256="$(sha256sum requirements.txt requirements.scenes.txt | sha256sum | awk '{print $1}')"
BASE_CONTRACT_SHA256="$(CUDA_BASE_IMAGE="$CUDA_BASE_IMAGE" bash scripts/release/compute_base_contract.sh)"
FRAMEWORK_VERSION="$(conda run -n "$CONDA_ENV" python -c \
  'from setuptools.config.pyprojecttoml import read_configuration; print(read_configuration("pyproject.toml", expand=False)["project"]["version"])')"

echo "==> 构建编译镜像 ${BASE_BUILDER_TAG} 和运行基础镜像 ${BASE_TAG}"
BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/vie-base-build.XXXXXX")"
cleanup() {
  rm -rf -- "$BUILD_CONTEXT"
}
trap cleanup EXIT
mkdir -p "$BUILD_CONTEXT/scripts/release" "$BUILD_CONTEXT/whl"
cp -a .dockerignore Dockerfile.base pyproject.toml setup.py \
  requirements.txt requirements.scenes.txt services schemas routers utils config \
  "$BUILD_CONTEXT/"
cp -a scripts/release/build_wheels.py "$BUILD_CONTEXT/scripts/release/"
cp -L "$ORT_WHEEL" "$BUILD_CONTEXT/$ORT_WHEEL"

DOCKER_BUILD_ARGS=(
  --build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}"
  --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256"
  --build-arg BASE_CONTRACT_SHA256="$BASE_CONTRACT_SHA256"
  --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION"
  -f "$BUILD_CONTEXT/Dockerfile.base"
)
docker build "${DOCKER_BUILD_ARGS[@]}" \
  --target base-builder -t "$BASE_BUILDER_TAG" "$BUILD_CONTEXT"
docker build "${DOCKER_BUILD_ARGS[@]}" \
  -t "$BASE_TAG" "$BUILD_CONTEXT"
test "$(docker image inspect --format '{{index .Config.Labels "io.vie.image-role"}}' "$BASE_BUILDER_TAG")" = "builder"
test "$(docker image inspect --format '{{index .Config.Labels "io.vie.image-role"}}' "$BASE_TAG")" = "runtime-base"
echo "==> 完成。插件使用 ${BASE_BUILDER_TAG} 编译，场景 runtime 继承 ${BASE_TAG}。"
