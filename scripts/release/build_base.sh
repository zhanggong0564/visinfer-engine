#!/usr/bin/env bash
# =========================================================
# 构建共享基础镜像 mobile_vision:base（Dockerfile.base）。
#
# base = cuda 系统库 + 构建工具链 + 装好【全部 pip 依赖】的 venv + Cython。
# 它是 panel-label/scenes 两个服务镜像的公共依赖层，最慢最重，只需构建一次。
#
# 何时重建：仅当 requirements.txt / onnxruntime 等依赖变化时。
#   业务代码（框架）变化【不需要】重建 base——它在 runtime 镜像 builder 阶段编译；
#   场景插件更不进镜像，由部署侧 pkg/ 覆盖层挂入。
#
# 用法（仓库根目录）：
#   bash scripts/release/build_base.sh                 # 构建 mobile_vision:base
#   TAG=mobile_vision:base-20260626 bash scripts/release/build_base.sh   # 自定义 tag
#
# 构建完 base 后，分别构建内置基线插件的服务镜像：
#   docker build -f Dockerfile.panel-label -t mobile_vision:panel-label .
#   docker build -f Dockerfile.scenes -t mobile_vision:scenes .
# =========================================================
set -euo pipefail

# 脚本位于 scripts/release/，距仓库根两级
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

TAG="${TAG:-mobile_vision:base}"
BASE_IMAGE="${BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
ORT_WHEEL="whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
test -f "$ORT_WHEEL"

echo "==> 构建基础镜像 ${TAG}（Dockerfile.base）—— 含全部 pip 依赖，首次较慢"
docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f Dockerfile.base -t "${TAG}" .
echo "==> 完成。各场景镜像现在可 FROM ${TAG} 快速构建（见本脚本头部注释）。"
