#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
用法: RELEASE_VERSION=2.1.3 build_docker_release.sh [--service panel|scenes|all]
     build_docker_release.sh [panel|scenes|all] 2.1.3

--service panel  只构建 panel-label 服务
--service scenes 只构建 scenes 服务
--service all    构建两个服务（默认，兼容旧用法）
EOF
}

TARGET="${BUILD_TARGET:-all}"
TARGET_SET=0
if [ -n "${BUILD_TARGET:-}" ]; then
  TARGET_SET=1
fi
RELEASE_VERSION="${RELEASE_VERSION:-}"
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --service|--target)
      [ "$TARGET_SET" -eq 0 ] || { echo "服务目标只能指定一次" >&2; exit 2; }
      shift
      TARGET="${1:?$0: --service 缺少值}"
      TARGET_SET=1
      ;;
    --version)
      shift
      RELEASE_VERSION="${1:?$0: --version 缺少值}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done

for arg in "${POSITIONAL[@]}"; do
  case "$arg" in
    panel|panel-label|scenes|all)
      [ "$TARGET_SET" -eq 0 ] || { echo "服务目标只能指定一次: $arg" >&2; exit 2; }
      TARGET="$arg"
      TARGET_SET=1
      ;;
    *)
      [ -z "$RELEASE_VERSION" ] || { echo "未知参数或重复版本号: $arg" >&2; usage >&2; exit 2; }
      RELEASE_VERSION="$arg"
      ;;
  esac
done

: "${RELEASE_VERSION:?用法: RELEASE_VERSION=2.1.3 $0 [--service panel|scenes|all]}"
case "$TARGET" in
  panel|panel-label)
    TARGET="panel-label"
    SERVICES=(panel-label)
    OUTPUT_SUFFIX="-panel-label"
    ;;
  scenes)
    SERVICES=(scenes)
    OUTPUT_SUFFIX="-scenes"
    ;;
  all)
    SERVICES=(panel-label scenes)
    OUTPUT_SUFFIX=""
    ;;
  *)
    echo "无效服务目标: $TARGET（可选 panel、scenes 或 all）" >&2
    exit 2
    ;;
esac

BASE_IMAGE="${BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
CONDA_ENV="${CONDA_ENV:-mobile_vision}"
export CONDA_ENV

project_version() {
  conda run -n "$CONDA_ENV" python -c \
    'import sys; from setuptools.config.pyprojecttoml import read_configuration; print(read_configuration(sys.argv[1], expand=False)["project"]["version"])' \
    "$1"
}

ORT_WHEEL="whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
ORT_WHEEL_SHA256="a5b4e1641db48752118dda353b8614c6d6570344062b58faea70b5350c41cf68"
test -f "$ORT_WHEEL" || {
  echo "缺少 ONNX Runtime wheel: $ORT_WHEEL" >&2
  exit 1
}
test "$(sha256sum "$ORT_WHEEL" | awk '{print $1}')" = "$ORT_WHEEL_SHA256" || {
  echo "ONNX Runtime wheel 不是 CUDA 12/cuDNN 9 官方构建: $ORT_WHEEL" >&2
  exit 1
}

PANEL_CONFIGS=(
  plugins/vie-plugin-panel-label/vie_plugin_panel_label/config.py
)
SCENES_CONFIGS=(
  plugins/vie-plugin-dc-fuse/vie_plugin_dc_fuse/config.py
  plugins/vie-plugin-indicator-light/vie_plugin_indicator_light/config.py
  plugins/vie-plugin-lap-surf/vie_plugin_lap_surf/config.py
  plugins/vie-plugin-line-squeeze/vie_plugin_line_squeeze/config.py
  plugins/vie-plugin-plate-screw/vie_plugin_plate_screw/config.py
)
WEIGHT_CONFIGS=()
case "$TARGET" in
  panel-label) WEIGHT_CONFIGS=("${PANEL_CONFIGS[@]}") ;;
  scenes) WEIGHT_CONFIGS=("${SCENES_CONFIGS[@]}") ;;
  all) WEIGHT_CONFIGS=("${PANEL_CONFIGS[@]}" "${SCENES_CONFIGS[@]}") ;;
esac

if [ "$TARGET" = "scenes" ] || [ "$TARGET" = "all" ]; then
  test -f weights/line_squeeze/rec_ppocrv5en_v1.onnx || {
    echo "缺少 line-squeeze ONNX OCR 权重，请先执行模型导出" >&2
    exit 1
  }
fi
WEIGHT_PATH_OUTPUT="$(conda run -n "$CONDA_ENV" python \
  scripts/release/collect_weight_paths.py --root weights "${WEIGHT_CONFIGS[@]}")"
mapfile -t WEIGHT_PATHS <<< "$WEIGHT_PATH_OUTPUT"
CUDA_SMOKE_MODEL=""
for weight_path in "${WEIGHT_PATHS[@]}"; do
  if [[ "$weight_path" = *.onnx ]]; then
    CUDA_SMOKE_MODEL="$weight_path"
    break
  fi
done
test -n "$CUDA_SMOKE_MODEL" || {
  echo "发布权重中没有可用于 CUDA Session 验证的 ONNX 模型" >&2
  exit 1
}

REQUIREMENTS_SHA256="$(sha256sum requirements.txt requirements.scenes.txt | sha256sum | awk '{print $1}')"
FRAMEWORK_VERSION="$(project_version pyproject.toml)"
OUT="${OUTPUT_DIR:-dist/docker-release-${RELEASE_VERSION}${OUTPUT_SUFFIX}}"
test ! -e "$OUT" || { echo "输出目录已存在: $OUT" >&2; exit 1; }

BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/vie-docker-release.XXXXXX")"
cleanup() {
  rm -rf -- "$BUILD_CONTEXT"
}
trap cleanup EXIT

mkdir -p "$BUILD_CONTEXT/scripts/release" "$BUILD_CONTEXT/whl"
cp -a .dockerignore Dockerfile.base Dockerfile.runtime \
  pyproject.toml setup.py requirements.txt requirements.scenes.txt app.py \
  services schemas routers utils config static \
  "$BUILD_CONTEXT/"
cp -a scripts/release/build_wheels.py "$BUILD_CONTEXT/scripts/release/"
# whl/ may be a symlink to storage outside the repository. Docker does not
# follow links outside its build context, so stage the wheel as a regular file.
cp -L "$ORT_WHEEL" "$BUILD_CONTEXT/$ORT_WHEEL"

if [ "${SKIP_BASE_BUILD:-0}" != "1" ]; then
  docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
    -f "$BUILD_CONTEXT/Dockerfile.base" -t mobile_vision:base "$BUILD_CONTEXT"
elif ! docker image inspect mobile_vision:base >/dev/null 2>&1; then
  echo "SKIP_BASE_BUILD=1 但本机不存在 mobile_vision:base" >&2
  exit 1
elif [ "$(docker image inspect --format '{{index .Config.Labels "io.vie.requirements-sha256"}}' mobile_vision:base)" != "$REQUIREMENTS_SHA256" ]; then
  echo "SKIP_BASE_BUILD=1 但 mobile_vision:base 依赖指纹不匹配" >&2
  exit 1
elif [ "$(docker image inspect --format '{{index .Config.Labels "io.vie.base-image"}}' mobile_vision:base)" != "$BASE_IMAGE" ]; then
  echo "SKIP_BASE_BUILD=1 但 mobile_vision:base 基础镜像不匹配" >&2
  exit 1
fi

RUNTIME_CONTRACT_SHA256="$(sha256sum requirements.txt requirements.scenes.txt Dockerfile.base Dockerfile.runtime | sha256sum | awk '{print $1}')"
RUNTIME_IMAGE="mobile_vision:runtime-${RELEASE_VERSION}"
docker build -f "$BUILD_CONTEXT/Dockerfile.runtime" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg RELEASE_VERSION="$RELEASE_VERSION" \
  --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
  --build-arg RUNTIME_CONTRACT_SHA256="$RUNTIME_CONTRACT_SHA256" \
  --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION" \
  -t "$RUNTIME_IMAGE" "$BUILD_CONTEXT"
docker run --rm --entrypoint python3.10 "$RUNTIME_IMAGE" \
  -c "import chromadb, importlib.metadata as m, onnxruntime as ort; from services.scenario_registry import scenario_registry; expected={'panel_label','dc_fuse','indicator_light','lap_surf','line_squeeze','plate_screw'}; actual={e.name for e in m.entry_points(group='vie.plugins')}; assert expected.isdisjoint(actual), actual; assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()"
docker run --rm --gpus all --entrypoint python3.10 \
  --volume "$ROOT/weights:/app/workspace/weights:ro" \
  --env "CUDA_SMOKE_MODEL=/app/workspace/weights/$CUDA_SMOKE_MODEL" \
  "$RUNTIME_IMAGE" \
  -c "import os, onnxruntime as ort; session=ort.InferenceSession(os.environ['CUDA_SMOKE_MODEL'], providers=['CUDAExecutionProvider']); assert session.get_providers()[0] == 'CUDAExecutionProvider', session.get_providers()"

mkdir -p "$OUT"
docker save "$RUNTIME_IMAGE" | gzip -1 > "$OUT/image.tar.gz"

for service in "${SERVICES[@]}"; do
  if [ "$service" = "panel-label" ]; then
    COMPOSE_FILE="docker-compose.panel-label.yml"
    HEALTH_URL="http://127.0.0.1:3001/health/ready"
    SYNC_SCRIPT="scripts/release/sync-plugin.sh"
    EXPECTED_PLUGINS=(panel_label)
  else
    COMPOSE_FILE="docker-compose.scenes.yml"
    HEALTH_URL="http://127.0.0.1:3005/health/ready"
    SYNC_SCRIPT="scripts/release/sync-plugin-scenes.sh"
    EXPECTED_PLUGINS=(dc_fuse indicator_light lap_surf line_squeeze plate_screw)
  fi

  INCLUDE_FRAMEWORK=0 RELEASE_ID="baseline-${RELEASE_VERSION}" \
    bash "$SYNC_SCRIPT" --local
  LOCAL_STAGE=".release-staging/$service/baseline-${RELEASE_VERSION}"
  EXPECTED_PLUGIN_NAMES="$(IFS=,; echo "${EXPECTED_PLUGINS[*]}")"
  docker run --rm --entrypoint python3.10 \
    --volume "$ROOT/$LOCAL_STAGE/pkg:/app/workspace/pkg:ro" \
    --volume "$ROOT/$LOCAL_STAGE/weights:/app/workspace/weights:ro" \
    --env PYTHONPATH=/app/workspace/pkg \
    --env "EXPECTED_VIE_PLUGINS=$EXPECTED_PLUGIN_NAMES" \
    "$RUNTIME_IMAGE" \
    -c "import importlib.metadata as m, os; expected=set(os.environ['EXPECTED_VIE_PLUGINS'].split(',')); entry_points=list(m.entry_points(group='vie.plugins')); actual={entry_point.name for entry_point in entry_points}; assert actual == expected, (actual, expected); [entry_point.load() for entry_point in entry_points]"
  mkdir -p "$OUT/$service"
  tar -czf "$OUT/$service/overlay.tar.gz" \
    -C "$LOCAL_STAGE" \
    pkg weights app.py static weight-paths.txt "$COMPOSE_FILE" release.env
  cp "$COMPOSE_FILE" "$OUT/$service/$COMPOSE_FILE"
  cat > "$OUT/$service/release.env" <<EOF
RELEASE_VERSION=${RELEASE_VERSION}
VIE_RUNTIME_IMAGE=${RUNTIME_IMAGE}
COMPOSE_FILE=${COMPOSE_FILE}
HEALTH_URL=${HEALTH_URL}
EOF
done

cp scripts/release/deploy_offline.sh "$OUT/deploy_offline.sh"
(cd "$OUT" && find image.tar.gz "${SERVICES[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "离线发布包已生成: $OUT"
