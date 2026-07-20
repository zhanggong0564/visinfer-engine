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

CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
BASE_TAG="${BASE_TAG:-mobile_vision:base}"
BASE_BUILDER_TAG="${BASE_BUILDER_TAG:-mobile_vision:base-builder}"
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
BASE_CONTRACT_SHA256="$(CUDA_BASE_IMAGE="$CUDA_BASE_IMAGE" bash scripts/release/compute_base_contract.sh)"
FRAMEWORK_VERSION="$(project_version pyproject.toml)"
OUT="${OUTPUT_DIR:-dist/docker-release-${RELEASE_VERSION}${OUTPUT_SUFFIX}}"
test ! -e "$OUT" || { echo "输出目录已存在: $OUT" >&2; exit 1; }

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vie-docker-release.XXXXXX")"
cleanup() {
  rm -rf -- "$BUILD_ROOT"
}
trap cleanup EXIT

BASE_CONTEXT="$BUILD_ROOT/base"
mkdir -p "$BASE_CONTEXT/scripts/release" "$BASE_CONTEXT/whl"
cp -a .dockerignore Dockerfile.base pyproject.toml setup.py \
  requirements.txt requirements.scenes.txt services schemas routers utils config \
  "$BASE_CONTEXT/"
cp -a scripts/release/build_wheels.py "$BASE_CONTEXT/scripts/release/"
cp -L "$ORT_WHEEL" "$BASE_CONTEXT/$ORT_WHEEL"

DOCKER_BASE_BUILD_ARGS=(
  --build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}"
  --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256"
  --build-arg BASE_CONTRACT_SHA256="$BASE_CONTRACT_SHA256"
  --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION"
  -f "$BASE_CONTEXT/Dockerfile.base"
)

if [ "${SKIP_BASE_BUILD:-0}" != "1" ]; then
  docker build "${DOCKER_BASE_BUILD_ARGS[@]}" \
    --target base-builder -t "$BASE_BUILDER_TAG" "$BASE_CONTEXT"
  docker build "${DOCKER_BASE_BUILD_ARGS[@]}" \
    -t "$BASE_TAG" "$BASE_CONTEXT"
fi

validate_base_image() {
  local image="$1"
  local expected_role="$2"
  docker image inspect "$image" >/dev/null 2>&1 || {
    echo "本机不存在 ${image}" >&2
    return 1
  }
  [ "$(docker image inspect --format '{{index .Config.Labels "io.vie.image-role"}}' "$image")" = "$expected_role" ] || {
    echo "${image} 镜像角色不匹配，期望 ${expected_role}" >&2
    return 1
  }
  [ "$(docker image inspect --format '{{index .Config.Labels "io.vie.base-contract-sha256"}}' "$image")" = "$BASE_CONTRACT_SHA256" ] || {
    echo "${image} 基础环境指纹不匹配" >&2
    return 1
  }
  [ "$(docker image inspect --format '{{index .Config.Labels "io.vie.base-image"}}' "$image")" = "$CUDA_BASE_IMAGE" ] || {
    echo "${image} CUDA 基础镜像不匹配" >&2
    return 1
  }
}

validate_base_image "$BASE_BUILDER_TAG" builder
validate_base_image "$BASE_TAG" runtime-base

RUNTIME_CONTRACT_SHA256="$(
  {
    printf '%s\n' "$BASE_CONTRACT_SHA256"
    sha256sum Dockerfile.runtime
  } | sha256sum | awk '{print $1}'
)"

stage_runtime_context() {
  local context="$1"
  shift
  mkdir -p "$context/scripts/release" "$context/plugins"
  cp -a .dockerignore Dockerfile.runtime app.py static "$context/"
  cp -a scripts/release/build_wheels.py "$context/scripts/release/"
  local plugin_name plugin_source plugin_target
  for plugin_name in "$@"; do
    plugin_source="plugins/vie-plugin-${plugin_name}"
    plugin_target="$context/plugins/vie-plugin-${plugin_name}"
    mkdir -p "$plugin_target"
    cp -a "$plugin_source/pyproject.toml" "$plugin_source/setup.py" "$plugin_target/"
    find "$plugin_source" -maxdepth 1 -type d -name 'vie_plugin_*' \
      -exec cp -a {} "$plugin_target/" \;
  done
}

validate_runtime_image() {
  local image="$1"
  local expected_plugins="$2"
  docker run --rm --entrypoint python3.10 \
    --env "EXPECTED_VIE_PLUGINS=${expected_plugins}" \
    "$image" \
    -c "import importlib.metadata as m, os, onnxruntime as ort; from services.scenario_registry import scenario_registry; expected=set(os.environ['EXPECTED_VIE_PLUGINS'].split(',')); entry_points=list(m.entry_points(group='vie.plugins')); actual={entry_point.name for entry_point in entry_points}; assert actual == expected, (actual, expected); [entry_point.load() for entry_point in entry_points]; assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()"
  docker run --rm --gpus all --entrypoint python3.10 \
    --volume "$ROOT/weights:/app/workspace/weights:ro" \
    --env "CUDA_SMOKE_MODEL=/app/workspace/weights/$CUDA_SMOKE_MODEL" \
    "$image" \
    -c "import os, onnxruntime as ort; session=ort.InferenceSession(os.environ['CUDA_SMOKE_MODEL'], providers=['CUDAExecutionProvider']); assert session.get_providers()[0] == 'CUDAExecutionProvider', session.get_providers()"
}

IMAGES=()
if [ "$TARGET" = "panel-label" ] || [ "$TARGET" = "all" ]; then
  PANEL_CONTEXT="$BUILD_ROOT/panel-label"
  stage_runtime_context "$PANEL_CONTEXT" panel-label
  PANEL_IMAGE="mobile_vision:panel-label-${RELEASE_VERSION}"
  PANEL_PLUGIN_VERSIONS="panel_label=$(project_version plugins/vie-plugin-panel-label/pyproject.toml)"
  docker build -f "$PANEL_CONTEXT/Dockerfile.runtime" \
    --build-arg "BASE_IMAGE=${BASE_TAG}" \
    --build-arg "BUILDER_IMAGE=${BASE_BUILDER_TAG}" \
    --build-arg PLUGINS="panel-label" \
    --build-arg PLUGIN_NAMES="panel_label" \
    --build-arg PLUGIN_VERSIONS="$PANEL_PLUGIN_VERSIONS" \
    --build-arg RELEASE_VERSION="$RELEASE_VERSION" \
    --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
    --build-arg BASE_CONTRACT_SHA256="$BASE_CONTRACT_SHA256" \
    --build-arg RUNTIME_CONTRACT_SHA256="$RUNTIME_CONTRACT_SHA256" \
    --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION" \
    -t "$PANEL_IMAGE" "$PANEL_CONTEXT"
  validate_runtime_image "$PANEL_IMAGE" "panel_label"
  IMAGES+=("$PANEL_IMAGE")
fi

if [ "$TARGET" = "scenes" ] || [ "$TARGET" = "all" ]; then
  SCENES_PLUGINS=(dc-fuse indicator-light lap-surf line-squeeze plate-screw)
  SCENES_CONTEXT="$BUILD_ROOT/scenes"
  stage_runtime_context "$SCENES_CONTEXT" "${SCENES_PLUGINS[@]}"
  SCENES_IMAGE="mobile_vision:scenes-${RELEASE_VERSION}"
  SCENES_PLUGIN_VERSIONS=""
  for plugin_name in "${SCENES_PLUGINS[@]}"; do
    plugin_version="$(project_version "plugins/vie-plugin-${plugin_name}/pyproject.toml")"
    plugin_label="${plugin_name//-/_}=${plugin_version}"
    SCENES_PLUGIN_VERSIONS="${SCENES_PLUGIN_VERSIONS:+${SCENES_PLUGIN_VERSIONS},}${plugin_label}"
  done
  docker build -f "$SCENES_CONTEXT/Dockerfile.runtime" \
    --build-arg "BASE_IMAGE=${BASE_TAG}" \
    --build-arg "BUILDER_IMAGE=${BASE_BUILDER_TAG}" \
    --build-arg PLUGINS="${SCENES_PLUGINS[*]}" \
    --build-arg PLUGIN_NAMES="dc_fuse,indicator_light,lap_surf,line_squeeze,plate_screw" \
    --build-arg PLUGIN_VERSIONS="$SCENES_PLUGIN_VERSIONS" \
    --build-arg RELEASE_VERSION="$RELEASE_VERSION" \
    --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
    --build-arg BASE_CONTRACT_SHA256="$BASE_CONTRACT_SHA256" \
    --build-arg RUNTIME_CONTRACT_SHA256="$RUNTIME_CONTRACT_SHA256" \
    --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION" \
    -t "$SCENES_IMAGE" "$SCENES_CONTEXT"
  validate_runtime_image "$SCENES_IMAGE" \
    "dc_fuse,indicator_light,lap_surf,line_squeeze,plate_screw"
  IMAGES+=("$SCENES_IMAGE")
fi

mkdir -p "$OUT"
docker save "${IMAGES[@]}" | gzip -1 > "$OUT/image.tar.gz"

for service in "${SERVICES[@]}"; do
  if [ "$service" = "panel-label" ]; then
    IMAGE="$PANEL_IMAGE"
    IMAGE_VAR="PANEL_LABEL_IMAGE"
    COMPOSE_FILE="docker-compose.panel-label.yml"
    HEALTH_URL="http://127.0.0.1:3001/health/ready"
    SYNC_SCRIPT="scripts/release/sync-plugin.sh"
  else
    IMAGE="$SCENES_IMAGE"
    IMAGE_VAR="SCENES_IMAGE"
    COMPOSE_FILE="docker-compose.scenes.yml"
    HEALTH_URL="http://127.0.0.1:3005/health/ready"
    SYNC_SCRIPT="scripts/release/sync-plugin-scenes.sh"
  fi

  INCLUDE_FRAMEWORK=0 INCLUDE_PLUGINS=0 RELEASE_ID="baseline-${RELEASE_VERSION}" \
    bash "$SYNC_SCRIPT" --local
  LOCAL_STAGE=".release-staging/$service/baseline-${RELEASE_VERSION}"
  mkdir -p "$OUT/$service"
  tar -czf "$OUT/$service/overlay.tar.gz" \
    -C "$LOCAL_STAGE" \
    pkg weights app.py static weight-paths.txt "$COMPOSE_FILE" release.env
  cp "$COMPOSE_FILE" "$OUT/$service/$COMPOSE_FILE"
  cat > "$OUT/$service/release.env" <<EOF
RELEASE_VERSION=${RELEASE_VERSION}
${IMAGE_VAR}=${IMAGE}
COMPOSE_FILE=${COMPOSE_FILE}
HEALTH_URL=${HEALTH_URL}
EOF
done

cp scripts/release/deploy_offline.sh "$OUT/deploy_offline.sh"
(cd "$OUT" && find image.tar.gz "${SERVICES[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "离线发布包已生成: $OUT"
