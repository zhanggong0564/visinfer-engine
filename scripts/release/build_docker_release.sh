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
test -f "$ORT_WHEEL" || {
  echo "缺少 ONNX Runtime wheel: $ORT_WHEEL" >&2
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
conda run -n "$CONDA_ENV" python scripts/release/collect_weight_paths.py --root weights \
  "${WEIGHT_CONFIGS[@]}" \
  >/dev/null

PANEL_REQUIREMENTS_SHA256="$(sha256sum requirements.txt | sha256sum | awk '{print $1}')"
SCENES_REQUIREMENTS_SHA256="$(sha256sum requirements.txt requirements.scenes.txt | sha256sum | awk '{print $1}')"
FRAMEWORK_VERSION="$(project_version pyproject.toml)"
OUT="${OUTPUT_DIR:-dist/docker-release-${RELEASE_VERSION}${OUTPUT_SUFFIX}}"
test ! -e "$OUT" || { echo "输出目录已存在: $OUT" >&2; exit 1; }

if [ "${SKIP_BASE_BUILD:-0}" != "1" ]; then
  docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -f Dockerfile.base -t mobile_vision:base .
elif ! docker image inspect mobile_vision:base >/dev/null 2>&1; then
  echo "SKIP_BASE_BUILD=1 但本机不存在 mobile_vision:base" >&2
  exit 1
fi

PANEL_RUNTIME_CONTRACT_SHA256="$(sha256sum requirements.txt Dockerfile.base Dockerfile.panel-label | sha256sum | awk '{print $1}')"
SCENES_RUNTIME_CONTRACT_SHA256="$(sha256sum requirements.txt requirements.scenes.txt Dockerfile.base Dockerfile.scenes | sha256sum | awk '{print $1}')"

if [ "$TARGET" = "panel-label" ] || [ "$TARGET" = "all" ]; then
  PANEL_PLUGIN_VERSIONS="panel_label=$(project_version plugins/vie-plugin-panel-label/pyproject.toml)"
  docker build -f Dockerfile.panel-label \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg RELEASE_VERSION="$RELEASE_VERSION" \
    --build-arg REQUIREMENTS_SHA256="$PANEL_REQUIREMENTS_SHA256" \
    --build-arg RUNTIME_CONTRACT_SHA256="$PANEL_RUNTIME_CONTRACT_SHA256" \
    --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION" \
    --build-arg PLUGIN_VERSIONS="$PANEL_PLUGIN_VERSIONS" \
    -t "mobile_vision:panel-label-${RELEASE_VERSION}" .
  docker run --rm --entrypoint python3.10 "mobile_vision:panel-label-${RELEASE_VERSION}" \
    -c "import importlib.metadata as m, importlib.util, onnxruntime as ort; assert {'panel_label'} <= {e.name for e in m.entry_points(group='vie.plugins')}; assert importlib.util.find_spec('chromadb') is None; assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()"
fi

if [ "$TARGET" = "scenes" ] || [ "$TARGET" = "all" ]; then
  SCENES_PLUGIN_VERSIONS=""
  for plugin in dc-fuse indicator-light lap-surf line-squeeze plate-screw; do
    plugin_version="$(project_version "plugins/vie-plugin-${plugin}/pyproject.toml")"
    plugin_label="${plugin//-/_}=${plugin_version}"
    SCENES_PLUGIN_VERSIONS="${SCENES_PLUGIN_VERSIONS:+${SCENES_PLUGIN_VERSIONS},}${plugin_label}"
  done
  docker build -f Dockerfile.scenes \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg RELEASE_VERSION="$RELEASE_VERSION" \
    --build-arg REQUIREMENTS_SHA256="$SCENES_REQUIREMENTS_SHA256" \
    --build-arg RUNTIME_CONTRACT_SHA256="$SCENES_RUNTIME_CONTRACT_SHA256" \
    --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION" \
    --build-arg PLUGIN_VERSIONS="$SCENES_PLUGIN_VERSIONS" \
    -t "mobile_vision:scenes-${RELEASE_VERSION}" .
  docker run --rm --entrypoint python3.10 "mobile_vision:scenes-${RELEASE_VERSION}" \
    -c "import chromadb, importlib.metadata as m, onnxruntime as ort; assert {'dc_fuse','indicator_light','lap_surf','line_squeeze','plate_screw'} <= {e.name for e in m.entry_points(group='vie.plugins')}; assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()"
fi

for service in "${SERVICES[@]}"; do
  if [ "$service" = "panel-label" ]; then
    IMAGE="mobile_vision:panel-label-${RELEASE_VERSION}"
    COMPOSE_FILE="docker-compose.panel-label.yml"
    IMAGE_VAR="PANEL_LABEL_IMAGE"
    HEALTH_URL="http://127.0.0.1:3001/health/ready"
    SYNC_SCRIPT="scripts/release/sync-plugin.sh"
  else
    IMAGE="mobile_vision:scenes-${RELEASE_VERSION}"
    COMPOSE_FILE="docker-compose.scenes.yml"
    IMAGE_VAR="SCENES_IMAGE"
    HEALTH_URL="http://127.0.0.1:3005/health/ready"
    SYNC_SCRIPT="scripts/release/sync-plugin-scenes.sh"
  fi

  RELEASE_ID="baseline-${RELEASE_VERSION}" bash "$SYNC_SCRIPT" --local
  mkdir -p "$OUT/$service"
  docker save "$IMAGE" | gzip -1 > "$OUT/$service/image.tar.gz"
  tar -czf "$OUT/$service/overlay.tar.gz" \
    -C ".release-staging/$service/baseline-${RELEASE_VERSION}" \
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
(cd "$OUT" && find "${SERVICES[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "离线发布包已生成: $OUT"
