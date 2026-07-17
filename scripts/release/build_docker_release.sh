#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RELEASE_VERSION="${RELEASE_VERSION:-${1:-}}"
: "${RELEASE_VERSION:?用法: RELEASE_VERSION=2.1.3 $0}"
BASE_IMAGE="${BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"

ORT_WHEEL="whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
WHEEL_CONTEXT="$(readlink -f whl)"
test -f "$(readlink -f "$ORT_WHEEL")" || {
  echo "缺少 ONNX Runtime wheel: $ORT_WHEEL" >&2
  exit 1
}
test -f weights/line_squeeze/rec_ppocrv5en_v1.onnx || {
  echo "缺少 line-squeeze ONNX OCR 权重，请先执行模型导出" >&2
  exit 1
}
conda run -n ppocr python scripts/release/collect_weight_paths.py --root weights \
  plugins/vie-plugin-panel-label/vie_plugin_panel_label/config.py \
  plugins/vie-plugin-dc-fuse/vie_plugin_dc_fuse/config.py \
  plugins/vie-plugin-indicator-light/vie_plugin_indicator_light/config.py \
  plugins/vie-plugin-lap-surf/vie_plugin_lap_surf/config.py \
  plugins/vie-plugin-line-squeeze/vie_plugin_line_squeeze/config.py \
  plugins/vie-plugin-plate-screw/vie_plugin_plate_screw/config.py \
  >/dev/null

REQUIREMENTS_SHA256="$(sha256sum requirements.txt | awk '{print $1}')"
FRAMEWORK_VERSION="$(conda run -n ppocr python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
OUT="dist/docker-release-${RELEASE_VERSION}"
test ! -e "$OUT" || { echo "输出目录已存在: $OUT" >&2; exit 1; }
mkdir -p "$OUT/panel-label" "$OUT/scenes"

if [ "${SKIP_BASE_BUILD:-0}" != "1" ]; then
  docker build --build-context "ort_wheel=${WHEEL_CONTEXT}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" -f Dockerfile.base -t mobile_vision:base .
fi
PANEL_RUNTIME_CONTRACT_SHA256="$(sha256sum requirements.txt Dockerfile.base Dockerfile.panel-label | sha256sum | awk '{print $1}')"
SCENES_RUNTIME_CONTRACT_SHA256="$(sha256sum requirements.txt Dockerfile.base Dockerfile.scenes | sha256sum | awk '{print $1}')"
PANEL_PLUGIN_VERSIONS="$(conda run -n ppocr python -c "import tomllib; print('panel_label=' + tomllib.load(open('plugins/vie-plugin-panel-label/pyproject.toml','rb'))['project']['version'])")"
SCENES_PLUGIN_VERSIONS="$(conda run -n ppocr python -c "import tomllib; from pathlib import Path; names=['dc-fuse','indicator-light','lap-surf','line-squeeze','plate-screw']; print(','.join(n.replace('-','_') + '=' + tomllib.load(open(Path('plugins') / ('vie-plugin-' + n) / 'pyproject.toml','rb'))['project']['version'] for n in names))")"
docker build -f Dockerfile.panel-label \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg RELEASE_VERSION="$RELEASE_VERSION" \
  --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
  --build-arg RUNTIME_CONTRACT_SHA256="$PANEL_RUNTIME_CONTRACT_SHA256" \
  --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION" \
  --build-arg PLUGIN_VERSIONS="$PANEL_PLUGIN_VERSIONS" \
  -t "mobile_vision:panel-label-${RELEASE_VERSION}" .
docker build -f Dockerfile.scenes \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg RELEASE_VERSION="$RELEASE_VERSION" \
  --build-arg REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
  --build-arg RUNTIME_CONTRACT_SHA256="$SCENES_RUNTIME_CONTRACT_SHA256" \
  --build-arg FRAMEWORK_VERSION="$FRAMEWORK_VERSION" \
  --build-arg PLUGIN_VERSIONS="$SCENES_PLUGIN_VERSIONS" \
  -t "mobile_vision:scenes-${RELEASE_VERSION}" .

docker run --rm --entrypoint python3.10 "mobile_vision:panel-label-${RELEASE_VERSION}" \
  -c "import importlib.metadata as m; assert {'panel_label'} <= {e.name for e in m.entry_points(group='vie.plugins')}"
docker run --rm --entrypoint python3.10 "mobile_vision:scenes-${RELEASE_VERSION}" \
  -c "import importlib.metadata as m; assert {'dc_fuse','indicator_light','lap_surf','line_squeeze','plate_screw'} <= {e.name for e in m.entry_points(group='vie.plugins')}"

RELEASE_ID="baseline-${RELEASE_VERSION}" bash scripts/release/sync-plugin.sh --local
RELEASE_ID="baseline-${RELEASE_VERSION}" bash scripts/release/sync-plugin-scenes.sh --local

for service in panel-label scenes; do
  if [ "$service" = "panel-label" ]; then
    IMAGE="mobile_vision:panel-label-${RELEASE_VERSION}"
    COMPOSE_FILE="docker-compose.panel-label.yml"
    IMAGE_VAR="PANEL_LABEL_IMAGE"
    HEALTH_URL="http://127.0.0.1:3001/health/ready"
  else
    IMAGE="mobile_vision:scenes-${RELEASE_VERSION}"
    COMPOSE_FILE="docker-compose.scenes.yml"
    IMAGE_VAR="SCENES_IMAGE"
    HEALTH_URL="http://127.0.0.1:3005/health/ready"
  fi
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
(cd "$OUT" && find panel-label scenes -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "离线发布包已生成: $OUT"
