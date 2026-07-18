#!/usr/bin/env bash
# Shared local driver for atomic plugin releases. Service wrappers define the arrays below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DO_BUILD=1
DO_PUSH=1
DO_WEIGHTS=1
REMOTE="${REMOTE:-}"
REMOTE_DIR="${REMOTE_DIR:-}"
RELEASE_ID="${RELEASE_ID:-$(date +%Y%m%d%H%M%S)-$(git rev-parse --short HEAD)}"
CONDA_ENV="${CONDA_ENV:-mobile_vision}"
CONDA_PYTHON=(conda run -n "$CONDA_ENV" python)
WHEEL_BUILDER_IMAGE="${WHEEL_BUILDER_IMAGE:-mobile_vision:base}"

usage() {
  echo "用法: $0 [--local] [--no-build] [--no-weights] [--remote user@host] [--remote-dir /path]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --local) DO_PUSH=0 ;;
    --no-build) DO_BUILD=0 ;;
    --no-weights) DO_WEIGHTS=0 ;;
    --remote) shift; REMOTE="${1:?--remote 缺少值}" ;;
    --remote-dir) shift; REMOTE_DIR="${1:?--remote-dir 缺少值}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ "$DO_PUSH" -eq 1 ]; then
  : "${REMOTE:?请通过 REMOTE 或 --remote 指定服务器}"
  : "${REMOTE_DIR:?请通过 REMOTE_DIR 或 --remote-dir 指定部署目录}"
fi

if [ "$DO_BUILD" -eq 1 ]; then
  for pattern in "${WHEEL_PATTERNS[@]}"; do
    find dist -maxdepth 1 -type f -name "$pattern" -delete 2>/dev/null || true
  done
  BUILD_WHEEL_ARGS=(--plugins "${PLUGINS[@]}")
  if "${CONDA_PYTHON[@]}" -c "import Cython" >/dev/null 2>&1; then
    BUILD_WHEEL_ARGS=(--no-isolation "${BUILD_WHEEL_ARGS[@]}")
    "${CONDA_PYTHON[@]}" scripts/release/build_wheels.py "${BUILD_WHEEL_ARGS[@]}"
  elif docker image inspect "$WHEEL_BUILDER_IMAGE" >/dev/null 2>&1; then
    echo "Conda 环境 ${CONDA_ENV} 未安装 Cython，使用 ${WHEEL_BUILDER_IMAGE} 构建 wheel"
    docker run --rm --user "$(id -u):$(id -g)" \
      --volume "$ROOT:/workspace" --workdir /workspace \
      "$WHEEL_BUILDER_IMAGE" python scripts/release/build_wheels.py \
      --no-isolation "${BUILD_WHEEL_ARGS[@]}"
  else
    echo "未找到 Cython 或 ${WHEEL_BUILDER_IMAGE}，使用隔离构建"
    "${CONDA_PYTHON[@]}" scripts/release/build_wheels.py "${BUILD_WHEEL_ARGS[@]}"
  fi
fi

LOCAL_STAGE=".release-staging/${SERVICE}/${RELEASE_ID}"
if [ -e "$LOCAL_STAGE" ]; then
  echo "发布暂存目录已存在: $LOCAL_STAGE" >&2
  exit 1
fi
mkdir -p "$LOCAL_STAGE/pkg" "$LOCAL_STAGE/static" "$LOCAL_STAGE/weights"

for pattern in "${WHEEL_PATTERNS[@]}"; do
  mapfile -t wheels < <(find dist -maxdepth 1 -type f -name "$pattern" -print | sort)
  if [ "${#wheels[@]}" -ne 1 ]; then
    echo "期望且仅允许一个 wheel 匹配 $pattern，实际 ${#wheels[@]} 个" >&2
    exit 1
  fi
  unzip -q "${wheels[0]}" -d "$LOCAL_STAGE/pkg"
done

for entrypoint in "${EXPECTED_ENTRYPOINTS[@]}"; do
  if ! rg -l "${entrypoint}" "$LOCAL_STAGE/pkg"/*dist-info/entry_points.txt >/dev/null; then
    echo "发布包缺少 entry point: $entrypoint" >&2
    exit 1
  fi
done

cp app.py "$LOCAL_STAGE/app.py"
cp -R static/swagger-ui "$LOCAL_STAGE/static/"
cp "$COMPOSE_FILE" "$LOCAL_STAGE/$COMPOSE_FILE"

"${CONDA_PYTHON[@]}" scripts/release/collect_weight_paths.py \
  --root weights "${CONFIGS[@]}" > "$LOCAL_STAGE/weight-paths.txt"
if [ "$DO_WEIGHTS" -eq 1 ]; then
  rsync -a --files-from="$LOCAL_STAGE/weight-paths.txt" weights/ "$LOCAL_STAGE/weights/"
fi

REQUIREMENTS_SHA256="$(sha256sum "${RUNTIME_REQUIREMENTS[@]}" | sha256sum | awk '{print $1}')"
PYTHON_ABI="$("${CONDA_PYTHON[@]}" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
RUNTIME_CONTRACT_SHA256="$(sha256sum "${RUNTIME_REQUIREMENTS[@]}" Dockerfile.base "$RUNTIME_DOCKERFILE" | sha256sum | awk '{print $1}')"
cat > "$LOCAL_STAGE/release.env" <<EOF
RELEASE_ID=${RELEASE_ID}
SERVICE=${SERVICE}
REQUIREMENTS_SHA256=${REQUIREMENTS_SHA256}
PYTHON_ABI=${PYTHON_ABI}
RUNTIME_CONTRACT_SHA256=${RUNTIME_CONTRACT_SHA256}
EOF

# 保持 --local 的旧行为：pkg/ 仍得到本次完整覆盖层。
rm -rf pkg
cp -a "$LOCAL_STAGE/pkg" pkg
if [ "$DO_PUSH" -eq 0 ]; then
  echo "本地发布已生成: $LOCAL_STAGE"
  exit 0
fi

REMOTE_STAGE="${REMOTE_DIR}/releases/${RELEASE_ID}.staging"
ssh "$REMOTE" "test ! -e '${REMOTE_STAGE}' && mkdir -p '${REMOTE_STAGE}'"
rsync -az --delete "$LOCAL_STAGE/" "${REMOTE}:${REMOTE_STAGE}/"

ssh "$REMOTE" bash -s -- \
  "$REMOTE_DIR" "$RELEASE_ID" "$COMPOSE_FILE" "$CONTAINER_NAME" \
  "$HEALTH_URL" "$REQUIREMENTS_SHA256" "$PYTHON_ABI" \
  "$RUNTIME_CONTRACT_SHA256" "$DO_WEIGHTS" \
  "$(IFS=,; echo "${EXPECTED_ENTRYPOINTS[*]}")" \
  < scripts/release/remote_activate.sh

echo "发布完成: ${SERVICE} ${RELEASE_ID}"
