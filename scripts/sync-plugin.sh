#!/usr/bin/env bash
# =========================================================
# panel_label 业务代码 + 模型权重热更新：
#   编译 .so → 解包到 pkg/ → 同步 pkg/ 与 weights/panel_label/ 到服务器 → 重启容器
#
# 适用场景：改了业务逻辑（框架 services/schemas/routers/utils/config 或
# panel_label 插件）或更新了模型权重，环境未变。免去重打整镜像（几个 G）
# 重传——代码只传几 MB 的 .so，权重 rsync 增量只传变化的模型文件。
# 原理见 docker-compose.panel-label.yml 里 PYTHONPATH 与 weights 覆盖层。
#
# 用法（仓库根目录）：
#   bash scripts/sync-plugin.sh                 # 编译+同步+远程重启（默认服务器）
#   bash scripts/sync-plugin.sh --local         # 只编译+解包到 pkg/，不连服务器
#   bash scripts/sync-plugin.sh --no-build      # 跳过编译，用已有 dist/*.whl
#   bash scripts/sync-plugin.sh --no-weights    # 跳过权重同步，只传代码
#   REMOTE=user@host REMOTE_DIR=/path bash scripts/sync-plugin.sh   # 覆盖目标
#
# ⚠️ ABI：本机用来编译的 Python 必须是 CPython 3.10（与镜像一致），且 glibc 不高于
#    容器（ubuntu22.04 / glibc 2.35）。本机 WSL ubuntu22.04 或 conda py310 一般满足。
#    若远程 import 报 .so 版本/符号错误，改在 builder 容器里编：
#      docker build --target builder -f Dockerfile.panel-label -t vie:builder .
#      docker run --rm -v "$PWD":/src -w /src vie:builder \
#        python scripts/build_wheels.py --no-isolation --plugins panel-label
#    再 bash scripts/sync-plugin.sh --no-build 同步。
# =========================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
REMOTE="${REMOTE:-sun@192.168.100.183}"
REMOTE_DIR="${REMOTE_DIR:-/media/sun/V1/zhanggong/deploy/mobile_vison/deploy3}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.panel-label.yml}"

DO_BUILD=1
DO_PUSH=1
DO_WEIGHTS=1
for arg in "$@"; do
  case "$arg" in
    --no-build)   DO_BUILD=0 ;;
    --local)      DO_PUSH=0 ;;
    --no-weights) DO_WEIGHTS=0 ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 1 ;;
  esac
done

if [ "$DO_BUILD" -eq 1 ]; then
  echo "==> [1/3] 编译 framework + panel_label → dist/*.whl"
  # 先清旧 wheel，避免跨多次构建堆积、后续 glob 选到陈旧版本
  rm -f dist/vie_framework-*.whl dist/vie_plugin_panel_label-*.whl
  "$PYTHON" scripts/build_wheels.py --no-isolation --plugins panel-label
fi

echo "==> [2/3] 解包 wheel 到 pkg/（PYTHONPATH 覆盖层）"
shopt -s nullglob
FW=(dist/vie_framework-*.whl)
PL=(dist/vie_plugin_panel_label-*.whl)
if [ ${#FW[@]} -eq 0 ] || [ ${#PL[@]} -eq 0 ]; then
  echo "!! dist/ 缺少 wheel，请先去掉 --no-build 编译" >&2; exit 1
fi
rm -rf pkg && mkdir -p pkg
# wheel 即 zip：解出 routers/services/schemas/utils/config + vie_plugin_panel_label + *.dist-info
unzip -o -q "${FW[-1]}" -d pkg
unzip -o -q "${PL[-1]}" -d pkg
echo "    pkg/ 顶层："; ls -1 pkg | sed 's/^/      /'

if [ "$DO_PUSH" -eq 0 ]; then
  echo "==> [3/3] --local：跳过同步与重启。本地 pkg/ 已就绪。"
  exit 0
fi

echo "==> [3/3] 同步 pkg/（及权重）到 ${REMOTE}:${REMOTE_DIR} 并重启容器"
ssh "$REMOTE" "mkdir -p '${REMOTE_DIR}/pkg' '${REMOTE_DIR}/weights/panel_label'"
rsync -avz --delete pkg/ "${REMOTE}:${REMOTE_DIR}/pkg/"
if [ "$DO_WEIGHTS" -eq 1 ]; then
  # 增量同步：内容没变的模型不会重传；--delete 保证远端与本地版本目录一致，
  # 本地源即远端真相（config 指的模型路径也来自本地，二者天然同步）。
  # 前提：远端 compose 已挂载 ./weights:/app/workspace/weights:ro 覆盖层。
  rsync -avz --delete weights/panel_label/ "${REMOTE}:${REMOTE_DIR}/weights/panel_label/"
fi
ssh "$REMOTE" "cd '${REMOTE_DIR}' && docker compose -f '${COMPOSE_FILE}' restart"
echo "==> 完成。验证：bash verify-QF2.sh  （或查日志 docker compose logs -f）"
