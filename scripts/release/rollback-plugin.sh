#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-}"
REMOTE_DIR="${REMOTE_DIR:-}"
SERVICE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --remote) shift; REMOTE="${1:?--remote 缺少值}" ;;
    --remote-dir) shift; REMOTE_DIR="${1:?--remote-dir 缺少值}" ;;
    --service) shift; SERVICE="${1:?--service 缺少值}" ;;
    -h|--help)
      echo "用法: $0 --remote user@host --remote-dir /path [--service panel-label|scenes]"
      exit 0
      ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done
: "${REMOTE:?必须指定 --remote}"
: "${REMOTE_DIR:?必须指定 --remote-dir}"

ssh "$REMOTE" bash -s -- "$REMOTE_DIR" "$SERVICE" <<'REMOTE_SCRIPT'
set -euo pipefail
ROOT="$1"
SERVICE="$2"
cd "$ROOT"
test -L current
test -L previous

if [ -z "$SERVICE" ]; then
  if [ -f "$(readlink -f previous)/docker-compose.panel-label.yml" ]; then
    SERVICE="panel-label"
  else
    SERVICE="scenes"
  fi
fi
if [ "$SERVICE" = "panel-label" ]; then
  COMPOSE_FILE="docker-compose.panel-label.yml"
  HEALTH_URL="http://127.0.0.1:3001/health/ready"
else
  COMPOSE_FILE="docker-compose.scenes.yml"
  HEALTH_URL="http://127.0.0.1:3005/health/ready"
fi

CURRENT_TARGET="$(readlink current)"
PREVIOUS_TARGET="$(readlink previous)"
ln -sfn "$PREVIOUS_TARGET" current.rollback
mv -Tf current.rollback current
ln -sfn "$CURRENT_TARGET" previous.rollback
mv -Tf previous.rollback previous
cp "$(readlink -f current)/$COMPOSE_FILE" "$ROOT/$COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" up -d --force-recreate
for _ in $(seq 1 60); do
  curl -fsS "$HEALTH_URL" >/dev/null && exit 0
  sleep 5
done
echo "回滚版本未通过 /health/ready，恢复原版本" >&2
ln -sfn "$CURRENT_TARGET" current.restore
mv -Tf current.restore current
ln -sfn "$PREVIOUS_TARGET" previous.restore
mv -Tf previous.restore previous
cp "$(readlink -f current)/$COMPOSE_FILE" "$ROOT/$COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" up -d --force-recreate
exit 1
REMOTE_SCRIPT
