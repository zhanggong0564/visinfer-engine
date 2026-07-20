#!/usr/bin/env bash
set -euo pipefail

select_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
  else
    echo "未找到 Docker Compose（docker compose 或 docker-compose）" >&2
    exit 1
  fi
}

BUNDLE=""
SERVICE=""
DEPLOY_DIR=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle) shift; BUNDLE="${1:?--bundle 缺少值}" ;;
    --service) shift; SERVICE="${1:?--service 缺少值}" ;;
    --deploy-dir) shift; DEPLOY_DIR="${1:?--deploy-dir 缺少值}" ;;
    -h|--help)
      echo "用法: $0 --bundle /path/docker-release-V --service panel-label|scenes --deploy-dir /path"
      exit 0
      ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done
: "${BUNDLE:?必须指定 --bundle}"
: "${SERVICE:?必须指定 --service}"
: "${DEPLOY_DIR:?必须指定 --deploy-dir}"
case "$SERVICE" in panel-label|scenes) ;; *) echo "无效服务: $SERVICE" >&2; exit 2 ;; esac
select_compose

BUNDLE="$(cd "$BUNDLE" && pwd)"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
source "$SERVICE/release.env"
gunzip -c image.tar.gz | docker load

mkdir -p "$DEPLOY_DIR"
DEPLOY_DIR="$(cd "$DEPLOY_DIR" && pwd)"
mkdir -p "$DEPLOY_DIR/releases/$RELEASE_VERSION" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/data"
if [ "$SERVICE" = "panel-label" ]; then
  DEPLOY_IMAGE="${PANEL_LABEL_IMAGE:?发布包缺少 PANEL_LABEL_IMAGE}"
else
  DEPLOY_IMAGE="${SCENES_IMAGE:?发布包缺少 SCENES_IMAGE}"
fi
if ! chown -R 1000:1000 "$DEPLOY_DIR/logs" "$DEPLOY_DIR/data" 2>/dev/null; then
  docker run --rm --user 0:0 --entrypoint chown \
    --volume "$DEPLOY_DIR:/deploy" \
    "$DEPLOY_IMAGE" -R 1000:1000 /deploy/logs /deploy/data
fi
tar -xzf "$SERVICE/overlay.tar.gz" -C "$DEPLOY_DIR/releases/$RELEASE_VERSION"
cp "$SERVICE/$COMPOSE_FILE" "$DEPLOY_DIR/$COMPOSE_FILE"
cp "$SERVICE/release.env" "$DEPLOY_DIR/.env"
cd "$DEPLOY_DIR"
if [ -L current ]; then
  ln -sfn "$(readlink current)" previous
fi
ln -sfn "releases/$RELEASE_VERSION" current.next
mv -Tf current.next current
"${COMPOSE[@]}" -f "$COMPOSE_FILE" config --quiet
"${COMPOSE[@]}" -f "$COMPOSE_FILE" up -d --force-recreate
for _ in $(seq 1 60); do
  curl -fsS "$HEALTH_URL" >/dev/null && exit 0
  sleep 5
done
"${COMPOSE[@]}" -f "$COMPOSE_FILE" logs --tail=200 >&2 || true
exit 1
