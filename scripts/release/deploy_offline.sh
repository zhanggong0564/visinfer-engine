#!/usr/bin/env bash
set -euo pipefail

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

BUNDLE="$(cd "$BUNDLE" && pwd)"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
source "$SERVICE/release.env"
gunzip -c "$SERVICE/image.tar.gz" | docker load

mkdir -p "$DEPLOY_DIR/releases/$RELEASE_VERSION" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/data"
chown -R 1000:1000 "$DEPLOY_DIR/logs" "$DEPLOY_DIR/data"
tar -xzf "$SERVICE/overlay.tar.gz" -C "$DEPLOY_DIR/releases/$RELEASE_VERSION"
cp "$SERVICE/$COMPOSE_FILE" "$DEPLOY_DIR/$COMPOSE_FILE"
cp "$SERVICE/release.env" "$DEPLOY_DIR/.env"
cd "$DEPLOY_DIR"
if [ -L current ]; then
  ln -sfn "$(readlink current)" previous
fi
ln -sfn "releases/$RELEASE_VERSION" current.next
mv -Tf current.next current
docker compose -f "$COMPOSE_FILE" config --quiet
docker compose -f "$COMPOSE_FILE" up -d --force-recreate
for _ in $(seq 1 60); do
  curl -fsS "$HEALTH_URL" >/dev/null && exit 0
  sleep 5
done
docker compose -f "$COMPOSE_FILE" logs --tail=200 >&2 || true
exit 1
