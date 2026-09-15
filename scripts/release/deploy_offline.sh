#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

mkdir -p "$DEPLOY_DIR"
DEPLOY_DIR="$(cd "$DEPLOY_DIR" && pwd)"
BUNDLE="$(cd "$BUNDLE" && pwd)"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
source "$SCRIPT_DIR/deployment_compose.sh"

# Read only release-owned scalar metadata; never execute the target's .env.
RELEASE_VERSION="$(sed -n 's/^RELEASE_VERSION=//p' "$SERVICE/release.env")"
if ! [[ "$RELEASE_VERSION" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "发布包版本无效" >&2
  exit 2
fi
COMPOSE_FILE="docker-compose.${SERVICE}.yml"
CONTAINER_NAME="mobile-vision-${SERVICE}"
if [ "$SERVICE" = "panel-label" ]; then
  IMAGE_VAR=PANEL_LABEL_IMAGE
else
  IMAGE_VAR=SCENES_IMAGE
fi
DEPLOY_IMAGE="$(sed -n "s/^${IMAGE_VAR}=//p" "$SERVICE/release.env")"
: "${DEPLOY_IMAGE:?发布包缺少成品镜像}"
export "${IMAGE_VAR}=${DEPLOY_IMAGE}"
if [ -e "$DEPLOY_DIR/releases/$RELEASE_VERSION" ]; then
  echo "发布目录已存在，不覆盖历史版本: $RELEASE_VERSION" >&2
  exit 1
fi

mkdir -p "$DEPLOY_DIR/.release-backups"
BACKUP_DIR="$(mktemp -d "$DEPLOY_DIR/.release-backups/offline-${RELEASE_VERSION}-XXXXXX")"
TARGET_ENV=/dev/null
if [ -f "$DEPLOY_DIR/.env" ]; then
  cp -p "$DEPLOY_DIR/.env" "$BACKUP_DIR/.env"
  TARGET_ENV="$BACKUP_DIR/.env"
fi
if [ -f "$DEPLOY_DIR/$COMPOSE_FILE" ]; then
  cp -p "$DEPLOY_DIR/$COMPOSE_FILE" "$BACKUP_DIR/original-compose.yml"
fi
for pointer in current previous; do
  if [ -L "$DEPLOY_DIR/$pointer" ]; then
    readlink "$DEPLOY_DIR/$pointer" > "$BACKUP_DIR/$pointer.target"
  fi
done

# Package metadata wins; all other target settings (including empty values and
# quoted host lists) win over package defaults. Keep their original text intact.
awk '
  function key(line) {
    sub(/^[[:space:]]*(export[[:space:]]+)?/, "", line)
    if (line !~ /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/) return ""
    sub(/[[:space:]]*=.*/, "", line)
    return line
  }
  function metadata(k) {
    return k ~ /^(RELEASE_VERSION|PANEL_LABEL_IMAGE|SCENES_IMAGE|COMPOSE_FILE|HEALTH_URL)$/
  }
  FILENAME == ARGV[1] { lines[++n]=$0; k=key($0); if(k!="") seen[k]=1; next }
  { k=key($0); if(k!="HEALTH_URL" && (metadata(k) || !seen[k])) print }
  END { for(i=1;i<=n;i++) if(!metadata(key(lines[i]))) print lines[i] }
' "$TARGET_ENV" "$SERVICE/release.env" > "$BACKUP_DIR/new.env"
chmod 600 "$BACKUP_DIR/new.env"

# Validate the merged configuration before loading images or changing the live
# configuration/pointers. Compose handles .env quoting and interpolation.
COMPOSE+=(--project-directory "$DEPLOY_DIR" --env-file "$BACKUP_DIR/new.env")
install_deployment_compose "$SERVICE/$COMPOSE_FILE" "$BACKUP_DIR/$COMPOSE_FILE"

gunzip -c image.tar.gz | docker load
mkdir -p "$DEPLOY_DIR/releases/$RELEASE_VERSION" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/data"
if ! chown -R 1000:1000 "$DEPLOY_DIR/logs" "$DEPLOY_DIR/data" 2>/dev/null; then
  docker run --rm --user 0:0 --entrypoint chown \
    --volume "$DEPLOY_DIR:/deploy" \
    "$DEPLOY_IMAGE" -R 1000:1000 /deploy/logs /deploy/data
fi
tar -xzf "$SERVICE/overlay.tar.gz" -C "$DEPLOY_DIR/releases/$RELEASE_VERSION"
cp "$BACKUP_DIR/$COMPOSE_FILE" "$DEPLOY_DIR/$COMPOSE_FILE"
cp "$BACKUP_DIR/new.env" "$DEPLOY_DIR/.env.next"
chmod 600 "$DEPLOY_DIR/.env.next"
mv -f "$DEPLOY_DIR/.env.next" "$DEPLOY_DIR/.env"
cd "$DEPLOY_DIR"
# Use the installed environment for activation and persist its actual health URL.
select_compose
if [ -L current ]; then
  ln -sfn "$(readlink current)" previous
fi
ln -sfn "releases/$RELEASE_VERSION" current.next
mv -Tf current.next current
"${COMPOSE[@]}" -f "$COMPOSE_FILE" config --quiet
"${COMPOSE[@]}" -f "$COMPOSE_FILE" up -d --no-build --force-recreate "$CONTAINER_NAME"
HEALTH_URL="$(deployment_health_url "$COMPOSE_FILE" "$CONTAINER_NAME")"
printf '\nHEALTH_URL=%s\n' "$HEALTH_URL" >> .env
echo "部署配置备份: $BACKUP_DIR"
for _ in $(seq 1 60); do
  curl -fsS --max-time 10 "$HEALTH_URL" >/dev/null && exit 0
  sleep 5
done
"${COMPOSE[@]}" -f "$COMPOSE_FILE" logs --tail=200 >&2 || true
exit 1
