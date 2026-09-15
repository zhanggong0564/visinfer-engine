#!/usr/bin/env bash
# Shared by local scripts and streamed SSH activation/rollback scripts.

install_deployment_compose() {
  local source_file="$1" target_file="$2"
  if [ "$(basename "$target_file")" = "docker-compose.scenes.yml" ]; then
    # Historical releases used a literal host port. Normalize only the deployed
    # copy; keep archived releases immutable and let Compose read the local .env.
    sed -E '
      s/"(\$\{SCENES_PORT:-3005\}|[0-9]+):[0-9]+"/"${SCENES_PORT:-3005}:${SCENES_PORT:-3005}"/
      /^[[:space:]]*-[[:space:]]*PORT=/d
      /^[[:space:]]*environment:/a\
      - PORT=${SCENES_PORT:-3005}
      s@http://127\.0\.0\.1:[0-9]+/health/ready@http://127.0.0.1:${SCENES_PORT:-3005}/health/ready@g
    ' \
      "$source_file" > "${target_file}.next"
  else
    cp "$source_file" "${target_file}.next"
  fi
  "${COMPOSE[@]}" -f "${target_file}.next" config --quiet
  mv -f "${target_file}.next" "$target_file"
}

deployment_health_url() {
  # Retain the compose-file argument for existing callers. Read published ports
  # without assuming that the application still listens on container port 3001.
  local binding port
  binding="$(docker port "$2")" || return 1
  port="$(printf '%s\n' "$binding" | sed 's/.*://' | sort -u)"
  if ! [[ "$port" =~ ^[0-9]+$ ]] || \
      [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "无法确定服务 $2 的健康检查端口" >&2
    return 1
  fi
  printf 'http://127.0.0.1:%s/health/ready\n' "$port"
}
