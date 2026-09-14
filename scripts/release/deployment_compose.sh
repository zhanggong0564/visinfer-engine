#!/usr/bin/env bash
# Shared by local scripts and streamed SSH activation/rollback scripts.

install_deployment_compose() {
  local source_file="$1" target_file="$2"
  if [ "$(basename "$target_file")" = "docker-compose.scenes.yml" ]; then
    # Historical releases used a literal host port. Normalize only the deployed
    # copy; keep archived releases immutable and let Compose read the local .env.
    sed -E 's/^([[:space:]]*-[[:space:]]*)"[0-9]+:3001"([[:space:]]*(#.*)?)$/\1"${SCENES_PORT:-3005}:3001"\2/' \
      "$source_file" > "${target_file}.next"
  else
    cp "$source_file" "${target_file}.next"
  fi
  "${COMPOSE[@]}" -f "${target_file}.next" config --quiet
  mv -f "${target_file}.next" "$target_file"
}

deployment_health_url() {
  local binding port
  binding="$("${COMPOSE[@]}" -f "$1" port "$2" 3001)" || return 1
  port="${binding##*:}"
  if [[ "$binding" = *$'\n'* ]] || ! [[ "$port" =~ ^[0-9]+$ ]] || \
      [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "无法确定服务 $2 的健康检查端口" >&2
    return 1
  fi
  printf 'http://127.0.0.1:%s/health/ready\n' "$port"
}
