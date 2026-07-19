#!/usr/bin/env bash
# panel_label atomic hot update: framework + plugin + app/static + referenced weights.
set -euo pipefail

SERVICE="panel-label"
RUNTIME_DOCKERFILE="Dockerfile.runtime"
RUNTIME_REQUIREMENTS=(requirements.txt requirements.scenes.txt)
PLUGINS=(panel-label)
WHEEL_PATTERNS=("vie_framework-*.whl" "vie_plugin_panel_label-*.whl")
CONFIGS=("plugins/vie-plugin-panel-label/vie_plugin_panel_label/config.py")
EXPECTED_ENTRYPOINTS=(panel_label)
COMPOSE_FILE="docker-compose.panel-label.yml"
CONTAINER_NAME="mobile-vision-panel-label"
HEALTH_URL="http://127.0.0.1:3001/health/ready"

source "$(dirname "$0")/sync-common.sh" "$@"
