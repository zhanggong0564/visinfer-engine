#!/usr/bin/env bash
# Five-scene atomic hot update: framework + plugins + app/static + referenced weights.
set -euo pipefail

SERVICE="scenes"
RUNTIME_DOCKERFILE="Dockerfile.runtime"
RUNTIME_REQUIREMENTS=(requirements.txt requirements.scenes.txt)
PLUGINS=(dc-fuse indicator-light lap-surf line-squeeze plate-screw)
WHEEL_PATTERNS=(
  "vie_framework-*.whl"
  "vie_plugin_dc_fuse-*.whl"
  "vie_plugin_indicator_light-*.whl"
  "vie_plugin_lap_surf-*.whl"
  "vie_plugin_line_squeeze-*.whl"
  "vie_plugin_plate_screw-*.whl"
)
CONFIGS=(
  "plugins/vie-plugin-dc-fuse/vie_plugin_dc_fuse/config.py"
  "plugins/vie-plugin-indicator-light/vie_plugin_indicator_light/config.py"
  "plugins/vie-plugin-lap-surf/vie_plugin_lap_surf/config.py"
  "plugins/vie-plugin-line-squeeze/vie_plugin_line_squeeze/config.py"
  "plugins/vie-plugin-plate-screw/vie_plugin_plate_screw/config.py"
)
EXPECTED_ENTRYPOINTS=(dc_fuse indicator_light lap_surf line_squeeze plate_screw)
COMPOSE_FILE="docker-compose.scenes.yml"
CONTAINER_NAME="mobile-vision-scenes"
HEALTH_URL="http://127.0.0.1:3005/health/ready"

source "$(dirname "$0")/sync-common.sh" "$@"
