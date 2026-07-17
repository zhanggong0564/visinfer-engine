#!/usr/bin/env bash
# Runs on the deployment host; activates a staged release and rolls back on failure.
set -euo pipefail

ROOT="$1"
RELEASE_ID="$2"
COMPOSE_FILE="$3"
CONTAINER_NAME="$4"
HEALTH_URL="$5"
EXPECTED_REQUIREMENTS_SHA="$6"
EXPECTED_PYTHON_ABI="$7"
EXPECTED_RUNTIME_CONTRACT_SHA="$8"
WITH_WEIGHTS="$9"
EXPECTED_ENTRYPOINTS="${10}"
STAGE="${ROOT}/releases/${RELEASE_ID}.staging"
FINAL="${ROOT}/releases/${RELEASE_ID}"

cd "$ROOT"
test -d "$STAGE/pkg"
test -f "$STAGE/app.py"
test -f "$STAGE/$COMPOSE_FILE"

IMAGE_REF="$(docker inspect --format '{{.Config.Image}}' "$CONTAINER_NAME")"
IMAGE_REQUIREMENTS_SHA="$(docker image inspect --format '{{index .Config.Labels "io.vie.requirements-sha256"}}' "$IMAGE_REF")"
IMAGE_PYTHON_ABI="$(docker image inspect --format '{{index .Config.Labels "io.vie.python-abi"}}' "$IMAGE_REF")"
IMAGE_RUNTIME_CONTRACT_SHA="$(docker image inspect --format '{{index .Config.Labels "io.vie.runtime-contract-sha256"}}' "$IMAGE_REF")"
if [ "$IMAGE_REQUIREMENTS_SHA" != "$EXPECTED_REQUIREMENTS_SHA" ]; then
  echo "依赖指纹不一致，必须重新构建镜像" >&2
  exit 1
fi
if [ "$IMAGE_PYTHON_ABI" != "$EXPECTED_PYTHON_ABI" ]; then
  echo "Python ABI 不一致，必须重新构建镜像" >&2
  exit 1
fi
if [ "$IMAGE_RUNTIME_CONTRACT_SHA" != "$EXPECTED_RUNTIME_CONTRACT_SHA" ]; then
  echo "系统运行时契约不一致，必须重新构建镜像" >&2
  exit 1
fi

if [ "$WITH_WEIGHTS" -eq 0 ]; then
  test -L current
  cp -al "$(readlink -f current)/weights/." "$STAGE/weights/"
fi
while IFS= read -r weight; do
  test -f "$STAGE/weights/$weight" || {
    echo "暂存发布缺少权重: $weight" >&2
    exit 1
  }
done < "$STAGE/weight-paths.txt"

docker run --rm --entrypoint python3.10 \
  -e PYTHONPATH=/app/workspace/pkg \
  -v "$STAGE/pkg:/app/workspace/pkg:ro" \
  "$IMAGE_REF" -c \
  "import importlib.metadata as m; expected=set('${EXPECTED_ENTRYPOINTS}'.split(',')); actual={e.name for e in m.entry_points(group='vie.plugins')}; assert expected <= actual, (expected, actual)"

mv "$STAGE" "$FINAL"
OLD_TARGET=""
if [ -L current ]; then
  OLD_TARGET="$(readlink current)"
  ln -sfn "$OLD_TARGET" previous
fi
ln -sfn "releases/${RELEASE_ID}" current.next
mv -Tf current.next current
cp "$FINAL/$COMPOSE_FILE" "$ROOT/$COMPOSE_FILE"

rollback() {
  trap - ERR
  if [ -n "$OLD_TARGET" ]; then
    ln -sfn "$OLD_TARGET" current.rollback
    mv -Tf current.rollback current
    cp "$(readlink -f current)/$COMPOSE_FILE" "$ROOT/$COMPOSE_FILE"
    docker compose -f "$COMPOSE_FILE" up -d --force-recreate
    for _ in $(seq 1 60); do
      curl -fsS "$HEALTH_URL" >/dev/null && return 0
      sleep 5
    done
    echo "旧版本也未通过 readiness，请立即人工检查" >&2
    return 1
  fi
}
trap 'rollback' ERR

docker compose -f "$COMPOSE_FILE" config --quiet
docker compose -f "$COMPOSE_FILE" up -d --force-recreate
for _ in $(seq 1 60); do
  if curl -fsS "$HEALTH_URL" >/dev/null; then
    exit 0
  fi
  sleep 5
done

echo "新发布未通过 readiness，自动 rollback" >&2
docker compose -f "$COMPOSE_FILE" logs --tail=200 >&2 || true
rollback
exit 1
