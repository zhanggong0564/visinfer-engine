import json
import os
from pathlib import Path
import subprocess

import pytest


HELPER = Path("scripts/release/deployment_compose.sh").resolve()


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("port", [None, "3007"])
def test_deployed_compose_uses_environment_port_without_editing_archive(tmp_path, legacy, port):
    source = tmp_path / "archived.yml"
    original = Path("docker-compose.scenes.yml").read_text()
    if legacy:
        original = original.replace("${SCENES_PORT:-3005}", "3005")
    source.write_text(original)
    target = tmp_path / "docker-compose.scenes.yml"
    if port:
        (tmp_path / ".env").write_text(f"SCENES_PORT={port}\n")
    env = os.environ.copy()
    env.pop("SCENES_PORT", None)
    subprocess.run(
        ["bash", "-euc", 'source "$1"; COMPOSE=(docker compose); '
         'install_deployment_compose "$2" "$3"', "bash", str(HELPER), str(source), str(target)],
        check=True, env=env, capture_output=True, text=True,
    )
    config = subprocess.run(
        ["docker", "compose", "-f", str(target), "config", "--format", "json"],
        check=True, env=env, capture_output=True, text=True,
    )
    service = json.loads(config.stdout)["services"]["mobile-vision-scenes"]
    assert service["ports"][0]["published"] == (port or "3005")
    assert service["ports"][0]["target"] == 3001
    assert service["healthcheck"]["test"][-1].endswith(":3001/health/ready")
    assert source.read_text() == original


@pytest.mark.parametrize("binding", ["0.0.0.0:3007", "[::]:3007", "127.0.0.1:3007"])
def test_readiness_uses_actual_published_port(binding):
    result = subprocess.run(
        ["bash", "-euc", 'source "$1"; fake_compose() { echo "$BINDING"; }; '
         'COMPOSE=(fake_compose); deployment_health_url compose.yml mobile-vision-scenes',
         "bash", str(HELPER)],
        env={**os.environ, "BINDING": binding}, check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "http://127.0.0.1:3007/health/ready"


@pytest.mark.parametrize("binding", ["", "0.0.0.0:0", "0.0.0.0:65536", "not-a-port", "a:3005\nb:3007"])
def test_readiness_rejects_missing_or_ambiguous_ports(binding):
    result = subprocess.run(
        ["bash", "-euc", 'source "$1"; fake_compose() { echo "$BINDING"; }; '
         'COMPOSE=(fake_compose); deployment_health_url compose.yml mobile-vision-scenes',
         "bash", str(HELPER)],
        env={**os.environ, "BINDING": binding}, capture_output=True, text=True,
    )
    assert result.returncode != 0
