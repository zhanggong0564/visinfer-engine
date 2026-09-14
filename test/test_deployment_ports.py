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


def test_invalid_port_does_not_replace_deployment_compose(tmp_path):
    source = tmp_path / "archive.yml"
    source.write_text(Path("docker-compose.scenes.yml").read_text())
    target = tmp_path / "docker-compose.scenes.yml"
    target.write_text("original configuration\n")
    result = subprocess.run(
        ["bash", "-euc", 'source "$1"; COMPOSE=(docker compose); '
         'install_deployment_compose "$2" "$3"', "bash", str(HELPER), str(source), str(target)],
        env={**os.environ, "SCENES_PORT": "invalid"}, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert target.read_text() == "original configuration\n"


@pytest.mark.parametrize("operation", ["activate", "failed_activate", "rollback"])
def test_release_workflows_keep_production_port_with_historical_compose(tmp_path, operation):
    root = tmp_path / "deploy"
    root.mkdir()
    (root / ".env").write_text("SCENES_PORT=3007\n")
    original = Path("docker-compose.scenes.yml").read_text().replace("${SCENES_PORT:-3005}", "3005")
    for release in ("old", "new.staging"):
        directory = root / "releases" / release
        (directory / "pkg").mkdir(parents=True)
        (directory / "weights").mkdir()
        (directory / "app.py").touch()
        (directory / "weight-paths.txt").touch()
        (directory / "docker-compose.scenes.yml").write_text(original)
    (root / "current").symlink_to("releases/old")
    (root / "docker-compose.scenes.yml").write_text(original)
    if operation == "rollback":
        (root / "releases/new.staging").rename(root / "releases/new")
        (root / "current").unlink()
        (root / "current").symlink_to("releases/new")
        (root / "previous").symlink_to("releases/old")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "ssh": '#!/bin/bash\nshift; exec "$@"\n',
        "sleep": '#!/bin/bash\nexit 0\n',
        "curl": '#!/bin/bash\n[[ "$*" == *":3007/health/ready" ]] || exit 22\n'
                'echo "$*" >> "$TEST_ROOT/health-calls"\n'
                'if [ "$FAIL_NEW" = 1 ] && [ "$(readlink current)" = releases/new ]; then exit 22; fi\n',
        "docker": '#!/bin/bash\nset -eu\n'
                  'case "$*" in\n'
                  ' "compose version") exit 0 ;;\n'
                  ' *Config.Image*) echo test-image ;;\n'
                  ' *requirements-sha256*) echo requirements-sha ;;\n'
                  ' *python-abi*) echo cp310 ;;\n'
                  ' *environment-contract-sha256*) echo environment-sha ;;\n'
                  ' "run "*) exit 0 ;;\n'
                  ' *"config --quiet"*) exit 0 ;;\n'
                  ' *"up -d --force-recreate"*) '
                  'grep -Fq \'${SCENES_PORT:-3005}:3001\' docker-compose.scenes.yml ;;\n'
                  ' *"port mobile-vision-scenes 3001"*) echo 0.0.0.0:3007 ;;\n'
                  ' *"logs --tail=200"*) exit 0 ;;\n'
                  ' *) echo "unexpected docker call: $*" >&2; exit 99 ;;\n'
                  'esac\n',
    }
    for name, source in scripts.items():
        script = bin_dir / name
        script.write_text(source)
        script.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "TEST_ROOT": str(root), "FAIL_NEW": str(int(operation == "failed_activate"))}
    if operation == "rollback":
        command = ["bash", "scripts/release/rollback-plugin.sh", "--remote", "mock-host",
                   "--remote-dir", str(root), "--service", "scenes"]
    else:
        command = ["bash", "scripts/release/remote_activate.sh", str(root), "new",
                   "docker-compose.scenes.yml", "mobile-vision-scenes",
                   "http://127.0.0.1:3005/health/ready", "requirements-sha", "cp310",
                   "runtime-sha", "0", "dc_fuse", "environment-sha", "0"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == (1 if operation == "failed_activate" else 0), result.stderr
    expected = "releases/new" if operation == "activate" else "releases/old"
    assert os.readlink(root / "current") == expected
    assert (root / "health-calls").read_text()
    assert (root / "releases/old/docker-compose.scenes.yml").read_text() == original
    assert (root / ".env").read_text() == "SCENES_PORT=3007\n"
