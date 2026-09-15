import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


@pytest.fixture
def offline_host(tmp_path):
    """Use real Compose validation, but mock every container/image operation."""
    real_docker = shutil.which("docker")
    assert real_docker, "Docker Compose is required for deployment contract tests"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("deploy_offline.sh", "deployment_compose.sh"):
        shutil.copyfile(Path("scripts/release") / name, bundle / name)
    (bundle / "image.tar.gz").write_bytes(gzip.compress(b"mock image archive"))
    for service, image_var in [("scenes", "SCENES_IMAGE"), ("panel-label", "PANEL_LABEL_IMAGE")]:
        package = bundle / service
        package.mkdir()
        shutil.copyfile(f"docker-compose.{service}.yml", package / f"docker-compose.{service}.yml")
        (package / "release.env").write_text(
            f"RELEASE_VERSION=9.0.1\n{image_var}=example/{service}:9.0.1\n"
            f"COMPOSE_FILE=docker-compose.{service}.yml\n"
            "HEALTH_URL=http://127.0.0.1:3005/health/ready\n"
            "INDICATOR_ALLOWED_HOSTS=package.example\nNEW_SETTING=package-default\n"
        )
        with tarfile.open(package / "overlay.tar.gz", "w:gz") as archive:
            archive.add(package / f"docker-compose.{service}.yml", arcname=f"docker-compose.{service}.yml")
    sums = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(bundle)}\n")
    (bundle / "SHA256SUMS").write_text("".join(sums))
    root = tmp_path / "deploy"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        '#!/bin/bash\nset -eu\n'
        'echo "$*" >> "$TEST_ROOT/docker-calls"\n'
        'if [ "$1" = compose ]; then\n'
        ' if [[ "$*" == *"logs --tail"* ]]; then exit 0\n'
        ' elif [[ "$*" == *"up -d"* ]]; then\n'
        '  "$REAL_DOCKER" compose -f "$3" config --format json > "$TEST_ROOT/resolved.json"\n'
        ' else exec "$REAL_DOCKER" "$@"; fi\n'
        'elif [ "$1" = load ]; then cat >/dev/null\n'
        'elif [ "$1" = port ]; then\n'
        ' echo "$TEST_PORT/tcp -> 0.0.0.0:$TEST_PORT"\n'
        ' echo "$TEST_PORT/tcp -> [::]:$TEST_PORT"\n'
        'else echo "Unexpected Docker mutation: $*" >&2; exit 99; fi\n'
    )
    docker.chmod(0o755)
    for name, script in {
        "chown": '#!/bin/bash\nexit 0\n',
        "sleep": '#!/bin/bash\nexit 0\n',
        "curl": '#!/bin/bash\necho "$*" >> "$TEST_ROOT/curl-calls"\n'
                '[[ "$*" == *"http://127.0.0.1:$TEST_PORT/health/ready"* ]] || exit 22\n'
                '[ "${TEST_HEALTH_FAIL:-0}" = 0 ]\n',
    }.items():
        path = bin_dir / name
        path.write_text(script)
        path.chmod(0o755)
    env = {**os.environ, "REAL_DOCKER": real_docker, "TEST_ROOT": str(root),
           "PATH": f"{bin_dir}:{os.environ['PATH']}", "TEST_PORT": "3007"}
    for key in ("SCENES_PORT", "SCENES_IMAGE", "PANEL_LABEL_IMAGE", "INDICATOR_ALLOWED_HOSTS"):
        env.pop(key, None)

    def deploy(service="scenes"):
        return subprocess.run(
            ["bash", str(bundle / "deploy_offline.sh"), "--bundle", "bundle",
             "--service", service, "--deploy-dir", "deploy"],
            cwd=tmp_path, env=env, capture_output=True, text=True,
        )

    return root, bundle, env, deploy


def test_offline_upgrade_preserves_site_settings_and_updates_metadata(offline_host):
    root, bundle, env, deploy = offline_host
    original = (
        "RELEASE_VERSION=old\nSCENES_IMAGE=example/scenes:old\n"
        "HEALTH_URL=http://127.0.0.1:3005/health/ready\n"
        "SCENES_PORT=3007\nINDICATOR_ALLOWED_HOSTS='site.example, ref.example'\n"
        "EMPTY_SETTING=\nLITERAL='$(touch should-not-exist)'\n"
    )
    (root / ".env").write_text(original)
    (root / "releases/old").mkdir(parents=True)
    (root / "current").symlink_to("releases/old")
    result = deploy()
    assert result.returncode == 0, result.stderr
    service = json.loads((root / "resolved.json").read_text())["services"]["mobile-vision-scenes"]
    assert service["image"] == "example/scenes:9.0.1"
    assert service["ports"][0]["published"] == "3007"
    assert service["ports"][0]["target"] == 3007
    assert service["environment"]["PORT"] == "3007"
    assert service["environment"]["INDICATOR_ALLOWED_HOSTS"] == "site.example, ref.example"
    text = (root / ".env").read_text()
    assert "NEW_SETTING=package-default" in text and "EMPTY_SETTING=" in text
    assert text.count("HEALTH_URL=") == 1 and "HEALTH_URL=http://127.0.0.1:3007/health/ready" in text
    assert "RELEASE_VERSION=9.0.1" in text and "SCENES_IMAGE=example/scenes:9.0.1" in text
    assert (root / "previous").readlink().as_posix() == "releases/old"
    backups = list((root / ".release-backups").iterdir())
    assert len(backups) == 1 and (backups[0] / ".env").read_text() == original
    assert not list(root.parent.rglob("should-not-exist"))
    assert "3005/health" not in (root / "curl-calls").read_text()


@pytest.mark.parametrize("service,port", [("scenes", "3005"), ("panel-label", "3001")])
def test_offline_first_install_uses_service_defaults(offline_host, service, port):
    root, bundle, env, deploy = offline_host
    env["TEST_PORT"] = port
    result = deploy(service)
    assert result.returncode == 0, result.stderr
    assert f"HEALTH_URL=http://127.0.0.1:{port}/health/ready" in (root / ".env").read_text()
    config = json.loads((root / "resolved.json").read_text())["services"][f"mobile-vision-{service}"]
    assert config["ports"][0]["published"] == port
    assert config["ports"][0]["target"] == int(port)
    assert not (root / "previous").exists()


def test_invalid_site_port_stops_before_loading_images_or_replacing_config(offline_host):
    root, bundle, env, deploy = offline_host
    original = "SCENES_PORT=invalid\nSCENES_IMAGE=example/scenes:old\n"
    (root / ".env").write_text(original)
    (root / "docker-compose.scenes.yml").write_text("old compose\n")
    result = deploy()
    assert result.returncode != 0
    assert (root / ".env").read_text() == original
    assert (root / "docker-compose.scenes.yml").read_text() == "old compose\n"
    assert "\nload\n" not in (root / "docker-calls").read_text()
    assert not (root / "current").exists()


def test_offline_rejects_tampered_helper(offline_host):
    root, bundle, env, deploy = offline_host
    with (bundle / "deployment_compose.sh").open("a") as stream:
        stream.write("\necho tampered\n")
    result = deploy()
    assert result.returncode != 0
    assert not (root / ".env").exists()
    assert "load" not in (root / "docker-calls").read_text()


def test_offline_does_not_overwrite_an_existing_release(offline_host):
    root, bundle, env, deploy = offline_host
    release = root / "releases/9.0.1"
    release.mkdir(parents=True)
    (release / "marker").write_text("retain")
    result = deploy()
    assert result.returncode != 0 and "发布目录已存在" in result.stderr
    assert (release / "marker").read_text() == "retain"
    assert "load" not in (root / "docker-calls").read_text()


def test_offline_readiness_failure_keeps_backup_and_returns_failure(offline_host):
    root, bundle, env, deploy = offline_host
    (root / ".env").write_text("SCENES_PORT=3007\n")
    env["TEST_HEALTH_FAIL"] = "1"
    result = deploy()
    assert result.returncode != 0
    backup = next((root / ".release-backups").iterdir())
    assert (backup / ".env").read_text() == "SCENES_PORT=3007\n"
    assert "3005/health" not in (root / "curl-calls").read_text()


def test_release_bundle_checksums_deployment_scripts():
    script = Path("scripts/release/build_docker_release.sh").read_text()
    assert 'cp scripts/release/deployment_compose.sh "$OUT/deployment_compose.sh"' in script
    assert 'find deploy_offline.sh deployment_compose.sh image.tar.gz' in script
