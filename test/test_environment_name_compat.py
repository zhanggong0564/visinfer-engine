"""Known image aliases must not weaken runtime or activation checks."""
import os
import subprocess
from pathlib import Path

import pytest

CONTRACT_SCRIPT = Path('scripts/release/compute_environment_contract.sh').resolve()
ACTIVATE_SCRIPT = Path('scripts/release/remote_activate.sh').resolve()
ORT = 'whl/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
RUNTIME = '''ARG BASE_IMAGE=mobile_vision/runtime-base:latest
ARG BUILDER_IMAGE=mobile_vision/build-base:latest
FROM ${BUILDER_IMAGE} AS plugin-builder
FROM ${BASE_IMAGE} AS runtime
RUN install-plugins
'''


@pytest.fixture
def contract_root(tmp_path):
    for name, content in {
        'Dockerfile.base': 'FROM cuda\nRUN install-python\n',
        'Dockerfile.runtime': RUNTIME,
        'requirements.txt': 'numpy==1.26.4\n',
        'requirements.scenes.txt': 'chromadb==1.5.9\n',
        ORT: 'fixed-wheel-bytes',
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def contract(root, legacy=False, env_updates=None):
    env = {**os.environ, 'VIE_CONTRACT_ROOT': str(root), **(env_updates or {})}
    result = subprocess.run(
        ['bash', str(CONTRACT_SCRIPT), *(['--legacy-image-names'] if legacy else [])],
        env=env, text=True, capture_output=True,
    )
    return result


def test_only_known_default_aliases_reproduce_old_contract(contract_root):
    current = contract(contract_root).stdout.strip()
    compatible = contract(contract_root, legacy=True)
    assert compatible.returncode == 0
    path = contract_root / 'Dockerfile.runtime'
    path.write_text(RUNTIME.replace('mobile_vision/runtime-base:latest', 'mobile_vision:base')
                   .replace('mobile_vision/build-base:latest', 'mobile_vision:base-builder'))
    old = contract(contract_root).stdout.strip()
    assert old != current
    assert compatible.stdout.strip() == old
    assert contract(contract_root, legacy=True).stdout.strip() == old


@pytest.mark.parametrize('changed_file', [
    'Dockerfile.base', 'Dockerfile.runtime', 'requirements.txt', 'requirements.scenes.txt', ORT,
])
def test_legacy_alias_contract_still_tracks_every_runtime_input(contract_root, changed_file):
    before = contract(contract_root, legacy=True).stdout.strip()
    path = contract_root / changed_file
    path.write_text(path.read_text() + '\nchanged-runtime\n')
    after = contract(contract_root, legacy=True)
    assert after.returncode == 0 and after.stdout.strip() != before


def test_legacy_alias_contract_still_tracks_cuda_image(contract_root):
    before = contract(contract_root, legacy=True).stdout.strip()
    after = contract(contract_root, legacy=True, env_updates={'CUDA_BASE_IMAGE': 'different-cuda'})
    assert after.returncode == 0 and after.stdout.strip() != before


@pytest.mark.parametrize('change', [
    ('mobile_vision/runtime-base:latest', 'custom/runtime:latest'),
    ('mobile_vision/build-base:latest', 'custom/build:latest'),
    ('ARG BASE_IMAGE=mobile_vision/runtime-base:latest\n', ''),
    ('ARG BUILDER_IMAGE=mobile_vision/build-base:latest\n', ''),
    ('ARG BASE_IMAGE=mobile_vision/runtime-base:latest',
     'ARG BASE_IMAGE=mobile_vision/runtime-base:latest\nARG BASE_IMAGE=mobile_vision:base'),
])
def test_unknown_or_ambiguous_aliases_are_rejected(contract_root, change):
    (contract_root / 'Dockerfile.runtime').write_text(RUNTIME.replace(*change))
    assert contract(contract_root, legacy=True).returncode != 0


@pytest.mark.parametrize('requirements,abi,image_contract,compatible,allowed', [
    ('requirements-sha', 'cp310', 'current-contract', '', True),
    ('requirements-sha', 'cp310', 'old-alias-contract', 'old-alias-contract', True),
    ('requirements-sha', 'cp310', 'unrelated-contract', 'old-alias-contract', False),
    ('requirements-sha', 'cp310', 'old-alias-contract', '', False),
    ('requirements-sha', 'cp310', '<no value>', 'old-alias-contract', False),
    ('changed-requirements', 'cp310', 'old-alias-contract', 'old-alias-contract', False),
    ('requirements-sha', 'cp311', 'old-alias-contract', 'old-alias-contract', False),
])
def test_activation_alias_allowance_preserves_guards_and_preflight(
    tmp_path, requirements, abi, image_contract, compatible, allowed,
):
    root = tmp_path / 'deploy'
    stage = root / 'releases/release-1.staging'
    (stage / 'pkg').mkdir(parents=True)
    for name in ('app.py', 'docker-compose.panel-label.yml', 'weight-paths.txt'):
        (stage / name).touch()
    (root / 'current').symlink_to('releases/old-release')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    docker = binaries / 'docker'
    docker.write_text('''#!/usr/bin/env bash
if [ "$1 $2" = 'compose version' ]; then exit 0; fi
case "$*" in
  *Config.Image*) echo production-image ;;
  *requirements-sha256*) echo "$TEST_REQUIREMENTS" ;;
  *python-abi*) echo "$TEST_ABI" ;;
  *environment-contract-sha256*) echo "$TEST_CONTRACT" ;;
  run*) echo PREFLIGHT_REACHED; exit 99 ;;
  *) exit 1 ;;
esac
''')
    docker.chmod(0o755)
    env = {**os.environ, 'PATH': str(binaries)+':'+os.environ['PATH'],
           'TEST_REQUIREMENTS': requirements, 'TEST_ABI': abi, 'TEST_CONTRACT': image_contract}
    result = subprocess.run([
        'bash', str(ACTIVATE_SCRIPT), str(root), 'release-1', 'docker-compose.panel-label.yml',
        'mobile-vision-panel-label', 'http://127.0.0.1:3001/health/ready',
        'requirements-sha', 'cp310', 'runtime-sha', '1', 'panel_label', 'current-contract', '0', compatible,
    ], capture_output=True, text=True, env=env)
    assert ('PREFLIGHT_REACHED' in result.stdout) is allowed
    assert result.returncode != 0  # A failed preflight must leave the live pointer untouched.
    assert (root / 'current').readlink() == Path('releases/old-release')
    assert stage.is_dir() and not (root / 'releases/release-1').exists()
