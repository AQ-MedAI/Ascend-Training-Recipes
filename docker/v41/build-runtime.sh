#!/usr/bin/env bash
# Build only image-owned frameworks and native operators. No recipe runtime or forks are installed.
set -eo pipefail
ROOT=/opt/ascend-training
INPUT=/opt/v41-image/locks
# Check installation inputs before fetching or compiling.
for lock in build; do
    [[ -s "$INPUT/$lock.lock" ]] || {
        printf 'Missing dependency lock: %s/%s.lock; see docker/v41/locks/README.md.\n' "$INPUT" "$lock" >&2
        exit 1
    }
done
cd "$ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
unset PYTHONPATH PYTHONHOME
export PYTHON_BIN="$(command -v python3)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# Build tools have their own lock; runtime locks include framework dependencies.
"$PYTHON_BIN" -m pip install --require-hashes --only-binary=:all: -r "$INPUT/build.lock"
# Prefer the full hash lock when available. Development builds may use fixed inputs.
if [[ -s "$INPUT/runtime.lock" ]]; then
    uv pip install --python "$PYTHON_BIN" --require-hashes --no-deps \
        --constraint /opt/v41-image/hardware-constraints.txt -r "$INPUT/runtime.lock"
else
    install_args=()
    case "${PYTHON_DEPENDENCY_MODE:-resolve}" in
        resolve) ;;
        no-deps) install_args+=(--no-deps)
            echo 'Installing direct runtime requirements only; transitive dependencies may be missing.' >&2 ;;
        *) echo 'PYTHON_DEPENDENCY_MODE must be resolve or no-deps' >&2; exit 2 ;;
    esac
    uv pip install --python "$PYTHON_BIN" \
        --overrides "$INPUT/runtime-overrides.txt" \
        ${install_args[@]+"${install_args[@]}"} -r "$INPUT/runtime.in"
fi
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$ROOT/Megatron-LM"
rollout="$ROOT/workspace/rollout/dsv41"
mkdir -p "$rollout/build-src"
"$PYTHON_BIN" - <<'PY'
import json
import subprocess
from pathlib import Path
registry = json.loads(Path('/opt/v41-image/registry.json').read_text())['rollout']['v41']
root = Path('/opt/ascend-training/workspace/rollout/dsv41/build-src')
for name, key in [('vllm', 'vllm'), ('vllm-ascend', 'vllm_ascend')]:
    source = registry[key]
    path = root / name
    subprocess.run(['git', 'init', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', source['repository'] + '.git'], check=True)
    subprocess.run(['git', '-C', str(path), 'fetch', '--depth=1', 'origin', source['commit']], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    assert subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() == source['commit']
PY
download_args=()
[[ "${DEPENDENCY_OFFLINE:-0}" != 1 ]] || download_args+=(--offline)
"$PYTHON_BIN" /opt/v41-image/download_dependencies.py --workspace "$ROOT/workspace" "${download_args[@]}"
"$PYTHON_BIN" "$rollout/prepare_native_deps.py"
build_root="$rollout/build-src"
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.1.dev1+ga97dacb7106ee.empty \
    VLLM_REQUIRE_RUST_FRONTEND=0 "$PYTHON_BIN" -m pip install \
    --no-deps --no-build-isolation -e "$build_root/vllm"
export PYTHONPATH="$build_root/vllm:$build_root/vllm-ascend"
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev1+ge67ab649500f COMPILE_CUSTOM_KERNELS=1 \
    "$PYTHON_BIN" -m pip install \
    --no-deps --no-build-isolation -e "$build_root/vllm-ascend"
unset PYTHONPATH
export V41_MHC_SOC=ascend910_93 V41_MHC_BUILD_JOBS="${MAX_JOBS:-4}"
bash workspace/tests/dsv41/build_mhc_premix.sh
"$PYTHON_BIN" -m pip install --no-index --no-deps \
    "$build_root"/mhc-premix/wheels/cann_ops_transformer_custom-*.whl
export ROLLOUT_PYTHON="$PYTHON_BIN"
source workspace/tests/dsv41/native_operator_env.sh
test -s "$V41_CSA2_NATIVE_LIBRARY"
test -s "$build_root/mhc-premix/env.sh"
# Report declared conflicts without blocking this compatibility deployment.
if ! "$PYTHON_BIN" -m pip check; then
    echo 'Dependency metadata conflicts reported above; continuing with the fixed deployment versions.' >&2
fi
"$PYTHON_BIN" -I /opt/v41-image/collect_environment.py > /opt/v41-image/runtime-environment.json
"$PYTHON_BIN" - <<'PY'
import json
import subprocess
from pathlib import Path
root = Path('/opt/v41-image')
expected = {'torch':'2.10.0', 'torch-npu':'2.10.0.post4', 'triton-ascend':'3.2.2', 'triton':'3.5.0'}
report = json.loads((root / 'runtime-environment.json').read_text())
packages = {p['name'].lower().replace('_', '-'):p['version'].split('+')[0] for p in report['packages']}
expected.update({'numpy': '1.26.4', 'transformers': '5.14.1', 'fastapi': '0.123.10',
                 'starlette': '0.50.0', 'opencv-python-headless': '5.0.0.93',
                 'prometheus-fastapi-instrumentator': '8.1.0', 'mcp': '2.2.0',
                 'model-hosting-container-standards': '0.1.16'})
assert all(packages.get(k) == v for k,v in expected.items()), packages
registry = json.loads((root/'registry.json').read_text())['rollout']['v41']
receipt = {'profile':'v41', 'python_environment':'single', 'vllm':registry['vllm']['commit'], 'vllm-ascend':registry['vllm_ascend']['commit'],
           'megatron':json.loads((root/'megatron-source.json').read_text())}
(root/'frameworks.json').write_text(json.dumps(receipt, indent=2)+'\n')
PY
