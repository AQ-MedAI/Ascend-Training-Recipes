#!/usr/bin/env bash
# Install all shared V4/V4.1 dependencies and build native operators.
set -eo pipefail
ROOT=/opt/ascend-training
SOURCES=$ROOT/third_party
INPUT=/tmp/ascend-image-build/locks
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
        --constraint /tmp/ascend-image-build/hardware-constraints.txt -r "$INPUT/runtime.lock"
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
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$SOURCES/Megatron-LM"
rollout="$ROOT/workspace/rollout/dsv41"
mkdir -p "$rollout/build-src"
export VLLM_ROOT="$SOURCES/vllm" VLLM_ASCEND_ROOT="$SOURCES/vllm-ascend"
download_args=()
[[ "${DEPENDENCY_OFFLINE:-0}" != 1 ]] || download_args+=(--offline)
"$PYTHON_BIN" /tmp/ascend-image-build/download_dependencies.py --workspace "$ROOT/workspace" "${download_args[@]}"
"$PYTHON_BIN" "$rollout/prepare_native_deps.py"
build_root="$rollout/build-src"
mkdir -p "$ROOT/manifest/build-logs"
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.23.0 \
    VLLM_REQUIRE_RUST_FRONTEND=0 "$PYTHON_BIN" -m pip install \
    --no-deps --no-build-isolation -e "$VLLM_ROOT"
export PYTHONPATH="$VLLM_ROOT:$VLLM_ASCEND_ROOT"
SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0 COMPILE_CUSTOM_KERNELS=1 \
    "$PYTHON_BIN" -m pip install -v \
    --no-deps --no-build-isolation -e "$VLLM_ASCEND_ROOT" 2>&1 | tee "$ROOT/manifest/build-logs/vllm-ascend.log"
unset PYTHONPATH
export V41_MHC_SOC=ascend910_93 V41_MHC_BUILD_JOBS="${MAX_JOBS:-4}"
bash workspace/tests/dsv41/build_mhc_premix.sh
"$PYTHON_BIN" -m pip install --no-index --no-deps \
    "$build_root"/mhc-premix/wheels/cann_ops_transformer_custom-*.whl
export ROLLOUT_PYTHON="$PYTHON_BIN"
source workspace/tests/dsv41/native_operator_env.sh
test -s "$V41_CSA2_NATIVE_LIBRARY"
test -s "$build_root/mhc-premix/env.sh"
# Install shared training sources after the framework/operator layer.
export PYTHONPATH="$SOURCES/MindSpeed:$SOURCES/MindSpeed-LLM:$SOURCES/mbridge:$SOURCES/verl:$SOURCES/Megatron-LM:$VLLM_ASCEND_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"
for name in MindSpeed MindSpeed-LLM mbridge verl; do
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$SOURCES/$name"
done
# Check editable installs and training-source integrity before emitting a receipt.
"$PYTHON_BIN" -I - <<'PY_SOURCES'
import importlib.util
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, '/tmp/ascend-image-build/tools')
from prepare_framework_sources import source_commit, verify
from prepare_image_sources import image_sources
root = Path('/opt/ascend-training/third_party')
packages = {'MindSpeed': 'mindspeed', 'MindSpeed-LLM': 'mindspeed_llm', 'mbridge': 'mbridge', 'verl': 'verl'}
for name, source in image_sources(Path('/tmp/ascend-image-build'))[1].items():
    checkout = root / name
    verify(checkout, source, tracked=True)
    actual = source_commit(checkout)
    assert actual == source['commit'], (name, actual)
    spec = importlib.util.find_spec(packages[name])
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(checkout):
        raise RuntimeError(f'Editable installation does not resolve to {checkout}: {spec}')
PY_SOURCES
# Fail on dependency drift even when a stale hash lock or no-deps mode is used.
"$PYTHON_BIN" -I - <<'PY_DATASETS'
from importlib.metadata import version

expected = {"datasets": "5.0.1", "pyarrow": "24.0.0", "numpy": "1.26.4"}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"Dataset dependency versions differ: expected {expected}, got {actual}")
from importlib.metadata import PackageNotFoundError
from pathlib import Path

mismatches = []
for line in Path("/tmp/ascend-image-build/locks/validated-runtime-constraints.txt").read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    name, wanted = line.split("==", 1)
    try:
        installed = version(name)
    except PackageNotFoundError:
        continue  # A constraint does not require an optional package to be installed.
    if installed != wanted:
        mismatches.append(f"{name}: expected {wanted}, got {installed}")
if mismatches:
    raise RuntimeError("Validated runtime version drift:\n" + "\n".join(mismatches))
import datasets
import pyarrow

print(f"Dataset imports verified: {actual}")
PY_DATASETS
# Report declared conflicts without blocking this compatibility deployment.
if ! "$PYTHON_BIN" -m pip check; then
    echo 'Dependency metadata conflicts reported above; continuing with the fixed deployment versions.' >&2
fi
mkdir -p "$ROOT/manifest"
"$PYTHON_BIN" -I /tmp/ascend-image-build/collect_environment.py > "$ROOT/manifest/runtime-environment.json"
"$PYTHON_BIN" - <<'PY'
import json
import subprocess
from pathlib import Path
root = Path('/tmp/ascend-image-build')
manifest = Path('/opt/ascend-training/manifest')
expected = {'torch':'2.10.0', 'torch-npu':'2.10.0.post4', 'triton-ascend':'3.2.2', 'triton':'3.5.0'}
report = json.loads((manifest / 'runtime-environment.json').read_text())
packages = {p['name'].lower().replace('_', '-'):p['version'].split('+')[0] for p in report['packages']}
expected.update({'numpy': '1.26.4', 'transformers': '5.8.1', 'fastapi': '0.123.10',
                 'starlette': '0.50.0', 'opencv-python-headless': '5.0.0.93',
                 'prometheus-fastapi-instrumentator': '8.1.0', 'mcp': '2.2.0',
                 'model-hosting-container-standards': '0.1.16'})
assert all(packages.get(k) == v for k,v in expected.items()), packages
registry = json.loads((root/'dependencies/registry.json').read_text())['rollout']['v41']
receipt = {'profile':'v4-v41', 'python_environment':'single', 'vllm':registry['vllm']['commit'], 'vllm-ascend':registry['vllm_ascend']['commit'],
           'megatron':json.loads((root/'dependencies/framework-sources.json').read_text())['sources']['Megatron-LM']}
receipt['sources'] = json.loads((root/'dependencies/framework-sources.json').read_text())
import sys
sys.path.insert(0, str(root / 'tools'))
from prepare_image_sources import image_sources
receipt['recipe_sources'] = image_sources(root)[1]
receipt['source_root'] = '/opt/ascend-training/third_party'

(manifest/'frameworks.json').write_text(json.dumps(receipt, indent=2)+'\n')
PY

# Keep provenance and licenses; remove build-only inputs after successful validation.
"$PYTHON_BIN" - <<'PY_CLEANUP'
import json
import shutil
from pathlib import Path
build = Path('/tmp/ascend-image-build')
root = Path('/opt/ascend-training')
for name in ('registry.json', 'native-dependencies.json', 'Megatron-LM-LICENSE'):
    shutil.copy2((build / 'dependencies' / name) if name == 'registry.json' else (build / name), root / 'manifest' / name)
# Preserve the same relative layout so the receipt can reconstruct sources too.
shutil.copytree(build / 'dependencies', root / 'manifest/dependencies', dirs_exist_ok=True)
shutil.copytree(build / 'patches', root / 'manifest/patches', dirs_exist_ok=True)
shutil.copytree(build / 'tools', root / 'manifest/tools', dirs_exist_ok=True)
workspace = root / 'workspace'
mhc = workspace / 'rollout/dsv41/build-src/mhc-premix'
# Preserve installed operators and Python wrappers; remove only build inputs.
for runtime_dir in (mhc / 'opp', mhc / 'python'):
    for link in runtime_dir.rglob('*'):
        if link.is_symlink() and any(link.resolve().is_relative_to(mhc / name) for name in ('source', 'wheels')):
            raise RuntimeError(f'Runtime link still depends on build inputs: {link}')
source = mhc / 'source'
# Retain third-party notices before dropping the unpacked build tree.
for notice in source.rglob('*'):
    if notice.is_file() and not notice.is_symlink() and notice.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
        target = root / 'manifest/licenses/mhc-premix' / notice.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notice, target)
for name in ('source', 'wheels'):
    path = mhc / name
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
for asset in json.loads((build / 'native-dependencies.json').read_text())['assets']:
    path = (workspace / asset['path']).resolve()
    if not path.is_relative_to(workspace.resolve()):
        raise ValueError(f'Archive outside workspace: {path}')
    if path.name.endswith(('.tar.gz', '.tgz', '.tar.xz', '.zip')):
        path.unlink(missing_ok=True)
for name in ('prepare_mhc_premix.py', 'build_mhc_premix.sh', 'native_operator_env.sh'):
    (workspace / 'tests/dsv41' / name).unlink(missing_ok=True)
(workspace / 'rollout/dsv41/prepare_native_deps.py').unlink(missing_ok=True)
shutil.rmtree(build)
# Prune only empty helper/material directories, never runtime directories.
for base in (workspace / 'tests', workspace / 'third_party/mhc-premix'):
    if base.is_dir():
        for path in sorted(base.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
                path.rmdir()
        if not any(base.iterdir()):
            base.rmdir()
PY_CLEANUP
