#!/usr/bin/env bash
# Install all shared V4/V4.1 dependencies and build native operators.
set -Eeo pipefail
trap 'status=$?; printf "build-runtime.sh failed at line %s (exit %s)\n" "$LINENO" "$status" >&2' ERR
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
# Check editable installs and training-source integrity before emitting a receipt.
echo '[verify] Checking training-source integrity and editable installations'
"$PYTHON_BIN" -u -I - <<'PY_SOURCES'
import importlib.machinery
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path
sys.path.insert(0, '/tmp/ascend-image-build/tools')
from prepare_framework_sources import source_commit, verify
from prepare_image_sources import image_sources
root = Path('/opt/ascend-training/third_party')
packages = {'MindSpeed': 'mindspeed', 'MindSpeed-LLM': 'mindspeed_llm', 'mbridge': 'mbridge', 'verl': 'verl'}
for name, source in image_sources(Path('/tmp/ascend-image-build'))[1].items():
    checkout = root / name
    print(f"[verify] {name}: checking source inventory at {checkout}", flush=True)
    generated = []
    if name == 'MindSpeed-LLM':
        # setup.py builds this extension in-place during editable installation.
        generated = ['ascendspeed_te_ops' + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    try:
        verify(checkout, source, tracked=True, generated_files=generated)
    except Exception:
        # Print independently of exception hooks installed by editable/site packages.
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        raise SystemExit(1)
    actual = source_commit(checkout)
    assert actual == source['commit'], (name, actual)
    print(f"[verify] {name}: locating Python package {packages[name]}", flush=True)
    spec = importlib.util.find_spec(packages[name])
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(checkout):
        raise RuntimeError(f'Editable installation does not resolve to {checkout}: {spec}')
    print(f"[verify] {name}: passed", flush=True)
PY_SOURCES
# Fail on dependency drift even when a stale hash lock or no-deps mode is used.
echo '[verify] Checking runtime dependency versions and dataset imports'
"$PYTHON_BIN" -u -I - <<'PY_DATASETS'
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
print('[verify] Importing datasets', flush=True)
import datasets
print('[verify] Importing pyarrow', flush=True)
import pyarrow

print(f"Dataset imports verified: {actual}")
PY_DATASETS
# Report declared conflicts without blocking this compatibility deployment.
if ! "$PYTHON_BIN" -m pip check; then
    echo 'Dependency metadata conflicts reported above; continuing with the fixed deployment versions.' >&2
fi
mkdir -p "$ROOT/manifest"
echo '[verify] Recording runtime environment'
"$PYTHON_BIN" -u -I /tmp/ascend-image-build/collect_environment.py > "$ROOT/manifest/runtime-environment.json"
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
echo '[cleanup] Removing build-only materials'
"$PYTHON_BIN" -u - <<'PY_CLEANUP'
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
