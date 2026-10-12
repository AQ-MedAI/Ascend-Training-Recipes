#!/usr/bin/env bash
# Run once on the build node, before torchrun/Ray; never compile from every rank.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMMIT=1461cca857ff3b2018404c0275e80e7aa13a3fc9
BUILD_ROOT="${V41_MHC_BUILD_ROOT:-$ROOT/workspace/rollout/dsv41/build-src/mhc-premix}"
SOURCE="${V41_MHC_SOURCE:-$BUILD_ROOT/source}"
OPS=mhc_pre_sinkhorn_premix,mhc_pre_sinkhorn_premix_backward
mkdir -p "$BUILD_ROOT"
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh before building}"
"$PYTHON_BIN" "$ROOT/workspace/tests/dsv41/prepare_mhc_premix.py" --output "$SOURCE"
cd "$SOURCE"
# This is the upstream experimental build, not a second kernel implementation.
if [[ "${V41_MHC_SKIP_KERNEL_BUILD:-0}" != 1 ]]; then
# Ascend910_9382 uses the A3 package target, not ascend910b (A2).
bash build.sh --pkg --experimental --soc="${V41_MHC_SOC:-ascend910_93}" \
    --vendor_name=custom --ops="$OPS" -j"${V41_MHC_BUILD_JOBS:-8}"
fi
mapfile -t packages < <(find build_out -type f -name '*.run' | sort)
[[ "${#packages[@]}" == 1 ]] || { echo 'Expected one operator .run package under build_out' >&2; exit 1; }
bash "${packages[0]}" --quiet --install-path="$BUILD_ROOT/opp"
TORCH_EXTENSION_EXPERIMENTAL=TRUE "$PYTHON_BIN" experimental/tools/build_torch_vendor.py --ops "$OPS" --vendor custom \
    --output-dir "$BUILD_ROOT/wheels" --install-dir "$BUILD_ROOT/python"
"$PYTHON_BIN" - "$BUILD_ROOT/wheels" <<'VERIFY'
from pathlib import Path
import sys
import zipfile
wheels = list(Path(sys.argv[1]).glob('cann_ops_transformer_custom-*.whl'))
if len(wheels) != 1:
    raise RuntimeError('Expected one custom wheel')
with zipfile.ZipFile(wheels[0]) as wheel:
    names = set(wheel.namelist())
    builder = wheel.read('cann_ops_transformer_custom/op_builder/builder.py').decode()
    if 'Library("cann_ops_transformer", "DEF")' in builder:
        raise RuntimeError('Premix must use FRAGMENT to coexist with CANN 9.1')
    for op in ('mhc_pre_sinkhorn_premix', 'mhc_pre_sinkhorn_premix_backward'):
        required = {f'cann_ops_transformer_custom/ops/mhc/{op}/{op}.py',
                    f'cann_ops_transformer_custom/csrc/mhc/{op}.cpp'}
        if not required <= names:
            raise RuntimeError(f'Incomplete Premix wheel: {sorted(required - names)}')
VERIFY
mapfile -t vendors < <(find "$BUILD_ROOT/opp" -type d -path '*/vendors/custom_transformer')
[[ "${#vendors[@]}" == 1 ]] || { echo 'Cannot locate installed custom vendor' >&2; exit 1; }
VENDOR="${vendors[0]}"
# The environment must be sourced before Python starts (dynamic loader paths).
{
    printf 'export PYTHONPATH=%q:"${PYTHONPATH:-}"\n' "$BUILD_ROOT/python"
    printf 'export ASCEND_CUSTOM_OPP_PATH=%q:"${ASCEND_CUSTOM_OPP_PATH:-}"\n' "$VENDOR"
    printf 'export LD_LIBRARY_PATH=%q:"${LD_LIBRARY_PATH:-}"\n' "$VENDOR/op_api/lib"
    printf 'export V41_MHC_PRE_BACKEND=premix\n'
} > "$BUILD_ROOT/env.sh"
printf "Premix build complete; NPU validation runs separately.\n"
