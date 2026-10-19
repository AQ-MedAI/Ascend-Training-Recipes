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
BUILD_RUNTIME_PHASE="${BUILD_RUNTIME_PHASE:-all}"
case "$BUILD_RUNTIME_PHASE" in
    all|build|verify) ;;
    *) echo 'BUILD_RUNTIME_PHASE must be all, build or verify' >&2; exit 2 ;;
esac
if [[ "$BUILD_RUNTIME_PHASE" != verify ]]; then
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
fi
# Keep compiled installations in a successful layer before validation/cleanup.
if [[ "$BUILD_RUNTIME_PHASE" == build ]]; then
    exit 0
fi
# Default all mode remains available for manual builds.
exec bash /tmp/ascend-image-build/verify-runtime.sh
