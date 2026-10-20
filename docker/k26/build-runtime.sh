#!/usr/bin/env bash
# Reuse compiled rollout dependencies and install the pinned K26 training sources.
set -Eeo pipefail
trap 'status=$?; printf "K26 build failed at line %s (exit %s)\n" "$LINENO" "$status" >&2' ERR
ROOT=/opt/ascend-training
SOURCES=$ROOT/third_party
INPUT=/tmp/k26-image-build
PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.15/bin/python3}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
set -u
unset PYTHONPATH PYTHONHOME
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
"$PYTHON_BIN" "$INPUT/verify_runtime.py" --base
# These editable installations are replaced together; inference packages remain installed.
"$PYTHON_BIN" -m pip uninstall -y megatron-core megatron-bridge mindspeed verl mindspeed-llm mbridge
for name in Megatron-LM Megatron-Bridge MindSpeed verl MindSpeed-LLM mbridge; do
    rm -rf "$SOURCES/$name"
done
"$PYTHON_BIN" "$INPUT/tools/prepare_k26_image_sources.py" --root "$SOURCES" --inputs "$INPUT"
for name in Megatron-LM Megatron-Bridge MindSpeed verl; do
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$SOURCES/$name"
done
