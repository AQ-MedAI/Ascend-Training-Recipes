#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
# Source paths are explicit for both interactive commands and the training launchers.
export PYTHONPATH="$MINDSPEED_ROOT:$MEGATRON_BRIDGE_ROOT/src:$MEGATRON_ROOT:$VERL_ROOT:$VLLM_ASCEND_ROOT:$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f "$RECIPE_ROOT/recipes/kimi-k2.6/runtime_env.sh" ]]; then
    printf 'Missing K26 recipe: %s\n' "$RECIPE_ROOT" >&2
    exit 1
fi
exec "$@"
