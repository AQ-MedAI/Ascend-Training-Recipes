#!/usr/bin/env bash
# Source in the shell that starts Ray and training; repeat on every node.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source tools/activate.sh [v41|v4]' >&2
    exit 2
fi
_recipe_profile="${1:-v41}"
if [[ "$_recipe_profile" != v41 && "$_recipe_profile" != v4 ]]; then
    echo 'Supported profiles: v41, v4' >&2
    return 2
fi
export RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MINDSPEED_ROOT="$RECIPE_ROOT/third_party/MindSpeed"
export MINDSPEED_LLM_ROOT="$RECIPE_ROOT/third_party/MindSpeed-LLM"
export MBRIDGE_ROOT="$RECIPE_ROOT/third_party/mbridge"
export VERL_ROOT="$RECIPE_ROOT/third_party/verl"
export PYTHONPATH="$RECIPE_ROOT:$VERL_ROOT:$MINDSPEED_ROOT:$MINDSPEED_LLM_ROOT:$MBRIDGE_ROOT:${PYTHONPATH:-}"
if [[ "$_recipe_profile" == v41 ]]; then
    export DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-/opt/ascend-training}"
    export MEGATRON_ROOT="${MEGATRON_ROOT:-$DEPENDENCY_ROOT/Megatron-LM}"
    export VLLM_ROOT="${VLLM_ROOT:-$DEPENDENCY_ROOT/vllm}"
    export VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-$DEPENDENCY_ROOT/vllm-ascend}"
    export PYTHON_BIN="${PYTHON_BIN:-python3}"
    export ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')}"
    export V41_CONFIG_SOURCE="$VLLM_ASCEND_ROOT/vllm_ascend/deepseek_v41_config.py"
    export V41_NATIVE_OPERATOR_ENV="$RECIPE_ROOT/tools/native_operator_env.sh"
else
    source "$RECIPE_ROOT/tools/v4_environment.sh"
fi
unset _recipe_profile
