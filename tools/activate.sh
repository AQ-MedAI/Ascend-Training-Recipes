#!/usr/bin/env bash
# Source in the shell that starts Ray and training; repeat on every node.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source tools/activate.sh [v41|v4|k26]' >&2
    exit 2
fi
_recipe_profile="${1:-${RECIPE_PROFILE:-v41}}"
if [[ "$_recipe_profile" != v41 && "$_recipe_profile" != v4 && "$_recipe_profile" != k26 ]]; then
    echo 'Supported profiles: v41, v4, k26' >&2
    return 2
fi
export RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$_recipe_profile" == k26 ]]; then
    export PYTHON_BIN="${PYTHON_BIN:-python3}"
    export VERL_ROOT="${VERL_ROOT:-$RECIPE_ROOT/third_party/verl}"
    source "$RECIPE_ROOT/recipes/kimi-k2.6/runtime_env.sh" || return $?
    export ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-$PYTHON_BIN}"
    unset _recipe_profile
    return 0
fi
source "$RECIPE_ROOT/tools/v4_environment.sh"
export ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')}"
export V41_CONFIG_SOURCE="${V41_CONFIG_SOURCE:-$VLLM_ASCEND_ROOT/vllm_ascend/deepseek_v41_config.py}"
export V41_NATIVE_OPERATOR_ENV="${V41_NATIVE_OPERATOR_ENV:-$RECIPE_ROOT/tools/native_operator_env.sh}"
unset _recipe_profile
