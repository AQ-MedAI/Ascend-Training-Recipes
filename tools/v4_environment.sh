#!/usr/bin/env bash
# Source from V4 launchers before starting Python or Ray.
export RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-/opt/ascend-training}"
export VERL_ROOT="${VERL_ROOT:-$DEPENDENCY_ROOT/third_party/verl}"
export MINDSPEED_ROOT="${MINDSPEED_ROOT:-$DEPENDENCY_ROOT/third_party/MindSpeed}"
export MINDSPEED_LLM_ROOT="${MINDSPEED_LLM_ROOT:-$DEPENDENCY_ROOT/third_party/MindSpeed-LLM}"
export MBRIDGE_ROOT="${MBRIDGE_ROOT:-$DEPENDENCY_ROOT/third_party/mbridge}"
export MEGATRON_ROOT="${MEGATRON_ROOT:-$DEPENDENCY_ROOT/third_party/Megatron-LM}"
# Shared image layout; override both paths when using a separate V4 runtime.
export VLLM_ROOT="${VLLM_ROOT:-$DEPENDENCY_ROOT/third_party/vllm}"
export VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-$DEPENDENCY_ROOT/third_party/vllm-ascend}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$RECIPE_ROOT:$VERL_ROOT:$MINDSPEED_ROOT:$MINDSPEED_LLM_ROOT:$MBRIDGE_ROOT:$MEGATRON_ROOT:$VLLM_ASCEND_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"
