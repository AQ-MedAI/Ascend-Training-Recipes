#!/usr/bin/env bash
# Launch the standalone DSV4 vLLM prefill diagnostic on one TP4 replica.

set -euo pipefail

RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$RECIPE_ROOT/tools/v4_environment.sh"
MODEL_PATH="${MODEL_PATH:-${TMPDIR:-/tmp}/dsv4_flash_bf16_4layers_model}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128}"
PROMPT_TOKEN_IDS_FILE="${PROMPT_TOKEN_IDS_FILE:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${VLLM_ASCEND_ROOT}:${VLLM_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export VLLM_ASCEND_TASK_QUEUE_ENABLE="${VLLM_ASCEND_TASK_QUEUE_ENABLE:-0}"
# The RL control script uses devices 0-7.  Default to the other half of a
# 16-NPU node so a timed-out Ray/vLLM job cannot collide with this TP4 probe.
export ASCEND_RT_VISIBLE_DEVICES="${DIAG_ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"

PROMPT_ARGS=(--prompt-length "${PROMPT_LENGTH}")
if [[ -n "${PROMPT_TOKEN_IDS_FILE}" ]]; then
    PROMPT_ARGS=(--prompt-token-ids-file "${PROMPT_TOKEN_IDS_FILE}")
fi

exec "${PYTHON_BIN:-python3}" \
    "${SCRIPT_DIR}/diagnose_dsv4_flash_vllm_prefill.py" \
    --model "${MODEL_PATH}" \
    "${PROMPT_ARGS[@]}" \
    "$@"
