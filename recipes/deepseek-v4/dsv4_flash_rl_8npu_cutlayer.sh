#!/usr/bin/env bash
# Pure-RL control run for the DeepSeek-V4-Flash MOPD smoke test.
#
# This intentionally reuses the MOPD script so the student model conversion,
# MindSpeed actor topology, rollout topology, sequence lengths, and vLLM
# settings stay identical.  Distillation is disabled, so no teacher worker is
# created and only the 8-NPU student/main resource pool is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_CUT_MODEL_OVERLAY="${USE_CUT_MODEL_OVERLAY:-True}"
export DISTILLATION_ENABLED=False
export ASCEND_RT_VISIBLE_DEVICES="${RL_ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export PROJECT_NAME="${PROJECT_NAME:-DeepSeek-V4-Flash-RL-Control}"
export EXP_NAME="${EXP_NAME:-dsv4_flash_${CUT_NUM_LAYERS:-4}layers_rl_8npu_control}"

exec bash "${SCRIPT_DIR}/dsv4_flash_mopd_16npu_cutlayer.sh" "$@"
