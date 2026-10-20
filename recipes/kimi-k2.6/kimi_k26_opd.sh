#!/usr/bin/env bash
# Single-teacher forward-KL top-k OPD. No task reward or policy-gradient loss.
# Reuse the common MOPD actor/runtime launcher with a single teacher profile.
set -euo pipefail
K26_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_PROFILE=single
export DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-forward_kl_topk}"
export USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-False}"
export USE_TASK_REWARDS="${USE_TASK_REWARDS:-False}"
export DISTILLATION_LOSS_MAX_CLAMP="${DISTILLATION_LOSS_MAX_CLAMP:-10.0}"
export DISTILLATION_LOSS_COEF="${DISTILLATION_LOSS_COEF:-1.0}"
export TEACHER_DP="${TEACHER_DP:-16}" TEACHER_PCP="${TEACHER_PCP:-1}"
export TEACHER_ENABLE_CHUNKED_PREFILL="${TEACHER_ENABLE_CHUNKED_PREFILL:-True}"
export TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-32}"
export TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-1024}"
# replicas=0 keeps the existing single-teacher automatic pool sizing policy.
export TEACHER_NUM_REPLICAS="${TEACHER_NUM_REPLICAS:-0}"
export TRAIN_FILES="${TRAIN_FILES:-${TRAIN_PATH:-${DATA_ROOT:-/workspace-verl/data}/k26_rl/train.parquet}}"
export VAL_FILES="${VAL_FILES:-${TEST_PATH:-${DATA_ROOT:-/workspace-verl/data}/k26_rl/test.parquet}}"
export REWARD_FN_PATH="${REWARD_FN_PATH:-${DATA_ROOT:-/workspace-verl/data}/k26_rl_reward.py}"
export SAVE_FREQ="${SAVE_FREQ:-10}" TEST_FREQ="${TEST_FREQ:-10}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1145}"
export EXP_NAME="${EXP_NAME:-kimi_k26_opd}"
exec bash "$K26_SCRIPT_DIR/kimi_k26_mopd.sh" "$@"
