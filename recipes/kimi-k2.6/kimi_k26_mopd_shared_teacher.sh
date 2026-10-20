#!/usr/bin/env bash
# Two-teacher reverse-KL distillation without task rewards or policy gradient.
# Common actor, optimizer, runtime, paths and Ray configuration live in kimi_k26_mopd.sh.
set -euo pipefail
K26_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_PROFILE=multi
export DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-reverse_kl_full_vocab}"
export USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-False}"
export USE_TASK_REWARDS="${USE_TASK_REWARDS:-False}"
export DISTILLATION_LOSS_MAX_CLAMP="${DISTILLATION_LOSS_MAX_CLAMP:-10.0}"
export DISTILLATION_LOSS_COEF="${DISTILLATION_LOSS_COEF:-0.5}"
export TEACHER_DP="${TEACHER_DP:-1}" TEACHER_PCP="${TEACHER_PCP:-16}"
export GSM8K_TEACHER_MODEL="${GSM8K_TEACHER_MODEL:-${TEACHER_MODEL_1:-${MODEL_PATH:-/path/to/Kimi-K2.6_bf16}}}"
export MATH_TEACHER_MODEL="${MATH_TEACHER_MODEL:-${TEACHER_MODEL_2:-${MODEL_PATH:-/path/to/Kimi-K2.6_bf16}}}"
# Preserve the original custom task's data_source and reward entry.
export MATH_TEACHER_KEY="${MATH_TEACHER_KEY:-k26_rl}"
export MATH_TRAIN_FILE="${MATH_TRAIN_FILE:-${DATA_ROOT:-/workspace-verl/data}/k26_rl/train.parquet}"
export MATH_VAL_FILE="${MATH_VAL_FILE:-${DATA_ROOT:-/workspace-verl/data}/k26_rl/test.parquet}"
export REWARD_FN_PATH="${REWARD_FN_PATH:-${DATA_ROOT:-/workspace-verl/data}/k26_rl_reward.py}"
export TEACHER_NUM_REPLICAS_GSM8K="${TEACHER_NUM_REPLICAS_GSM8K:-${TEACHER_NUM_REPLICAS_1:-2}}"
export TEACHER_NUM_REPLICAS_MATH="${TEACHER_NUM_REPLICAS_MATH:-${TEACHER_NUM_REPLICAS_2:-2}}"
# Both teachers reuse the same pool; only one may be awake.
export DISTILLATION_SHARE_GPU_GROUP=True
export MAX_AWAKE_TEACHERS=1
export SAVE_FREQ="${SAVE_FREQ:-100}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-30}"
export EXP_NAME="${EXP_NAME:-kimi_k26_mopd_shared_teacher}"
export TEST_FREQ="${TEST_FREQ:-10}"
exec bash "$K26_SCRIPT_DIR/kimi_k26_mopd.sh" "$@"
