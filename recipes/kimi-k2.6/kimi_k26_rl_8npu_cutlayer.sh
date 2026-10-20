#!/usr/bin/env bash
# Two-layer GRPO smoke test: one node, eight NPUs, two optimizer steps.
# Create the two-layer checkpoint on first use; validate and reuse it afterwards.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL_PATH="${MODEL_PATH:-${CUT_MODEL_PATH:-${TMPDIR:-/tmp}/kimi_k26_bf16_2layers_model}}"
export FULL_MODEL_PATH="${FULL_MODEL_PATH:-/path/to/kimi-k2.6-bf16}"
if (( $# != 0 )); then
    echo 'This smoke entry does not accept Hydra overrides; use environment variables for paths.' >&2
    exit 2
fi
PREPARE_ARGS=(--output-path "$MODEL_PATH" --num-layers 2)
if [[ -n "${FULL_MODEL_PATH:-}" ]]; then
    PREPARE_ARGS+=(--input-path "$FULL_MODEL_PATH")
fi
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_kimi_k26_cut_model.py" "${PREPARE_ARGS[@]}"
export NNODES=1 GPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ACTOR_TP="${ACTOR_TP:-${TP:-4}}" ACTOR_PP="${ACTOR_PP:-${PP:-1}}" ACTOR_CP="${ACTOR_CP:-${CP:-2}}"
export ACTOR_EP="${ACTOR_EP:-${EP:-8}}" ACTOR_ETP="${ACTOR_ETP:-${ETP:-1}}"
export ROLLOUT_TP=8 ROLLOUT_DP=1 ROLLOUT_EP=8
export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 N_SAMPLES=2
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 3))}"
export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_MAX_TOKEN_LEN_PER_GPU}}"
export ROLLOUT_MAX_NUM_SEQS=8
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
export ROLLOUT_ENFORCE_EAGER=True
export ROLLOUT_CUDAGRAPH_CAPTURE_SIZES='[1,4,16,32,64]'
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
export TOTAL_TRAINING_STEPS=2 TOTAL_EPOCHS=1 SAVE_FREQ="${SAVE_FREQ:-1}" TEST_FREQ=-1
export LR_WARMUP_STEPS_RATIO=0
export LOGGER='["console"]' WITH_TRACKING=false
export EXP_NAME="${EXP_NAME:-kimi_k26_rl_2layers_smoke}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
exec bash "$SCRIPT_DIR/kimi_k26_rl.sh" \
    distillation.enabled=False \
    data.truncation="${DATA_TRUNCATION:-left}" \
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-True}" \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.enforce_eager=True
