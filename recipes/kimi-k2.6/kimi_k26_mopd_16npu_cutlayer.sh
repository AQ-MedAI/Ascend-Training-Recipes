#!/usr/bin/env bash
# Two-layer MOPD smoke: eight student NPUs and two TP4 teachers on eight more NPUs.
# Reuse the validated RL launcher for actor, optimizer, rollout and checkpoint settings.
set -euo pipefail
K26_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL_PATH="${MODEL_PATH:-${CUT_MODEL_PATH:-${TMPDIR:-/tmp}/kimi_k26_bf16_2layers_model}}"
export FULL_MODEL_PATH="${FULL_MODEL_PATH:-/path/to/kimi-k2.6-bf16}"
PREPARE_ARGS=(--output-path "$MODEL_PATH" --num-layers 2)
if [[ -n "${FULL_MODEL_PATH:-}" ]]; then
    PREPARE_ARGS+=(--input-path "$FULL_MODEL_PATH")
fi
"$PYTHON_BIN" "$K26_SCRIPT_DIR/prepare_kimi_k26_cut_model.py" "${PREPARE_ARGS[@]}"

# Match kimi_k26_rl_8npu_cutlayer.sh; the teacher pool is allocated separately.
export NNODES=1 GPUS_PER_NODE=8
export ACTOR_TP="${ACTOR_TP:-${TP:-4}}" ACTOR_PP="${ACTOR_PP:-${PP:-1}}" ACTOR_CP="${ACTOR_CP:-${CP:-2}}"
export ACTOR_EP="${ACTOR_EP:-${EP:-8}}" ACTOR_ETP="${ACTOR_ETP:-${ETP:-1}}"
export ROLLOUT_TP=8 ROLLOUT_DP=1 ROLLOUT_EP=8
export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 N_SAMPLES=2
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 3))}"
export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}}"
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_MAX_TOKEN_LEN_PER_GPU}}"
export ROLLOUT_MAX_NUM_SEQS=8
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
export TOTAL_TRAINING_STEPS=2 TOTAL_EPOCHS=1 SAVE_FREQ="${SAVE_FREQ:-1}" TEST_FREQ=-1
export LR_WARMUP_STEPS_RATIO=0
export LOGGER='["console"]' WITH_TRACKING=false
export PROJECT_NAME="${PROJECT_NAME:-verl_distill_mopd_math_gsm8k}"
export EXP_NAME="${EXP_NAME:-${EXPERIMENT_NAME:-kimi_k26_mopd_2layers_smoke}}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"

# Student and teacher pools need disjoint devices; defaults require all 16 NPUs.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_USE_V1=1 TASK_QUEUE_ENABLE=0

# Each parquet's data_source must match its teacher key and reward branch.
DATA_ROOT="${DATA_ROOT:-/workspace-verl/data}"
MATH_TRAIN_FILE="${MATH_TRAIN_FILE:-${DATA_ROOT}/math/train.parquet}"
MATH_VAL_FILE="${MATH_VAL_FILE:-${DATA_ROOT}/math/test.parquet}"
GSM8K_TRAIN_FILE="${GSM8K_TRAIN_FILE:-${DATA_ROOT}/gsm8k/train.parquet}"
GSM8K_VAL_FILE="${GSM8K_VAL_FILE:-${DATA_ROOT}/gsm8k/test.parquet}"
export TRAIN_FILES="$("$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$MATH_TRAIN_FILE" "$GSM8K_TRAIN_FILE")"
export VAL_FILES="$("$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$MATH_VAL_FILE" "$GSM8K_VAL_FILE")"
export REWARD_FN_PATH="${REWARD_FN_PATH:-${K26_SCRIPT_DIR}/multi_reward.py}"

# Smoke teachers reuse the cropped checkpoint; overrides must share its vocabulary.
MATH_TEACHER_MODEL="${MATH_TEACHER_MODEL:-${MODEL_PATH}}"
GSM8K_TEACHER_MODEL="${GSM8K_TEACHER_MODEL:-${MODEL_PATH}}"
teacher_tp="${TEACHER_TP:-4}"
math_replicas="${TEACHER_NUM_REPLICAS_MATH:-1}"
gsm8k_replicas="${TEACHER_NUM_REPLICAS_GSM8K:-1}"
teacher_world_size=$(( (math_replicas + gsm8k_replicas) * teacher_tp ))

MOPD=(
    distillation.enabled=True
    distillation.n_gpus_per_node="$teacher_world_size"
    distillation.nnodes=1
    distillation.teacher_key=data_source
    +distillation.teacher_models.math.key=DigitalLearningGmbH/MATH-lighteval
    +distillation.teacher_models.math.model_path="$MATH_TEACHER_MODEL"
    +distillation.teacher_models.math.num_replicas="$math_replicas"
    +distillation.teacher_models.gsm8k.key=openai/gsm8k
    +distillation.teacher_models.gsm8k.model_path="$GSM8K_TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas="$gsm8k_replicas"
)
for teacher in math gsm8k; do
    MOPD+=(
        "+distillation.teacher_models.${teacher}.inference.name=vllm"
        "+distillation.teacher_models.${teacher}.inference.enforce_eager=True"
        "+distillation.teacher_models.${teacher}.inference.tensor_model_parallel_size=${teacher_tp}"
        "+distillation.teacher_models.${teacher}.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL:-0.5}"
        "+distillation.teacher_models.${teacher}.inference.enable_chunked_prefill=False"
        "+distillation.teacher_models.${teacher}.inference.max_num_seqs=1"
        "+distillation.teacher_models.${teacher}.inference.max_model_len=${MAX_MODEL_LEN}"
        "+distillation.teacher_models.${teacher}.inference.max_num_batched_tokens=${MAX_MODEL_LEN}"
    )
done
MOPD+=(
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE:-forward_kl_full_vocab}"
    distillation.distillation_loss.full_vocab_chunk_tokens="${FULL_VOCAB_CHUNK_TOKENS:-2048}"
    distillation.distillation_loss.topk="${DISTILLATION_TOPK:-64}"
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT:-True}"
    distillation.distillation_loss.loss_max_clamp=2.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.distillation_loss_coef="${DISTILLATION_LOSS_COEF:-0.5}"
)


# Propagate additional teacher/native runtime settings alongside the RL environment.
for name in HF_HUB_OFFLINE TRANSFORMERS_OFFLINE VLLM_USE_V1 TASK_QUEUE_ENABLE \
    ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME \
    ASCEND_TOOLKIT_LATEST_HOME ATB_HOME_PATH TOOLCHAIN_HOME CMAKE_PREFIX_PATH ASCEND_RUNTIME_OPTIONS; do
    value="${!name:-}"
    if [[ -n "$value" ]]; then
        MOPD+=("++ray_kwargs.ray_init.runtime_env.env_vars.${name}='${value}'")
    fi
done

exec bash "$K26_SCRIPT_DIR/kimi_k26_rl.sh" \
    data.truncation="${DATA_TRUNCATION:-left}" \
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-True}" \
    data.filter_overlong_prompts_workers="${FILTER_WORKERS:-4}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES:-8}" \
    data.val_max_samples="${VAL_MAX_SAMPLES:-8}" \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    "${MOPD[@]}" "$@"
