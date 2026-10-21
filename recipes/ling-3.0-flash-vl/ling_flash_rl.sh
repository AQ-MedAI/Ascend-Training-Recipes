#!/usr/bin/env bash
# Ling-3.0-flash-VL text-only GRPO training with verl + Megatron + vLLM.
# Config: https://huggingface.co/inclusionAI/Ling-3.0-flash-VL/blob/main/config.json
# Requires a Bridge converter and vLLM/Ascend support for BailingMoeV3VLForConditionalGeneration.
# The K26 image alone does not provide the Ling model implementation or weight mappings.
# Vision training and K26's Kimi-to-DeepSeek conversion are not used by this recipe.
# Usage: MODEL_PATH=/models/Ling-3.0-flash-VL bash recipes/ling-3.0-flash-vl/ling_flash_rl.sh
# Set TRAIN_FILES / VAL_FILES to verl parquet files with prompt, data_source and reward_model.
# Built-in rewards use data_source; REWARD_FN_PATH optionally supplies compute_score.

set -euo pipefail
LING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Reuse only the established environment, runtime checks and PP layout helpers.
K26_SCRIPT_DIR="$(cd "$LING_SCRIPT_DIR/../kimi-k2.6" && pwd)"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "$K26_SCRIPT_DIR/common_env.sh"

# ------------------------- Paths -------------------------
EXP_NAME="${EXP_NAME:-ling_flash_rl}"
MODEL_PATH="${MODEL_PATH:-/path/to/Ling-3.0-flash-VL}"
DATA_ROOT="${DATA_ROOT:-/workspace-verl/data}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_ROOT}/ling_rl/train.parquet}"
VAL_FILES="${VAL_FILES:-${DATA_ROOT}/ling_rl/test.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/ckpt}"
LOG_DIR="${LOG_DIR:-/path/to/log}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------- Parallelism -------------------------
# Actor/ref use Megatron parallelism: TP=tensor, PP=pipeline, CP=context, EP=expert.
ACTOR_TP="${ACTOR_TP:-${TP:-4}}"
ACTOR_PP="${ACTOR_PP:-${PP:-2}}"
ACTOR_CP="${ACTOR_CP:-${CP:-2}}"
ACTOR_EP="${ACTOR_EP:-${EP:-8}}"
ACTOR_ETP="${ACTOR_ETP:-${ETP:-1}}"

# Default: two nodes with eight NPUs each; 42 decoder layers split evenly over PP=2.
# EP partitions experts jointly with ETP; it is not another factor in TP * PP * CP.
# An explicitly empty ROLLOUT_EP omits the expert-parallel override.
ROLLOUT_TP="${ROLLOUT_TP:-${GEN_TP:-8}}"
ROLLOUT_DP="${ROLLOUT_DP:-2}"
ROLLOUT_EP="${ROLLOUT_EP-16}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${NGPUS_PER_NODE:-8}}"
NNODES="${NNODES:-2}"

# ------------------------- Data And Lengths -------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

TOTAL_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$TOTAL_TOKEN_LEN}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$TOTAL_TOKEN_LEN}"

# ------------------------- Optimization -------------------------
LR="${LR:-5.0e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
OPTIMIZER="${OPTIMIZER:-adam}"
# Offload optimizer states to CPU in addition to phase-level offload.
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-True}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-False}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-False}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-True}"
BETAS="${BETAS:-[0.9,0.95]}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
MIN_LR="${MIN_LR:-0.0}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

# ------------------------- Rollout Sampling -------------------------
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEM_UTIL:-0.4}}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-2048}}"
ROLLOUT_SCHEDULING_POLICY="${ROLLOUT_SCHEDULING_POLICY:-fcfs}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"

# ------------------------- Trainer -------------------------
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
SEED="${SEED:-42}"
SAVE_FREQ="${SAVE_FREQ:-10}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra]}"
CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-[model,optimizer,extra]}"
export ASCEND_CHECKPOINT_WRITER="${ASCEND_CHECKPOINT_WRITER:-inprocess}"
case "$ASCEND_CHECKPOINT_WRITER" in
    inprocess|multiprocess) ;;
    *) echo 'ASCEND_CHECKPOINT_WRITER must be inprocess or multiprocess' >&2; exit 2 ;;
esac
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1145}"

# ------------------------- Tracking -------------------------
# open-instruct uses --with_tracking; verl enables the same behavior through trainer.logger.
WITH_TRACKING="${WITH_TRACKING:-false}"
PROJECT_NAME="${PROJECT_NAME:-ling_flash_verl}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR%/}/wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/path/to/tensorboard}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
if [[ -z "${LOGGER:-}" ]]; then
    if [[ "${WITH_TRACKING}" == "true" ]]; then
        LOGGER='["console","tensorboard","wandb"]'
    else
        LOGGER='["console"]'
    fi
fi

export WANDB_MODE
export WANDB_DIR
if [[ -n "${WANDB_ENTITY}" ]]; then
    export WANDB_ENTITY
fi
export TENSORBOARD_DIR

# Ling-specific layer types, attention gating and weight mappings must come from the Ling Bridge.
# Do not force Kimi/DeepSeek model settings onto the Bailing architecture.
# ------------------------- Hydra Overrides -------------------------
DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-True}"
    data.filter_overlong_prompts_workers="${FILTER_WORKERS:-4}"
    data.truncation="${DATA_TRUNCATION:-left}"
    data.shuffle=True
    data.train_max_samples="${TRAIN_MAX_SAMPLES:-5000}"
    data.val_max_samples="${VAL_MAX_SAMPLES:-1000}"
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.model.mtp.enable_train=False
)

ACTOR=(
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}"
    actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD}"
    actor_rollout_ref.actor.optim.optimizer="${OPTIMIZER}"
    actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}"
    actor_rollout_ref.actor.checkpoint.load_contents="${CHECKPOINT_LOAD_CONTENTS}"
    actor_rollout_ref.actor.checkpoint.async_save=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload="${OPTIMIZER_CPU_OFFLOAD}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="${OPTIMIZER_OFFLOAD_FRACTION}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d="${OVERLAP_CPU_OPTIMIZER_D2H_H2D}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPTIMIZER}"
    actor_rollout_ref.actor.optim.betas="${BETAS}"
    actor_rollout_ref.actor.optim.lr_decay_style="${LR_DECAY_STYLE}"
    actor_rollout_ref.actor.optim.min_lr="${MIN_LR}"
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${LR_WARMUP_STEPS_RATIO}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=3.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
    actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}"
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ACTOR_ETP}"
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_mask_type=causal
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size="${ACTOR_CP}"
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel="${SEQUENCE_PARALLEL}"
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
)

# Resolve endpoint layers against the checkpoint depth before initializing NPU workers.
PIPELINE_ARGS=(--model-path "$MODEL_PATH" --pp "$ACTOR_PP")
if [[ -n "${FIRST_PIPELINE_LAYERS+x}" ]]; then
    PIPELINE_ARGS+=(--first "$FIRST_PIPELINE_LAYERS")
fi
if [[ -n "${LAST_PIPELINE_LAYERS+x}" ]]; then
    PIPELINE_ARGS+=(--last "$LAST_PIPELINE_LAYERS")
fi
PIPELINE_OVERRIDES="$("$PYTHON_BIN" "$K26_SCRIPT_DIR/pipeline_layout.py" "${PIPELINE_ARGS[@]}")"
while IFS= read -r override; do
    if [[ -n "$override" ]]; then
        ACTOR+=("$override")
    fi
done <<< "$PIPELINE_OVERRIDES"

ROLLOUT=(
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}"
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.n="${N_SAMPLES}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.scheduling_policy="${ROLLOUT_SCHEDULING_POLICY}"
    actor_rollout_ref.rollout.dtype="${ROLLOUT_DTYPE}"
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${TOP_P}"
    actor_rollout_ref.rollout.top_k="${TOP_K}"
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-True}"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cudagraph_capture_sizes="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-[1,4,8]}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto
)

if [[ -n "${ROLLOUT_EP}" ]]; then
    ROLLOUT+=(actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}")
fi

REF=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
    actor_rollout_ref.ref.megatron.context_parallel_size="${ACTOR_CP}"
    actor_rollout_ref.ref.megatron.expert_model_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${ACTOR_ETP}"
    actor_rollout_ref.ref.megatron.param_offload=True
)

ALGORITHM=(
    algorithm.adv_estimator="${ADV_ESTIMATOR}"
    algorithm.gamma=1.0
    algorithm.lam=1.0
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXP_NAME}"
    trainer.n_gpus_per_node="${GPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.val_before_train=False
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.default_local_dir="${OUTPUT_DIR%/}/${PROJECT_NAME}/${EXP_NAME}"
)

REWARD=()
if [[ -n "$REWARD_FN_PATH" ]]; then
    REWARD+=(
        reward.custom_reward_function.path="$REWARD_FN_PATH"
        reward.custom_reward_function.name="${REWARD_FN_NAME:-compute_score}"
    )
fi

# Reuse the validated Ascend BF16 runtime; model-specific Ling adapters must already be installed.
source "$K26_SCRIPT_DIR/runtime_env.sh"
source "$K26_SCRIPT_DIR/ray_env.sh"
EXTRA=(
    model_engine=megatron
    +seed="${SEED}"
    distillation.enabled=False
    ${K26_RUNTIME_ENV_ARGS[@]+"${K26_RUNTIME_ENV_ARGS[@]}"}
)

# ------------------------- Launch -------------------------
mkdir -p "${LOG_DIR}"
START_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${START_TIME}_${EXP_NAME}_verl_grpo_${NNODES}N.log"

CMD=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    "${DATA[@]}"
    "${ALGORITHM[@]}"
    "${MODEL[@]}"
    "${ROLLOUT[@]}"
    "${ACTOR[@]}"
    "${REF[@]}"
    "${TRAINER[@]}"
    ${REWARD[@]+"${REWARD[@]}"}
    "${EXTRA[@]}"
    "$@"
)

printf 'Logging to %s\n' "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
