#!/usr/bin/env bash
# Kimi-K2.6 GRPO training with verl + Megatron + vLLM.
# Usage: override any variable before the command, e.g. ACTOR_TP=4 NNODES=8 bash recipes/kimi-k2.6/kimi_k26_verl.sh

set -xeuo pipefail

RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export VERL_ROOT="${VERL_ROOT:-$RECIPE_ROOT/third_party/verl}"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

# ------------------------- Environment -------------------------
# NPU/HCCL defaults. Override from the shell when cluster settings differ.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export HYBRID_IGNORE_LOAD_CHECK="${HYBRID_IGNORE_LOAD_CHECK:-1}"
export USE_MAX_V2="${USE_MAX_V2:-0}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-62000-62050}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-62100-62200}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"

# ------------------------- Paths -------------------------
EXP_NAME="${EXP_NAME:-kimi_k26}"
MODEL_PATH="${MODEL_PATH:-/path/to/Kimi-K2.6_bf16}"
TRAIN_FILES="${TRAIN_FILES:-/path/to/k26_rl_train.parquet}"
VAL_FILES="${VAL_FILES:-/path/to/k26_rl_valid.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-/path/to/k26_rl_reward.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/ckpt}"
LOG_DIR="${LOG_DIR:-/path/to/log}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------- Parallelism -------------------------
# Actor/ref use Megatron parallelism: TP=tensor, PP=pipeline, CP=context, EP=expert.
ACTOR_TP="${ACTOR_TP:-8}"
ACTOR_PP="${ACTOR_PP:-10}"
ACTOR_CP="${ACTOR_CP:-8}"
ACTOR_EP="${ACTOR_EP:-64}"
ACTOR_ETP="${ACTOR_ETP:-1}"

# Rollout uses vLLM parallelism. Leave ROLLOUT_EP empty unless the backend requires expert parallelism.
ROLLOUT_TP="${ROLLOUT_TP:-8}"
ROLLOUT_DP="${ROLLOUT_DP:-16}"
ROLLOUT_EP="${ROLLOUT_EP:-}"

GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
NNODES="${NNODES:-40}"

# ------------------------- Data And Lengths -------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-65536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

TOTAL_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$((TOTAL_TOKEN_LEN / ACTOR_CP))}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((TOTAL_TOKEN_LEN / ACTOR_CP))}"

# ------------------------- Optimization -------------------------
LR="${LR:-5.0e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
OPTIMIZER="${OPTIMIZER:-adam}"
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
TOP_K="${TOP_K:-1000000}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
ROLLOUT_SCHEDULING_POLICY="${ROLLOUT_SCHEDULING_POLICY:-fcfs}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"

# ------------------------- Trainer -------------------------
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
SEED="${SEED:-42}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1145}"

# ------------------------- Tracking -------------------------
# open-instruct uses --with_tracking; verl enables the same behavior through trainer.logger.
WITH_TRACKING="${WITH_TRACKING:-true}"
PROJECT_NAME="${PROJECT_NAME:-kimi_k26_verl}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR%/}/wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/path/to/tensorboard}"
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

# ------------------------- Hydra Overrides -------------------------
DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}"
    actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD}"
    actor_rollout_ref.actor.optim.optimizer="${OPTIMIZER}"
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
    +actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.multi_latent_attention=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_mask_type=causal
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size="${ACTOR_CP}"
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_score_function=sigmoid
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_topk_scaling_factor=2.5
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
)

# Uneven pipeline splits apply only to the full-model pipeline profile.
if (( ACTOR_PP > 1 )); then
    ACTOR+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage="${FIRST_PIPELINE_LAYERS:-3}"
        +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage="${LAST_PIPELINE_LAYERS:-2}"
    )
fi

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
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cudagraph_capture_sizes=[1,4,16,32,64]
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_dtype=float32
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

REWARD=(
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
)

EXTRA=(
    model_engine=megatron
    +seed="${SEED}"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE="'200'"
    +ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE="'200'"
    +ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE="'0'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'${PYTHONPATH}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROOT="'${VERL_ROOT}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="'${LD_LIBRARY_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_CUSTOM_OPP_PATH="'${ASCEND_CUSTOM_OPP_PATH:-}'"
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
    "${REWARD[@]}"
    "${EXTRA[@]}"
    "$@"
)

printf 'Logging to %s\n' "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
