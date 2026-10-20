#!/usr/bin/env bash
# Full-model, two-teacher full-vocabulary MOPD; defaults to reverse KL.
# Use matching runtimes and identical shared paths on every Ray node.
# Prepopulate remote model code in the shared HF_HOME cache for offline workers.
# Full-vocabulary teachers require matching tokenizers and vocabulary dimensions.
set -xeuo pipefail
K26_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

#export HF_HOME=${HF_HOME:-/path/to/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export GROUP_HCCL_BUFFER_SIZE="${GROUP_HCCL_BUFFER_SIZE:-200}"
export P2P_HCCL_BUFFSIZE="${P2P_HCCL_BUFFSIZE:-0}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export VERL_ROOT="${VERL_ROOT:-$(cd "$K26_SCRIPT_DIR/../.." && pwd)/third_party/verl}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

# ---- user-adjustable ----
export MODEL_PATH="${MODEL_PATH:-/path/to/Kimi-K2.6_bf16}"
STUDENT_MODEL="$MODEL_PATH"
GSM8K_TEACHER_MODEL="${GSM8K_TEACHER_MODEL:-${MODEL_PATH}}"
MATH_TEACHER_MODEL="${MATH_TEACHER_MODEL:-${MODEL_PATH}}"

NNODES=${NNODES:-40}
GPUS_PER_NODE=${GPUS_PER_NODE:-${NGPUS_PER_NODE:-16}}
ACTOR_TP=${ACTOR_TP:-${TP:-8}}
ACTOR_PP=${ACTOR_PP:-${PP:-10}}
ACTOR_CP=${ACTOR_CP:-${CP:-8}}
ACTOR_EP=${ACTOR_EP:-${EP:-64}}
ACTOR_ETP=${ACTOR_ETP:-${ETP:-1}}
ROLLOUT_TP=${ROLLOUT_TP:-${GEN_TP:-8}}
ROLLOUT_DP=${ROLLOUT_DP:-16}
ROLLOUT_EP=${ROLLOUT_EP:-128}
OFFLOAD=${OFFLOAD:-True}

TEACHER_NNODES=${TEACHER_NNODES:-16}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-16}
TEACHER_TP=${TEACHER_TP:-8}
TEACHER_DP=${TEACHER_DP:-2}
TEACHER_PCP=${TEACHER_PCP:-8}
TEACHER_EP=${TEACHER_EP-128}
TEACHER_NUM_REPLICAS_GSM8K=${TEACHER_NUM_REPLICAS_GSM8K:-1}
TEACHER_NUM_REPLICAS_MATH=${TEACHER_NUM_REPLICAS_MATH:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.7}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-reverse_kl_full_vocab}
full_vocab_chunk_tokens=${FULL_VOCAB_CHUNK_TOKENS:-4096}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# Full-model defaults; the cropped entry supplies its own batch and token limits.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-65536}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
N_SAMPLES=${N_SAMPLES:-1}
TOTAL_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU:-$TOTAL_TOKEN_LEN}}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$TOTAL_TOKEN_LEN}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((TOTAL_TOKEN_LEN + 1))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-2048}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEM_UTIL:-0.4}}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_SCHEDULING_POLICY=${ROLLOUT_SCHEDULING_POLICY:-fcfs}
ROLLOUT_DTYPE=${ROLLOUT_DTYPE:-bfloat16}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-1000000}
LR=${LR:-${ACTOR_LR:-5.0e-7}}
WEIGHT_DECAY=${WEIGHT_DECAY:-1.0e-2}
CLIP_GRAD=${CLIP_GRAD:-1.0}
OPTIMIZER=${OPTIMIZER:-adam}
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-True}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1}
OVERLAP_CPU_OPTIMIZER_D2H_H2D=${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-False}
USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-False}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-True}
BETAS=${BETAS:-[0.9,0.95]}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-constant}
MIN_LR=${MIN_LR:-0.0}
LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.03}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
CHECKPOINT_SAVE_CONTENTS=${CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra]}
CHECKPOINT_LOAD_CONTENTS=${CHECKPOINT_LOAD_CONTENTS:-[model,optimizer,extra]}
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-$MAX_MODEL_LEN}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-50}
project_name=${PROJECT_NAME:-verl_distill_mopd_full}
experiment_name=${EXP_NAME:-${EXPERIMENT_NAME:-k26_full_mopd_2teacher_fullvocab}}
# Each parquet's data_source must match its teacher key and reward branch.
DATA_ROOT="${DATA_ROOT:-/workspace-verl/data}"
MATH_TRAIN_FILE="${MATH_TRAIN_FILE:-${DATA_ROOT}/math/train.parquet}"
MATH_VAL_FILE="${MATH_VAL_FILE:-${DATA_ROOT}/math/test.parquet}"
GSM8K_TRAIN_FILE="${GSM8K_TRAIN_FILE:-${DATA_ROOT}/gsm8k/train.parquet}"
GSM8K_VAL_FILE="${GSM8K_VAL_FILE:-${DATA_ROOT}/gsm8k/test.parquet}"
export TRAIN_FILES="${TRAIN_FILES:-$("$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$MATH_TRAIN_FILE" "$GSM8K_TRAIN_FILE")}"
export VAL_FILES="${VAL_FILES:-$("$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$MATH_VAL_FILE" "$GSM8K_VAL_FILE")}"
export REWARD_FN_PATH="${REWARD_FN_PATH:-${K26_SCRIPT_DIR}/multi_reward.py}"


# ---- end user-adjustable ----

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator="${ADV_ESTIMATOR:-grpo}"
    algorithm.gamma=1.0
    algorithm.lam=1.0
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)


DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-True}"
    data.filter_overlong_prompts_workers=${FILTER_WORKERS:-4}
    data.truncation="${DATA_TRUNCATION:-left}"
    data.shuffle=True
    data.train_max_samples=${TRAIN_MAX_SAMPLES:-5000}
    data.val_max_samples=${VAL_MAX_SAMPLES:-1000}
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.trust_remote_code=True
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
    actor_rollout_ref.actor.megatron.param_offload="${OFFLOAD}"
    actor_rollout_ref.actor.megatron.optimizer_offload="${OFFLOAD}"
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
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-False}"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cudagraph_capture_sizes="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-[1,4,8]}"
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

TRAINER=(
    trainer.device=npu
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-null}"
    trainer.critic_warmup=0
    trainer.default_local_dir="${OUTPUT_DIR:-/path/to/ckpt}/${project_name}/${experiment_name}"
    trainer.max_actor_ckpt_to_keep=5
)

REWARD=(
    reward.custom_reward_function.path="$REWARD_FN_PATH"
    reward.custom_reward_function.name=compute_score
)

export ASCEND_CHECKPOINT_WRITER="${ASCEND_CHECKPOINT_WRITER:-inprocess}"
case "$ASCEND_CHECKPOINT_WRITER" in
    inprocess|multiprocess) ;;
    *) echo 'ASCEND_CHECKPOINT_WRITER must be inprocess or multiprocess' >&2; exit 2 ;;
esac
source "$K26_SCRIPT_DIR/runtime_env.sh"

CANN_ENV_PROPAGATE=()
CANN_ENV_NAMES=(LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH \
    ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_TOOLKIT_LATEST_HOME \
    ATB_HOME_PATH TOOLCHAIN_HOME CMAKE_PREFIX_PATH ASCEND_RUNTIME_OPTIONS \
    HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE HCCL_CONNECT_TIMEOUT \
    HCCL_OP_EXPANSION_MODE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE VLLM_USE_V1 TASK_QUEUE_ENABLE
    ASCEND_LAUNCH_BLOCKING PYTHONFAULTHANDLER)
for _v in "${CANN_ENV_NAMES[@]}"; do
    _val="${!_v:-}"
    [ -z "$_val" ] && continue
    CANN_ENV_PROPAGATE+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_v}='${_val}'")
done
unset _v _val

# Multi-teacher: one teacher per dataset, routed by sample's `data_source` value.
EXTRA=(
    model_engine=megatron
    +seed=42
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="$GSM8K_TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas=${TEACHER_NUM_REPLICAS_GSM8K}
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${TEACHER_TP}
    +distillation.teacher_models.gsm8k.inference.data_parallel_size=${TEACHER_DP}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    +distillation.teacher_models.gsm8k.inference.enforce_eager="${TEACHER_ENFORCE_EAGER:-False}"
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.cudagraph_capture_sizes=[1]
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.prefill_context_parallel_size=${TEACHER_PCP}
    +distillation.teacher_models.gsm8k.inference.enable_chunked_prefill=False
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=1
    +distillation.teacher_models.gsm8k.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=${teacher_max_model_len}
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.kv_cache_dtype=auto
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.mamba_cache_dtype=float32
    +distillation.teacher_models.math.key="DigitalLearningGmbH/MATH-lighteval"
    +distillation.teacher_models.math.model_path="$MATH_TEACHER_MODEL"
    +distillation.teacher_models.math.num_replicas=${TEACHER_NUM_REPLICAS_MATH}
    +distillation.teacher_models.math.inference.name=vllm
    +distillation.teacher_models.math.inference.tensor_model_parallel_size=${TEACHER_TP}
    +distillation.teacher_models.math.inference.data_parallel_size=${TEACHER_DP}
    +distillation.teacher_models.math.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    +distillation.teacher_models.math.inference.enforce_eager="${TEACHER_ENFORCE_EAGER:-False}"
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.cudagraph_capture_sizes=[1]
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.prefill_context_parallel_size=${TEACHER_PCP}
    +distillation.teacher_models.math.inference.enable_chunked_prefill=False
    +distillation.teacher_models.math.inference.max_num_seqs=1
    +distillation.teacher_models.math.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.math.inference.max_num_batched_tokens=${teacher_max_model_len}
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.kv_cache_dtype=auto
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.mamba_cache_dtype=float32
    # --- loss: task-reward RL + fullvocab distillation regularizer ---
    # Task-reward policy loss plus the configured teacher distillation loss.
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.full_vocab_chunk_tokens=${full_vocab_chunk_tokens}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=2.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.distillation_loss_coef="${DISTILLATION_LOSS_COEF:-0.5}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NPU_ALLOC_CONF='${PYTORCH_NPU_ALLOC_CONF}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE='${HCCL_BUFFSIZE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE='${GROUP_HCCL_BUFFER_SIZE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_TEACHER_DEDICATED_RACK='${VERL_TEACHER_DEDICATED_RACK:-1}'"
    "${CANN_ENV_PROPAGATE[@]}"
)

if [[ -n "$TEACHER_EP" ]]; then
    for teacher in math gsm8k; do
        EXTRA+=("+distillation.teacher_models.${teacher}.inference.expert_parallel_size=${TEACHER_EP}")
    done
fi

EXTRA+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE='${P2P_HCCL_BUFFSIZE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS='${OMP_NUM_THREADS}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MKL_NUM_THREADS='${MKL_NUM_THREADS}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROOT='${VERL_ROOT}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_CHECKPOINT_WRITER='${ASCEND_CHECKPOINT_WRITER}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.MINDSPEED_ROOT='${MINDSPEED_ROOT}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_ROOT='${MEGATRON_ROOT}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_BRIDGE_ROOT='${MEGATRON_BRIDGE_ROOT}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_VERBOSITY='${TRANSFORMERS_VERBOSITY:-error}'"
)

########################### launch ###########################
LOG_DIR="${LOG_DIR:-/path/to/k26_full_mopd_fullvocab}"
mkdir -p "${LOG_DIR}"
start_time=$(date +%Y%m%d_%H%M%S)
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/${start_time}_${experiment_name}.log"
