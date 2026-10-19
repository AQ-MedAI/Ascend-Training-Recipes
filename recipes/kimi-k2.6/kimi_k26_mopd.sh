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
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60200
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61200
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export HCCL_BUFFSIZE=512
export GROUP_HCCL_BUFFER_SIZE=256
export P2P_HCCL_BUFFSIZE=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

VERL_ROOT=${VERL_ROOT:-/path/to/verl}
cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

# ---- user-adjustable ----
HF_MODEL_PATH=/path/to/Kimi-K2.6_bf16
STUDENT_MODEL=${HF_MODEL_PATH}
GSM8K_TEACHER_MODEL=${HF_MODEL_PATH}
MATH_TEACHER_MODEL=${HF_MODEL_PATH}

NNODES=${NNODES:-40}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
TP=${TP:-8}
PP=${PP:-10}
CP=${CP:-8}
EP=${EP:-64}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}            # rollout vllm tensor_parallel_size
ROLLOUT_DP=${ROLLOUT_DP:-16}
ROLLOUT_EP=${ROLLOUT_EP:-128}
OFFLOAD=${OFFLOAD:-True}       # megatron param/optimizer offload

TEACHER_NNODES=${TEACHER_NNODES:-16}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-16}
TEACHER_TP=${TEACHER_TP:-8}
TEACHER_DP=${TEACHER_DP:-2}
TEACHER_PCP=${TEACHER_PCP:-8}
TEACHER_EP=${TEACHER_EP:-128}
TEACHER_NUM_REPLICAS_GSM8K=${TEACHER_NUM_REPLICAS_GSM8K:-1}
TEACHER_NUM_REPLICAS_MATH=${TEACHER_NUM_REPLICAS_MATH:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.7}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-reverse_kl_full_vocab}
full_vocab_chunk_tokens=${FULL_VOCAB_CHUNK_TOKENS:-4096}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# ---- batch + token (OPD 64k/8k) ----
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 64))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
N_SAMPLES=${N_SAMPLES:-1}
actor_ppo_max_token_len=$(( max_prompt_length + max_response_length ))
infer_ppo_max_token_len=$(( max_prompt_length + max_response_length ))
actor_ppo_max_token_len=$(( actor_ppo_max_token_len / CP ))
infer_ppo_max_token_len=$(( infer_ppo_max_token_len / CP ))
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$actor_ppo_max_token_len}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-2048}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-1000000}
actor_lr=${ACTOR_LR:-5.0e-7}

rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-$max_num_tokens}

total_epochs=${TOTAL_EPOCHS:-1}   # smoke
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-50}
project_name=${PROJECT_NAME:-verl_distill_mopd_full}
experiment_name=${EXPERIMENT_NAME:-k26_full_mopd_2teacher_fullvocab}
reward_fn_path="${REWARD_FN_PATH:-${K26_SCRIPT_DIR}/math_dapo_reward.py}"

# Dataset A: gsm8k  (data_source = "openai/gsm8k") -> teacher gsm8k
gsm8k_train=/path/to/train.parquet
gsm8k_test=/path/to/test.parquet
math_train=/path/to/orz_math_train.parquet
math_test=/path/to/orz_math_train.parquet
train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"
# ---- end user-adjustable ----

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=True
    algorithm.kl_penalty=kl
    +algorithm.kl_coef=0.001
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=${FILTER_WORKERS:-8}
    data.truncation='error'
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
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=1.0e-2
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=adam
    actor_rollout_ref.actor.optim.betas=[0.9,0.95]
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.min_lr=0.0
    actor_rollout_ref.actor.optim.lr_warmup_steps=0.03
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # ---- PPO clip (OPD) ----
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=3.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.multi_latent_attention=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_mask_type=causal
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${CP}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_score_function=sigmoid
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_topk_scaling_factor=2.5
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=3
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=2
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${N_SAMPLES}
    actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.scheduling_policy=fcfs
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TOP_P}
    actor_rollout_ref.rollout.top_k=${TOP_K}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cudagraph_capture_sizes=[1,4,8]
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_dtype=float32
)

TRAINER=(
    trainer.device=npu
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir=/path/to/ckpt
    trainer.max_actor_ckpt_to_keep=5
)

REWARD=(
    reward.custom_reward_function.path=${reward_fn_path}
    reward.custom_reward_function.name=compute_score
)

CANN_ENV_PROPAGATE=()
CANN_ENV_NAMES=(PYTHONPATH LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH \
    ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_TOOLKIT_LATEST_HOME \
    ATB_HOME_PATH TOOLCHAIN_HOME CMAKE_PREFIX_PATH ASCEND_RUNTIME_OPTIONS \
    HCCL_OP_EXPANSION_MODE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE)
for _v in "${CANN_ENV_NAMES[@]}"; do
    _val="${!_v}"
    [ -z "$_val" ] && continue
    if [ "$_v" = "PYTHONPATH" ]; then
        _val="${VERL_ROOT}:${_val}"
    fi
    CANN_ENV_PROPAGATE+=("+ray_kwargs.ray_init.runtime_env.env_vars.${_v}='${_val}'")
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
    +distillation.teacher_models.gsm8k.inference.expert_parallel_size=${TEACHER_EP}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    +distillation.teacher_models.gsm8k.inference.enforce_eager=False
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.cudagraph_capture_sizes=[1]
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.prefill_context_parallel_size=${TEACHER_PCP}
    +distillation.teacher_models.gsm8k.inference.enable_chunked_prefill=False
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=1
    +distillation.teacher_models.gsm8k.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=${teacher_max_model_len}
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.kv_cache_dtype=auto
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.mamba_cache_dtype=float32
    +distillation.teacher_models.math.key="math"
    +distillation.teacher_models.math.model_path="$MATH_TEACHER_MODEL"
    +distillation.teacher_models.math.num_replicas=${TEACHER_NUM_REPLICAS_MATH}
    +distillation.teacher_models.math.inference.name=vllm
    +distillation.teacher_models.math.inference.tensor_model_parallel_size=${TEACHER_TP}
    +distillation.teacher_models.math.inference.data_parallel_size=${TEACHER_DP}
    +distillation.teacher_models.math.inference.expert_parallel_size=${TEACHER_EP}
    +distillation.teacher_models.math.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    +distillation.teacher_models.math.inference.enforce_eager=False
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.cudagraph_capture_sizes=[1]
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.prefill_context_parallel_size=${TEACHER_PCP}
    +distillation.teacher_models.math.inference.enable_chunked_prefill=False
    +distillation.teacher_models.math.inference.max_num_seqs=1
    +distillation.teacher_models.math.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.math.inference.max_num_batched_tokens=${teacher_max_model_len}
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.kv_cache_dtype=auto
    +distillation.teacher_models.math.inference.engine_kwargs.vllm.mamba_cache_dtype=float32
    # --- loss: task-reward RL + fullvocab distillation regularizer ---
    # L = policy_loss(task correctness) + distillation_loss_coef * fullvocab_forward_kl(q||p)
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.full_vocab_chunk_tokens=${full_vocab_chunk_tokens}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=2.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.distillation_loss_coef=0.5
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:256'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE='200'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE='200'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_TEACHER_DEDICATED_RACK='1'"
    "${CANN_ENV_PROPAGATE[@]}"
)

########################### launch ###########################
LOG_DIR=/path/to/k26_full_mopd_fullvocab
mkdir -p "${LOG_DIR}"
start_time=$(date +%Y%m%d_%H%M%S)
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/${start_time}_k26_full_mopd_2teacher_fullvocab.log"
