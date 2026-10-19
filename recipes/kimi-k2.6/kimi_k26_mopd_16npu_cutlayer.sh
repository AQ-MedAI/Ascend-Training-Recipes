#!/usr/bin/env bash
# Two-layer, two-teacher full-vocabulary MOPD smoke test on one 16-NPU node.
# The student pool uses eight NPUs and the two teacher replicas use eight NPUs.
# Prepopulate remote model code in the shared HF_HOME cache for offline workers.
# Full-vocabulary teachers require matching tokenizers and vocabulary dimensions.
set -xeuo pipefail
K26_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_CHECKPOINT_WRITER="${ASCEND_CHECKPOINT_WRITER:-inprocess}"
case "$ASCEND_CHECKPOINT_WRITER" in
    inprocess|multiprocess) ;;
    *) echo 'ASCEND_CHECKPOINT_WRITER must be inprocess or multiprocess' >&2; exit 2 ;;
esac

# ---- NPU environment ----
export HF_HOME=${HF_HOME:-/path/to/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
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

# All 16 dies (0..15) must be available. Keep the student and teacher pools
# disjoint: NGPUS_PER_NODE + TEACHER_WORLD_SIZE must not exceed this list.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL_PATH="${MODEL_PATH:-${CUT_MODEL_PATH:-${TMPDIR:-/tmp}/kimi_k26_bf16_2layers_model}}"
export FULL_MODEL_PATH="${FULL_MODEL_PATH:-/path/to/kimi-k2.6-bf16}"
PREPARE_ARGS=(--output-path "$MODEL_PATH" --num-layers 2)
if [[ -n "${FULL_MODEL_PATH:-}" ]]; then
    PREPARE_ARGS+=(--input-path "$FULL_MODEL_PATH")
fi
"$PYTHON_BIN" "$K26_SCRIPT_DIR/prepare_kimi_k26_cut_model.py" "${PREPARE_ARGS[@]}"
VERL_ROOT=${VERL_ROOT:-$(cd "$K26_SCRIPT_DIR/../.." && pwd)/third_party/verl}
cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

# ---- user-adjustable ----
HF_MODEL_PATH="$MODEL_PATH"
STUDENT_MODEL=${HF_MODEL_PATH}
# Smoke-test teachers reuse the cropped model. Replacement teachers must share
# the student's tokenizer and vocabulary so full-vocabulary log-probs align.
GSM8K_TEACHER_MODEL=${HF_MODEL_PATH}
ORZ_TEACHER_MODEL=${HF_MODEL_PATH}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Teacher pool = sum(num_replicas) * teacher_tp = 8 NPUs by default.
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_NUM_REPLICAS_GSM8K=${TEACHER_NUM_REPLICAS_GSM8K:-1}
TEACHER_NUM_REPLICAS_MBPP=${TEACHER_NUM_REPLICAS_MBPP:-1}
teacher_tp=${TEACHER_TP:-4}
TEACHER_WORLD_SIZE=$(( (TEACHER_NUM_REPLICAS_GSM8K + TEACHER_NUM_REPLICAS_MBPP) * teacher_tp ))

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-forward_kl_full_vocab}
full_vocab_chunk_tokens=${FULL_VOCAB_CHUNK_TOKENS:-2048}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# Smoke-test batch and sequence lengths.
# Length sizing standard:
#   - max_response_length caps the STUDENT rollout (not the teacher). Base it on the
#     teachers' ACTUAL response distribution, not their training cap.
#     mbpp teacher converged to resp mean ~800, max mostly <2k (cap was 11776, unused);
#     gsm8k teacher <2k. So 2048 covers mean x2.5 + student (8B) margin.
#   - max_prompt=2048 matches the gsm8k teacher's training prompt cap (actual prompts
#     are ~70-200 tokens, so this is pure safety margin).
#   - After launch, watch response_length/clip_ratio; if >0.2, raise max_response.
train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-1}
# ---- megatron parallelism ----
TP=${ACTOR_TP:-${TP:-2}}
PP=${ACTOR_PP:-${PP:-2}}
CP=${ACTOR_CP:-${CP:-2}}
EP=${ACTOR_EP:-${EP:-4}}
ETP=${ACTOR_ETP:-${ETP:-1}}
ROLLOUT_EP=${ROLLOUT_EP:-8}
ROLLOUT_DP=${ROLLOUT_DP:-1}
OFFLOAD=${OFFLOAD:-True}     # megatron param/optimizer offload

actor_ppo_max_token_len=$(( max_prompt_length + max_response_length + 1))
infer_ppo_max_token_len=$(( max_prompt_length + max_response_length + 1))
# Keep token budgets large enough for a complete sequence before CP partitioning.
infer_ppo_max_token_len=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$infer_ppo_max_token_len}
ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU:-$actor_ppo_max_token_len}}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-2048}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-1000000}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-8}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.5}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-1}
# Keep validation infrequent because full-vocabulary teacher inference is expensive.
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_distill_mopd_gsm8k_mbpp}
experiment_name=${EXPERIMENT_NAME:-k26_2teacher_mopd}

# Custom reward (now IN THE LOSS via use_task_rewards=True): gsm8k -> default answer check;
# mbpp -> verl.utils.reward_score.mbpp (executes generated code vs test_list asserts).
# reward_fn_path must be readable on the remote.
reward_fn_path=${REWARD_FN_PATH:-/path/to/orz_math_service_reward.py}
# ---- end user-adjustable ----

# Dataset A: gsm8k (data_source = openai/gsm8k) -> teacher gsm8k
# Dataset B: mbpp  (data_source = mbpp)            -> teacher mbpp
# Adjust paths / data_source to match your mbpp parquet.
gsm8k_train=/path/to/train.parquet
gsm8k_test=/path/to/test.parquet
mbpp_train=/path/to/orz_math_train.parquet
mbpp_test=/path/to/orz_math_train.parquet

train_files="['$gsm8k_train', '$mbpp_train']"
val_files="['$gsm8k_test', '$mbpp_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
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
    data.filter_overlong_prompts_workers=${FILTER_WORKERS:-4}
    data.truncation='error'
    data.shuffle=True
    data.train_max_samples=${TRAIN_MAX_SAMPLES:-8}
    data.val_max_samples=${VAL_MAX_SAMPLES:-8}
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
    actor_rollout_ref.actor.checkpoint.async_save=False
    actor_rollout_ref.actor.optim.betas=[0.9,0.999]
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.min_lr=1.0e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
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
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    # GRPO needs more than one sample per prompt for a nonzero group-relative advantage.
    actor_rollout_ref.rollout.n=4
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
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cudagraph_capture_sizes=[1,2,4,8,16,32]
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
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-2}
    trainer.default_local_dir=/path/to/ckpt
    # Keep enough checkpoints to compare smoke-test steps after the run.
    trainer.max_actor_ckpt_to_keep=30
)

REWARD=(
    reward.custom_reward_function.path=${reward_fn_path}
    reward.custom_reward_function.name=compute_score
)

CANN_ENV_PROPAGATE=()
CANN_ENV_NAMES=(LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH \
    ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_TOOLKIT_LATEST_HOME \
    ATB_HOME_PATH TOOLCHAIN_HOME CMAKE_PREFIX_PATH ASCEND_RUNTIME_OPTIONS \
    HCCL_OP_EXPANSION_MODE HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE)
for _v in "${CANN_ENV_NAMES[@]}"; do
    _val="${!_v:-}"
    [ -z "$_val" ] && continue
    if [ "$_v" = "PYTHONPATH" ]; then
        _val="${VERL_ROOT}:${_val}"
    fi
    CANN_ENV_PROPAGATE+=("+ray_kwargs.ray_init.runtime_env.env_vars.${_v}='${_val}'")
done
unset _v _val

# Multi-teacher: one teacher per dataset, routed by the sample's `data_source` value.
# Each teacher key MUST equal the data_source value in its parquet.
EXTRA=(
    model_engine=megatron
    +seed=42
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    # --- teacher A: gsm8k expert ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="$GSM8K_TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas=${TEACHER_NUM_REPLICAS_GSM8K}
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.enforce_eager=True
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    # +distillation.teacher_models.gsm8k.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.gsm8k.inference.enable_chunked_prefill=False
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=1
    +distillation.teacher_models.gsm8k.inference.max_model_len=$((max_prompt_length + max_response_length + 1))
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=$((max_prompt_length + max_response_length + 1))
    # --- teacher B: math expert ---
    +distillation.teacher_models.math.key="math"
    +distillation.teacher_models.math.model_path="$ORZ_TEACHER_MODEL"
    +distillation.teacher_models.math.num_replicas=${TEACHER_NUM_REPLICAS_MBPP}
    +distillation.teacher_models.math.inference.name=vllm
    +distillation.teacher_models.math.inference.enforce_eager=True
    +distillation.teacher_models.math.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.math.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    # +distillation.teacher_models.math.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.math.inference.enable_chunked_prefill=False
    +distillation.teacher_models.math.inference.max_num_seqs=1
    +distillation.teacher_models.math.inference.max_model_len=$((max_prompt_length + max_response_length + 1))
    +distillation.teacher_models.math.inference.max_num_batched_tokens=$((max_prompt_length + max_response_length + 1))
    # --- loss: task-reward RL + distillation regularizer ---
    # L = policy_loss(task correctness) + distillation_loss_coef * distill_loss(k1)
    # task reward (gsm8k answer / mbpp code) is the main driver; distill keeps student
    # near teacher. k1 + use_policy_gradient=True (PG OPD); clamp=2 caps distill advantage.
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.full_vocab_chunk_tokens=${full_vocab_chunk_tokens}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=2.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.distillation_loss_coef=0.5
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:256'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE='512'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE='256'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE='0'"
    "${CANN_ENV_PROPAGATE[@]}"
)

# Select the K26 runtime and propagate the same source paths to Ray workers.
source "${K26_SCRIPT_DIR}/runtime_env.sh"
EXTRA+=(
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_CHECKPOINT_WRITER='${ASCEND_CHECKPOINT_WRITER}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_ROOT='${MEGATRON_ROOT}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_BRIDGE_ROOT='${MEGATRON_BRIDGE_ROOT}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_VERBOSITY='${TRANSFORMERS_VERBOSITY}'"
)

########################### launch ###########################
start_time=$(date +%Y%m%d_%H%M%S)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee /path/to/k26_cutlayer_16npu_${start_time}_mopd_2teacher_fullvocab.log
