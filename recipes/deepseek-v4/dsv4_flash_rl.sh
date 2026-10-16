#!/usr/bin/env bash
# DeepSeek-V4-Flash GRPO training with verl + MindSpeed + vLLM.

set -xeuo pipefail

# ------------------------- Runtime Helpers -------------------------
QUIET_SITE_DIR="${TMPDIR:-/tmp}/verl_quiet_sitecustomize"
mkdir -p "${QUIET_SITE_DIR}"

cat > "${QUIET_SITE_DIR}/sitecustomize.py" <<'PY'
import logging
import warnings

class _TransformersAliasFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if (
            record.name.startswith("transformers")
            and msg.startswith("Accessing `")
            and "image_processing_pil_" in msg
            and "alias will be removed in future versions" in msg
        ):
            return False
        return True

logging.getLogger("transformers").addFilter(_TransformersAliasFilter())

warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `.*` from `\.models\..*image_processing_pil_.*`.*",
)
PY

RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$RECIPE_ROOT/tools/v4_environment.sh"
MBRIDGE_OVERLAY="${TMPDIR:-/tmp}/mbridge_overlay"

if [[ -d "${MBRIDGE_ROOT}/mbridge" ]]; then
    rm -rf "${MBRIDGE_OVERLAY}"
    mkdir -p "${MBRIDGE_OVERLAY}"
    cp -a "${MBRIDGE_ROOT}/mbridge" "${MBRIDGE_OVERLAY}/mbridge"
    printf '%s\n' 'from .deepseek_v3 import DeepseekV3Bridge' > "${MBRIDGE_OVERLAY}/mbridge/models/__init__.py"
    export PYTHONPATH="${MBRIDGE_OVERLAY}:${MBRIDGE_ROOT}:${QUIET_SITE_DIR}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${QUIET_SITE_DIR}:${PYTHONPATH:-}"
fi

if [[ -d "${VLLM_ROOT}/vllm" ]]; then
    export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH}"
fi

if [[ -d "${VLLM_ASCEND_ROOT}/vllm_ascend" ]]; then
    export PYTHONPATH="${VLLM_ASCEND_ROOT}:${PYTHONPATH}"
fi

# ------------------------- Environment -------------------------
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export HYBRID_IGNORE_LOAD_CHECK="${HYBRID_IGNORE_LOAD_CHECK:-1}"
export USE_MAX_V2="${USE_MAX_V2:-0}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:.*Accessing `.*` from `\\.models\\..*image_processing_pil_.*`:Warning}"

# ------------------------- Paths -------------------------
# Model, data and output paths can be supplied through environment variables.
PROJECT_NAME="${PROJECT_NAME:-DeepSeek-V4-Flash}"
EXP_NAME="${EXP_NAME:-DeepSeek-V4-Flash-8-nodes}"
MODEL_PATH="${MODEL_PATH:-/path/to/deepseek-v4-flash-0731-bf16}"
TRAIN_FILES="${TRAIN_FILES:-/path/to/k26_rl_train.parquet}"
VAL_FILES="${VAL_FILES:-/path/to/k26_rl_valid.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-/path/to/k26_rl_reward.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/ckpt}"
LOG_DIR="${LOG_DIR:-/path/to/log}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
LOG_DIR="${LOG_DIR%/}"

# ------------------------- Cluster -------------------------
NNODES="${NNODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPUS_PER_NODE:-16}}"

# ------------------------- Data And Lengths -------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# ------------------------- Algorithm / Optimization -------------------------
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
KL_COEF="${KL_COEF:-0.0}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
ALL_OFFLOAD="${ALL_OFFLOAD:-True}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-True}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-False}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
SWAP_OPTIMIZER="${SWAP_OPTIMIZER:-False}"
SWAP_OPTIMIZER_TIMES="${SWAP_OPTIMIZER_TIMES:-64}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
BYPASS_RECOMPUTING_LOGPROBS="${BYPASS_RECOMPUTING_LOGPROBS:-False}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MS_MAX_BATCH_SIZE="${MS_MAX_BATCH_SIZE:-1}"
MS_MANUAL_GC_INTERVAL="${MS_MANUAL_GC_INTERVAL:-10}"
MOE_SHARED_EXPERT_OVERLAP="${MOE_SHARED_EXPERT_OVERLAP:-False}"
CONTEXT_PARALLEL_ALGO="${CONTEXT_PARALLEL_ALGO:-kvallgather_cp_algo}"

# ------------------------- Parallelism -------------------------
ACTOR_TP="${ACTOR_TP:-4}"
ACTOR_PP="${ACTOR_PP:-2}"
ACTOR_CP="${ACTOR_CP:-8}"
ACTOR_EP="${ACTOR_EP:-64}"
ACTOR_ETP="${ACTOR_ETP:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-8}"
ROLLOUT_DP="${ROLLOUT_DP:-16}"
ROLLOUT_EP="${ROLLOUT_EP:-128}"
# DeepSeek-V4 uses sliding-window KV cache, which this vLLM-Ascend build
# currently rejects when decode context parallelism is greater than one.
ROLLOUT_DCP="${ROLLOUT_DCP:-1}"
TRAIN_SEQ_LENGTH="${TRAIN_SEQ_LENGTH:-${MAX_MODEL_LEN}}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_MODEL_LEN + ACTOR_CP - 1) / ACTOR_CP))}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_MODEL_LEN + ACTOR_CP - 1) / ACTOR_CP))}"

# ------------------------- Rollout / Trainer -------------------------
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LOGGER="${LOGGER:-[\"console\", \"tensorboard\", \"wandb\"]}"
SEED="${SEED:-42}"

DATA_CONFIG=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.prompt_key=prompt
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=left
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.actor.mindspeed.use_remove_padding=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator="${ADV_ESTIMATOR}"
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}"
    algorithm.kl_ctrl.kl_coef="${KL_COEF}"
    algorithm.rollout_correction.bypass_mode="${BYPASS_RECOMPUTING_LOGPROBS}"
    algorithm.rollout_correction.loss_type=ppo_clip
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size="${ACTOR_PP}"
    actor_rollout_ref.actor.mindspeed.context_parallel_size="${ACTOR_CP}"
    actor_rollout_ref.actor.mindspeed.expert_model_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.actor.mindspeed.expert_tensor_parallel_size="${ACTOR_ETP}"
    actor_rollout_ref.actor.mindspeed.param_offload="${ALL_OFFLOAD}"
    actor_rollout_ref.actor.mindspeed.optimizer_offload=False
    actor_rollout_ref.actor.mindspeed.grad_offload="${ALL_OFFLOAD}"
    actor_rollout_ref.actor.mindspeed.use_dist_checkpointing=True
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload="${OPTIMIZER_CPU_OFFLOAD}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPTIMIZER}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="${OPTIMIZER_OFFLOAD_FRACTION}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer="${SWAP_OPTIMIZER}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer_times="${SWAP_OPTIMIZER_TIMES}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.enable_dsa_indexer=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.index_n_heads=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.index_head_dim=128
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.index_topk=512
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.hc_mult=4
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.enable_mhc=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.kv_compress=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.norm_eps=1e-6
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.multi_latent_attention=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_pos_emb_head_dim=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_head_dim=512
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.q_lora_rank=1024
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.o_lora_rank=1024
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.kv_lora_rank=512
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.v_head_dim=128
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_layernorm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mla_fa_without_pad=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_g2_attention=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.o_groups=8
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.g2_window_size=128
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_head_dim=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.original_seq_len=65536
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_factor=16
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.compress_rope_theta=160000.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.max_batch_size="${MS_MAX_BATCH_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.compress_ratios="[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4]"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_permutation_async_comm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_layer_freq=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.first_k_dense_replace=-1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_experts=256
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk=6
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_ffn_hidden_size=2048
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_load_balancing_type=none
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_group_topk=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_num_groups=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk_scaling_factor=1.5
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_aux=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_score_function=sqrtsoftplus
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_intermediate_size=2048
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.fix_router=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_dtype=fp32
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.n_hash_layers=3
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_num_layers=0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_loss_scaling_factor=0.3
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_mem_efficient_logits=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.beta_fast=32
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.beta_slow=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_factor=16
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_mscale=1.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_mscale_all_dim=1.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_original_max_position_embeddings=65536
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_theta=10000.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_type=yarn
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.transformer_impl=local
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.spec="['mindspeed_llm.tasks.models.spec.deepseek4_spec', 'layer_spec']"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.manual_gc=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.manual_gc_interval="${MS_MANUAL_GC_INTERVAL}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_shared_storage=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_distributed_optimizer=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_flash_attn=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_mcore_models=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_layers=43
    '+actor_rollout_ref.actor.mindspeed.llm_kwargs.num_layer_list="21,22"'
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.hidden_size=4096
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.ffn_hidden_size=4096
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_attention_heads=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_type=PretrainedFromHF
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_name_or_path="${MODEL_PATH}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length="${TRAIN_SEQ_LENGTH}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.max_position_embeddings=1048576
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.global_batch_size="${TRAIN_BATCH_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.make_vocab_size_divisible_by=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.lr="${LR}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.train_iters=2000
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.lr_decay_style=constant
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.untie_embeddings_and_output_weights=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.disable_bias_linear=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.add_bias_linear=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.attention_dropout=0.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.init_method_std=0.02
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.hidden_dropout=0.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.position_embedding_type=deepseek4
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.normalization=RMSNorm
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_swiglu=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_rmsnorm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.swiglu=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.swiglu_limit=10.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_masked_softmax_fusion=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.min_lr="${LR}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.weight_decay="${WEIGHT_DECAY}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.clip_grad="${CLIP_GRAD}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.adam_beta1=0.9
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.adam_beta2=0.95
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.initial_loss_scale=65536
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.vocab_size=129280
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.padded_vocab_size=129280
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rotary_base=10000
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.norm_epsilon=1e-6
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_load_optim=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_load_rng=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.bf16=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.distributed_timeout_minutes=120
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.gradient_accumulation_fusion=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_save_optim=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_save_rng=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.context_parallel_algo="${CONTEXT_PARALLEL_ALGO}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.masked_softmax_fusion=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_overlap="${MOE_SHARED_EXPERT_OVERLAP}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.indexer_loss_coeff=0.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_sfa=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_mhc=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_sinkhorn=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_rmsnorm_without_weight=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_pad_to_seq_lengths=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_lightning_indexer=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_lightning_indexer_loss=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_sparse_flash_attn=True
)

REF_CONFIG=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.mindspeed.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.ref.mindspeed.pipeline_model_parallel_size="${ACTOR_PP}"
    actor_rollout_ref.ref.mindspeed.context_parallel_size="${ACTOR_CP}"
    actor_rollout_ref.ref.mindspeed.expert_model_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.ref.mindspeed.expert_tensor_parallel_size="${ACTOR_ETP}"
    actor_rollout_ref.ref.mindspeed.param_offload="${ALL_OFFLOAD}"
    actor_rollout_ref.ref.mindspeed.use_dist_checkpointing=False
    actor_rollout_ref.ref.mindspeed.use_mbridge=True
    actor_rollout_ref.ref.mindspeed.vanilla_mbridge=True
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1,2,4,8,16,32]"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size="${ROLLOUT_DCP}"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_dsa_cp=True
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.n="${N_SAMPLES}"
    actor_rollout_ref.rollout.top_p="${TOP_P}"
    actor_rollout_ref.rollout.top_k="${TOP_K}"
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
)

TRAINER_CONFIG=(
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXP_NAME}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${GPUS_PER_NODE}"
    trainer.device=npu
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.val_before_train=False
    trainer.test_freq="${TEST_FREQ}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.default_local_dir="${OUTPUT_DIR}/${PROJECT_NAME}/${EXP_NAME}"
    trainer.use_legacy_worker_impl=disable
)

REWARD_CONFIG=(
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
)

EXTRA_CONFIG=(
    --config-name=ppo_trainer.yaml
    model_engine=mindspeed
    +seed="${SEED}"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE="'200'"
    +ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE="'200'"
    +ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE="'0'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_FLASHCOMM1="'${VLLM_ASCEND_ENABLE_FLASHCOMM1}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'${PYTHONPATH}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.RECIPE_ROOT="'${RECIPE_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.DEPENDENCY_ROOT="'${DEPENDENCY_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROOT="'${VERL_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.MINDSPEED_ROOT="'${MINDSPEED_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.MINDSPEED_LLM_ROOT="'${MINDSPEED_LLM_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.MBRIDGE_ROOT="'${MBRIDGE_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_ROOT="'${MEGATRON_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ROOT="'${VLLM_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ROOT="'${VLLM_ASCEND_ROOT:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="'${LD_LIBRARY_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_CUSTOM_OPP_PATH="'${ASCEND_CUSTOM_OPP_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_HOME_PATH="'${ASCEND_HOME_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_OPP_PATH="'${ASCEND_OPP_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_AICPU_PATH="'${ASCEND_AICPU_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ATB_HOME_PATH="'${ATB_HOME_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_VERBOSITY="'error'"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_NPU_SOCKET_PORT_RANGE="'${HCCL_NPU_SOCKET_PORT_RANGE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_HOST_SOCKET_PORT_RANGE="'${HCCL_HOST_SOCKET_PORT_RANGE}'"
)

# ------------------------- Launch -------------------------
mkdir -p "${LOG_DIR}"
START_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${START_TIME}_${EXP_NAME}_verl_grpo_${NNODES}N.log"

CMD=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    "${EXTRA_CONFIG[@]}"
    "${DATA_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${ACTOR_CONFIG[@]}"
    "${REF_CONFIG[@]}"
    "${ROLLOUT_CONFIG[@]}"
    "${ALGORITHM_CONFIG[@]}"
    "${TRAINER_CONFIG[@]}"
    "${REWARD_CONFIG[@]}"
    "$@"
)

printf 'Logging to %s\n' "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
