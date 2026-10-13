#!/usr/bin/env bash
# DeepSeek-V4-Flash MOPD smoke test on one Ascend node.
#
# Layout for a 16-card node:
#   - main pool:    8 NPUs, Megatron actor/ref + vLLM rollout
#   - teacher pool: 8 NPUs, one vLLM teacher
#
# The script uses MindSpeed-LLM for the actor/ref model construction so the
# DeepSeek-V4-Flash deepseek4_spec matches the checkpoint layout.  The MOPD
# full-vocabulary KL path is megatron-compatible and is enabled for MindSpeed
# in the local verl patch.

set -xeuo pipefail

if (( $# != 0 )); then
    printf 'This script does not accept Hydra overrides; edit an explicit test input instead.\n' >&2
    exit 2
fi

RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$RECIPE_ROOT/tools/v4_environment.sh"

if [[ -d "${MBRIDGE_ROOT}/mbridge" ]]; then
    MBRIDGE_OVERLAY="${MBRIDGE_OVERLAY:-${TMPDIR:-/tmp}/mbridge_overlay}"
    rm -rf "${MBRIDGE_OVERLAY}"
    mkdir -p "${MBRIDGE_OVERLAY}"
    cp -a "${MBRIDGE_ROOT}/mbridge" "${MBRIDGE_OVERLAY}/mbridge"
    printf '%s\n' 'from .deepseek_v3 import DeepseekV3Bridge' > "${MBRIDGE_OVERLAY}/mbridge/models/__init__.py"
    export PYTHONPATH="${MBRIDGE_OVERLAY}:${MBRIDGE_ROOT}:${PYTHONPATH}"
fi

if [[ -d "${VLLM_ROOT}/vllm" ]]; then
    export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH}"
fi

if [[ -d "${VLLM_ASCEND_ROOT}/vllm_ascend" ]]; then
    export PYTHONPATH="${VLLM_ASCEND_ROOT}:${PYTHONPATH}"
fi

if [[ -d "${VERL_ROOT}/verl" ]]; then
    export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH}"
fi

if [[ -d "${MEGATRON_ROOT}/megatron" ]]; then
    export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH}"
fi

if [[ -d "${MINDSPEED_ROOT}/mindspeed" ]]; then
    export PYTHONPATH="${MINDSPEED_ROOT}:${PYTHONPATH}"
fi

if [[ -d "${MINDSPEED_LLM_ROOT}/mindspeed_llm" ]]; then
    export PYTHONPATH="${MINDSPEED_LLM_ROOT}:${PYTHONPATH}"
fi

# ------------------------- Environment -------------------------
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export RAY_max_grpc_message_size=5368709120
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export HYBRID_IGNORE_LOAD_CHECK=1
export USE_MAX_V2=0
export VLLM_USE_V1=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
# Required by vLLM-Ascend online/layerwise weight reload.  NZ parameters use a
# kernel-specific layout and cannot be safely replaced with checkpoint-format
# actor weights.  The known-good DSV4 RL submission uses the same setting.
export VLLM_ASCEND_ENABLE_NZ=0
export TASK_QUEUE_ENABLE=0
export VLLM_ASCEND_TASK_QUEUE_ENABLE=0
export VLLM_DP_POLLING_LOAD_BALANCE_ENABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=0
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_EXEC_TIMEOUT=1200
export HCCL_EVENT_TIMEOUT=14500
export HCCL_ASYNC_ERROR_HANDLING=0
export ACL_DEVICE_SYNC_TIMEOUT=14400
export ACL_STREAM_TIMEOUT=14500000
# This smoke test colocates several independent HCCL domains (MindSpeed actor,
# vLLM rollout, and teacher) on the same NPUs.  Let the OS allocate distinct
# host/device listener ports; fixed ranges can make concurrent domains select
# the same first port and may eventually produce the invalid fallback 65536.
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_BUFFSIZE=200
export GROUP_HCCL_BUFFER_SIZE=200
export P2P_HCCL_BUFFSIZE=0
export HCCL_OP_EXPANSION_MODE=AI_CPU
export NCCL_ALGO=Ring
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VERL_MOPD_DEBUG="${VERL_MOPD_DEBUG:-1}"

# ------------------------- Paths -------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_MODEL_PATH="${FULL_MODEL_PATH:-/path/to/DeepSeek-V4-Flash}"
CUT_NUM_LAYERS=4
CUT_MODEL_OVERLAY="${CUT_MODEL_OVERLAY:-${TMPDIR:-/tmp}/dsv4_flash_bf16_4layers_model}"
FORCE_REBUILD_CUT_MODEL_OVERLAY="${FORCE_REBUILD_CUT_MODEL_OVERLAY:-False}"

TRANSFER_QUEUE_VERSION="$(
    "${PYTHON_BIN}" -c \
        'from importlib.metadata import version; import transfer_queue; print(version("TransferQueue"))' \
        2>/dev/null || true
)"
if [[ "${TRANSFER_QUEUE_VERSION}" != "0.1.7" ]]; then
    echo "ERROR: MOPD full-vocab transport requires TransferQueue==0.1.7, but found '${TRANSFER_QUEUE_VERSION:-not installed}'." >&2
    echo "Install it in the verl runtime environment: ${PYTHON_BIN} -m pip install 'TransferQueue==0.1.7'" >&2
    exit 1
fi

PREPARE_CUT_MODEL_ARGS=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_dsv4_flash_bf16_cut_model.py"
    --input-path "${FULL_MODEL_PATH}"
    --output-path "${CUT_MODEL_OVERLAY}"
    --num-layers 4
)
if [[ "${FORCE_REBUILD_CUT_MODEL_OVERLAY}" == "True" || "${FORCE_REBUILD_CUT_MODEL_OVERLAY}" == "true" ]]; then
    PREPARE_CUT_MODEL_ARGS+=(--force)
fi
"${PREPARE_CUT_MODEL_ARGS[@]}"
MODEL_PATH="${CUT_MODEL_OVERLAY}"

STUDENT_MODEL="${MODEL_PATH}"
TEACHER_MODEL="${MODEL_PATH}"

eval "$("${PYTHON_BIN:-python3}" - "${MODEL_PATH}/config.json" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

def first_int(*names, default):
    for name in names:
        value = cfg.get(name)
        if value is not None:
            return int(value)
    return int(default)

hidden_size = first_int("hidden_size", default=4096)
ffn_hidden_size = first_int("intermediate_size", "ffn_hidden_size", default=4096)
moe_ffn_hidden_size = first_int("moe_intermediate_size", "moe_ffn_hidden_size", default=2048)
num_attention_heads = first_int("num_attention_heads", default=64)
vocab_size = first_int("vocab_size", default=129280)
padded_vocab_size = first_int("padded_vocab_size", default=vocab_size)

values = {
    "DSV4_HIDDEN_SIZE": hidden_size,
    "DSV4_FFN_HIDDEN_SIZE": ffn_hidden_size,
    "DSV4_MOE_FFN_HIDDEN_SIZE": moe_ffn_hidden_size,
    "DSV4_NUM_ATTENTION_HEADS": num_attention_heads,
    "DSV4_VOCAB_SIZE": vocab_size,
    "DSV4_PADDED_VOCAB_SIZE": padded_vocab_size,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

TRAIN_FILES="${TRAIN_FILES:-/path/to/k26_rl_train.parquet}"
VAL_FILES="${VAL_FILES:-/path/to/k26_rl_valid.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-/path/to/k26_rl_reward.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/ckpt}"
LOG_DIR="${LOG_DIR:-/path/to/log}"
PROJECT_NAME="${PROJECT_NAME:-DeepSeek-V4-Flash-MOPD}"
EXP_NAME="${EXP_NAME:-dsv4_flash_${CUT_NUM_LAYERS}layers_mopd_16npu_smoke}"

MODEL_PATH="${MODEL_PATH%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
LOG_DIR="${LOG_DIR%/}"

# The following topology is the validated 16-NPU smoke-test layout.  These
# values are passed as literals below so inherited environment variables cannot
# silently select an untested actor, rollout, or teacher configuration.
# Main pool: 8 NPUs.  TP4 x CP2 leaves dense-model DP at 1, which matches the
# one-sample smoke-test batch.  Shard the 256 routed experts across all 8 ranks
# (32 local experts per rank); ETP1 x EP8 x PP1 = 8.
# Rollout runs in the same 8-NPU main pool as one TP8/EP8 replica.  This is the
# topology already verified by the original DeepSeek-V4-Flash RL run; two
# colocated TP4 replicas can stall in the Ray hybrid rollout path even though a
# standalone TP4 engine works.  EP must equal TP * DP.
# Teacher pool: one teacher world size is TP4 x DP2 = 8.

# ------------------------- Data And Lengths -------------------------
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 3))
# A full-vocab teacher must prefill the student prompt and response in one
# forward, then generate one token.  Align its capacity to TP because the
# Ascend sequence-parallel path pads scheduled tokens to a TP multiple.
TEACHER_REQUIRED_CONTEXT_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 3))
TEACHER_MAX_MODEL_LEN=$((((TEACHER_REQUIRED_CONTEXT_LEN + 3) / 4) * 4))
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-128}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
FILTER_WORKERS="${FILTER_WORKERS:-4}"

TRAIN_SEQ_LENGTH="${MAX_MODEL_LEN}"
ACTOR_MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1) / 2))
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"

# ------------------------- Optimization / Distillation -------------------------
LR="${LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
DISTILLATION_LOSS_COEF="${DISTILLATION_LOSS_COEF:-0.5}"

# ------------------------- Rollout / Trainer -------------------------
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
LOGGER="${LOGGER:-[\"console\"]}"
SEED="${SEED:-42}"

# DeepSeek-V4-Flash tokenizer can ship without tokenizer.chat_template.  verl's
# RL dataset and AgentLoop both call apply_chat_template(), so provide a compact
# text-only fallback for the smoke test.  Override CHAT_TEMPLATE if you have the
# model's official template.
DEFAULT_CHAT_TEMPLATE='{% for message in messages %}{% if message["role"] == "system" %}{{ message["content"] }}{{ "\n" }}{% elif message["role"] == "user" %}{{ message["content"] }}{% elif message["role"] == "assistant" %}{{ "\n" }}{{ message["content"] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ "\n" }}{% endif %}'
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${DEFAULT_CHAT_TEMPLATE}}"

cd "${VERL_ROOT}"

DATA_CONFIG=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.prompt_key=prompt
    data.train_batch_size=1
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers="${FILTER_WORKERS}"
    data.truncation=left
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"
    data.shuffle=True
    data.trust_remote_code=True
    "+data.apply_chat_template_kwargs.chat_template='${CHAT_TEMPLATE}'"
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.trust_remote_code=True
    "actor_rollout_ref.model.custom_chat_template='${CHAT_TEMPLATE}'"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.actor.mindspeed.use_remove_padding=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.loss_type=ppo_clip
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.ppo_mini_batch_size=1
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}"
    actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD}"
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=4
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.mindspeed.context_parallel_size=2
    actor_rollout_ref.actor.mindspeed.expert_model_parallel_size=8
    actor_rollout_ref.actor.mindspeed.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.mindspeed.param_offload=True
    actor_rollout_ref.actor.mindspeed.optimizer_offload=True
    actor_rollout_ref.actor.mindspeed.grad_offload=True
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    actor_rollout_ref.actor.mindspeed.use_dist_checkpointing=True
    +actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=False
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]
    actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
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
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.max_batch_size=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.compress_ratios="[0,0,4,128]"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_permutation_async_comm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_layer_freq=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.first_k_dense_replace=-1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_experts=256
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk=6
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_ffn_hidden_size="${DSV4_MOE_FFN_HIDDEN_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_intermediate_size="${DSV4_MOE_FFN_HIDDEN_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.deprecated_moe_intermediate_size="${DSV4_MOE_FFN_HIDDEN_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_load_balancing_type=none
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_group_topk=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_num_groups=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk_scaling_factor=1.5
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_aux=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_score_function=sqrtsoftplus
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_intermediate_size="${DSV4_MOE_FFN_HIDDEN_SIZE}"
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
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.manual_gc_interval=10
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_shared_storage=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_distributed_optimizer=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_flash_attn=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_mcore_models=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_layers="${CUT_NUM_LAYERS}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.hidden_size="${DSV4_HIDDEN_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.ffn_hidden_size="${DSV4_FFN_HIDDEN_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_attention_heads="${DSV4_NUM_ATTENTION_HEADS}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_type=PretrainedFromHF
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_name_or_path="${MODEL_PATH}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length="${TRAIN_SEQ_LENGTH}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.max_position_embeddings=1048576
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.global_batch_size=1
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
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.vocab_size="${DSV4_VOCAB_SIZE}"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.padded_vocab_size="${DSV4_PADDED_VOCAB_SIZE}"
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
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.context_parallel_algo=kvallgather_cp_algo
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.masked_softmax_fusion=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_overlap=False
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.mindspeed.tensor_model_parallel_size=4
    actor_rollout_ref.ref.mindspeed.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.mindspeed.context_parallel_size=2
    actor_rollout_ref.ref.mindspeed.expert_model_parallel_size=8
    actor_rollout_ref.ref.mindspeed.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.mindspeed.param_offload=True
    actor_rollout_ref.ref.mindspeed.use_dist_checkpointing=False
    actor_rollout_ref.ref.mindspeed.use_mbridge=True
    actor_rollout_ref.ref.mindspeed.vanilla_mbridge=True
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
    actor_rollout_ref.rollout.max_num_seqs=1
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1,2,4,8,16]"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size=1
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.top_p="${TOP_P}"
    actor_rollout_ref.rollout.top_k="${TOP_K}"
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.tensor_model_parallel_size=8
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=8
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
)

TEACHER_CONFIG=(
    distillation.enabled=True
    distillation.share_gpu_group=False
    distillation.n_gpus_per_node=8
    distillation.nnodes=1
    distillation.teacher_key=data_source
    distillation.distillation_loss.loss_mode=forward_kl_full_vocab
    distillation.distillation_loss.full_vocab_chunk_tokens=128
    distillation.distillation_loss.full_vocab_lm_head_layer=head.weight
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.loss_max_clamp=2.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.distillation_loss_coef="${DISTILLATION_LOSS_COEF}"
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="${TEACHER_MODEL}"
    +distillation.teacher_models.gsm8k.num_replicas=1
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.data_parallel_size=2
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=4
    +distillation.teacher_models.gsm8k.inference.expert_parallel_size=8
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.45
    +distillation.teacher_models.gsm8k.inference.max_model_len="${TEACHER_MAX_MODEL_LEN}"
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens="${TEACHER_MAX_MODEL_LEN}"
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=1
    +distillation.teacher_models.gsm8k.inference.enforce_eager=False
    ++distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4]"
    +distillation.teacher_models.gsm8k.inference.enable_chunked_prefill=False
    ++distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.additional_config.enable_dsa_cp=True
    +distillation.teacher_models.gsm8k.inference.enable_prefix_caching=False
    +distillation.teacher_models.gsm8k.inference.free_cache_engine=False
    +distillation.teacher_models.gsm8k.inference.enable_sleep_mode=True
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.prefill_context_parallel_size=1
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.kv_cache_dtype=auto
)

TRAINER_CONFIG=(
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXP_NAME}"
    trainer.nnodes=1
    trainer.n_gpus_per_node=8
    trainer.device=npu
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
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

CANN_ENV_PROPAGATE=()
CANN_ENV_NAMES=(
    RECIPE_ROOT DEPENDENCY_ROOT VERL_ROOT MINDSPEED_ROOT MINDSPEED_LLM_ROOT MBRIDGE_ROOT MEGATRON_ROOT VLLM_ROOT VLLM_ASCEND_ROOT ASCEND_CUSTOM_OPP_PATH
    PYTHONPATH LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH
    ASCEND_TOOLKIT_HOME ASCEND_TOOLKIT_LATEST_HOME ATB_HOME_PATH TOOLCHAIN_HOME
    CMAKE_PREFIX_PATH ASCEND_RUNTIME_OPTIONS HCCL_OP_EXPANSION_MODE HF_HOME
    HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PYTORCH_NPU_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
    HCCL_BUFFSIZE GROUP_HCCL_BUFFER_SIZE P2P_HCCL_BUFFSIZE HCCL_NPU_SOCKET_PORT_RANGE
    HCCL_HOST_SOCKET_PORT_RANGE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT HCCL_EVENT_TIMEOUT
    HCCL_ASYNC_ERROR_HANDLING ACL_DEVICE_SYNC_TIMEOUT ACL_STREAM_TIMEOUT
    VLLM_ASCEND_ENABLE_FLASHCOMM1 VLLM_ASCEND_ENABLE_NZ TASK_QUEUE_ENABLE
    VLLM_ASCEND_TASK_QUEUE_ENABLE VLLM_WORKER_MULTIPROC_METHOD VLLM_USE_V1
    VLLM_SERVER_DEV_MODE VLLM_ALLOW_INSECURE_SERIALIZATION
    VLLM_DP_POLLING_LOAD_BALANCE_ENABLE VLLM_RPC_TIMEOUT
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS TORCHDYNAMO_DISABLE TORCH_COMPILE_DISABLE
    NCCL_CUMEM_ENABLE CUDA_DEVICE_MAX_CONNECTIONS HYDRA_FULL_ERROR
    RAY_memory_monitor_refresh_ms RAY_memory_usage_threshold RAY_max_grpc_message_size
    VERL_MOPD_DEBUG OMP_PROC_BIND OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
)
for _v in "${CANN_ENV_NAMES[@]}"; do
    _val="${!_v-}"
    [[ -z "${_val}" ]] && continue
    CANN_ENV_PROPAGATE+=("+ray_kwargs.ray_init.runtime_env.env_vars.${_v}='${_val}'")
done
unset _v _val

EXTRA_CONFIG=(
    --config-name=ppo_trainer.yaml
    model_engine=mindspeed
    +seed="${SEED}"
    "${CANN_ENV_PROPAGATE[@]}"
)

# ------------------------- Launch -------------------------
mkdir -p "${LOG_DIR}"
START_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${START_TIME}_${EXP_NAME}.log"

CMD=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    "${EXTRA_CONFIG[@]}"
    "${DATA_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${ACTOR_CONFIG[@]}"
    "${REF_CONFIG[@]}"
    "${ROLLOUT_CONFIG[@]}"
    "${ALGORITHM_CONFIG[@]}"
    "${TEACHER_CONFIG[@]}"
    "${TRAINER_CONFIG[@]}"
    "${REWARD_CONFIG[@]}"
)

printf 'Using model path: %s\n' "${MODEL_PATH}"
printf 'Logging to %s\n' "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
