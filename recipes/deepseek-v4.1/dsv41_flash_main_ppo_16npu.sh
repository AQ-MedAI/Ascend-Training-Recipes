#!/usr/bin/env bash
# Native verl entry: first container acceptance required; retained smoke entry is separate.
set -xeuo pipefail

if (( $# != 0 )); then
    printf 'This script does not accept Hydra overrides; edit an explicit test input instead.\n' >&2
    exit 2
fi

# Fork sources default to the pinned submodules; external runtimes use DEPENDENCY_ROOT.
RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-/path/to/dsv4_flash_npu_dependency}"
VERL_ROOT="${VERL_ROOT:-${RECIPE_ROOT}/third_party/verl}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${DEPENDENCY_ROOT}/Megatron-LM}"
MINDSPEED_ROOT="${MINDSPEED_ROOT:-${RECIPE_ROOT}/third_party/MindSpeed}"
MINDSPEED_LLM_ROOT="${MINDSPEED_LLM_ROOT:-${RECIPE_ROOT}/third_party/MindSpeed-LLM}"
MBRIDGE_ROOT="${MBRIDGE_ROOT:-${RECIPE_ROOT}/third_party/mbridge}"
VLLM_ROOT="${VLLM_ROOT:-${DEPENDENCY_ROOT}/vllm}"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${DEPENDENCY_ROOT}/vllm-ascend}"

export PYTHONPATH="${PYTHONPATH:-}"

if [[ -d "${MBRIDGE_ROOT}/mbridge" ]]; then
    export PYTHONPATH="${MBRIDGE_ROOT}:${PYTHONPATH}"
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
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-250}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.95}"
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
export VLLM_ASCEND_ENABLE_NZ=0
export TASK_QUEUE_ENABLE=0
export VLLM_ASCEND_TASK_QUEUE_ENABLE=0
# This smoke test colocates several independent HCCL domains (MindSpeed actor,
# and vLLM rollout) on the same NPUs.  Let the OS allocate distinct
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

# ------------------------- Paths -------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${WORK_DIR:-${DEPENDENCY_ROOT}}"
ENGRAM_HASH_SOURCE="${ENGRAM_HASH_SOURCE:-${MINDSPEED_LLM_ROOT}/mindspeed_llm/tasks/models/transformer/deepseek41/engram_hash.py}"
if [[ ! -f "$ENGRAM_HASH_SOURCE" ]]; then
    printf 'Missing Engram hash implementation: %s; use a MindSpeed-LLM checkout with V4.1 Engram support, or set ENGRAM_HASH_SOURCE.\n' "$ENGRAM_HASH_SOURCE" >&2
    exit 1
fi
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH}"
REPORT_DIR="${REPORT_DIR:-$ROOT/workspace/reports/main-ppo-$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="${LOG_FILE:-$REPORT_DIR/launch.log}"
TRAIN_FILES="${TRAIN_FILES:-/path/to/k26_rl_train.parquet}"
VAL_FILES="${VAL_FILES:-/path/to/k26_rl_valid.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-/path/to/k26_rl_reward.py}"
MODEL="${MODEL:-/path/to/dsv41-smoke-40L-16E-BF16}"
if [[ -z "${ENGRAM_CROP+x}" ]]; then
    ENGRAM_CROP=""
    [[ ! -d "$MODEL/engram-grpo-crop" ]] || ENGRAM_CROP="$MODEL/engram-grpo-crop"
fi
export ENGRAM_CROP
export ENGRAM_TABLE_TRAINABLE="${ENGRAM_TABLE_TRAINABLE:-True}"
case "$ENGRAM_TABLE_TRAINABLE" in
    1|true|True) ENGRAM_TABLE_TRAINABLE=True ;;
    0|false|False) ENGRAM_TABLE_TRAINABLE=False ;;
    *) echo 'ENGRAM_TABLE_TRAINABLE must be 0/1 or True/False' >&2; exit 2 ;;
esac
if [[ -n "$ENGRAM_CROP" ]]; then
    export ENGRAM_SOURCE="${ENGRAM_SOURCE:-/path/to/deepseek-v4.1-flash-0910-BF16}"
else
    export ENGRAM_SOURCE="${ENGRAM_SOURCE:-$MODEL}"
fi
export ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-$ROOT/workspace/rollout/dsv41/.venv/bin/python}"
export V41_NATIVE_TRAINER=1
if [[ -n "$ENGRAM_CROP" ]]; then
    ROLLOUT_TP="${ROLLOUT_TP:-8}"
    ROLLOUT_MEMORY_UTILIZATION="${ROLLOUT_MEMORY_UTILIZATION:-0.6}"
else
    ROLLOUT_TP="${ROLLOUT_TP:-8}"
    ROLLOUT_MEMORY_UTILIZATION="${ROLLOUT_MEMORY_UTILIZATION:-0.35}"
fi
# Native training imports verl rollout classes too; resolve the base V4 packages.
# V41Replica separately prepends the pinned V4.1 sources for rollout servers.

# ------------------------- Data And Lengths -------------------------
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 3))
FILTER_WORKERS="${FILTER_WORKERS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-128}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1) / 2))}"

# ------------------------- Optimization ------------------------------
LR="${LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"

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

# DeepSeek-V4.1-Flash tokenizer can ship without tokenizer.chat_template.  verl's
# RL dataset and AgentLoop both call apply_chat_template(), so provide a compact
# text-only fallback for the smoke test.  Override CHAT_TEMPLATE if you have the
# model's official template.
DEFAULT_CHAT_TEMPLATE='{% for message in messages %}{% if message["role"] == "system" %}{{ message["content"] }}{{ "\n" }}{% elif message["role"] == "user" %}{{ message["content"] }}{% elif message["role"] == "assistant" %}{{ "\n" }}{{ message["content"] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ "\n" }}{% endif %}'
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${DEFAULT_CHAT_TEMPLATE}}"

# Encode the template as a Hydra string; shell quotes alone do not protect its grammar.
CHAT_TEMPLATE="$("${PYTHON_BIN:-python3}" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$CHAT_TEMPLATE")"

# ------------------------- Configuration -------------------------
DATA_CONFIG=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.prompt_key=prompt
    data.train_batch_size="${TRAIN_BATCH_SIZE:-1}"
    data.dataloader_num_workers="${DATALOADER_WORKERS}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers="${FILTER_WORKERS}"
    data.truncation=left
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"
    data.shuffle=True
    data.trust_remote_code=True
    "+data.apply_chat_template_kwargs.chat_template=${CHAT_TEMPLATE}"
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="$MODEL"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.tokenizer_path="$ENGRAM_SOURCE"
    actor_rollout_ref.model.external_lib=verl.utils.deepseek_v41_config
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.custom_chat_template="$CHAT_TEMPLATE"
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    ++algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.loss_type=ppo_clip
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-1}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}"
    actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD}"
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    ++actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0
    ++actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
    +actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=True
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]
    actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=8
    actor_rollout_ref.actor.mindspeed.context_parallel_size=2
    actor_rollout_ref.actor.mindspeed.expert_model_parallel_size=16
    actor_rollout_ref.actor.mindspeed.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.mindspeed.sequence_parallel=True
    actor_rollout_ref.actor.mindspeed.use_distributed_optimizer=True
    actor_rollout_ref.actor.mindspeed.use_dist_checkpointing=True
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    actor_rollout_ref.actor.mindspeed.param_offload=True
    actor_rollout_ref.actor.mindspeed.grad_offload=True
    actor_rollout_ref.actor.mindspeed.optimizer_offload=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_crop="$ENGRAM_CROP"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_table_trainable="$ENGRAM_TABLE_TRAINABLE"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_source="$ENGRAM_SOURCE"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_hash_source="$ENGRAM_HASH_SOURCE"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer_times=64
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.indexer_loss_coeff="${INDEXER_LOSS_COEFF:-0.0}"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length="$MAX_MODEL_LEN"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.transformer_impl=local
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.normalization=RMSNorm
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_shared_storage=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_flash_attn=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_type=PretrainedFromHF
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_name_or_path="${MODEL}"
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${TOP_P}"
    actor_rollout_ref.rollout.top_k="${TOP_K}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.max_num_seqs=1
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${WEIGHT_BUCKET_MB:-3072}"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.safetensors_load_strategy=lazy
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_engram=False
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0
)

REWARD_CONFIG=(
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
)

TRAINER_CONFIG=(
    trainer.nnodes="${NNODES:-1}"
    trainer.resume_mode="${RESUME_MODE:-auto}"
    trainer.n_gpus_per_node=16
    trainer.device=npu
    trainer.use_legacy_worker_impl=disable
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.val_before_train=False
    trainer.test_freq="${TEST_FREQ}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.logger="${LOGGER}"
    trainer.default_local_dir="$REPORT_DIR"
)

EXTRA_CONFIG=(
    --config-name=ppo_trainer.yaml
    model_engine=mindspeed
    +seed="${SEED}"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE="'${HCCL_BUFFSIZE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE="'${GROUP_HCCL_BUFFER_SIZE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE="'${P2P_HCCL_BUFFSIZE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_FLASHCOMM1="'${VLLM_ASCEND_ENABLE_FLASHCOMM1}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'${PYTHONPATH}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_VERBOSITY="'${TRANSFORMERS_VERBOSITY}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_NPU_SOCKET_PORT_RANGE="'${HCCL_NPU_SOCKET_PORT_RANGE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_HOST_SOCKET_PORT_RANGE="'${HCCL_HOST_SOCKET_PORT_RANGE}'"
)

# ------------------------- Launch -------------------------
ARGS=(
    "${EXTRA_CONFIG[@]}"
    "${DATA_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${ALGORITHM_CONFIG[@]}"
    "${ACTOR_CONFIG[@]}"
    "${ROLLOUT_CONFIG[@]}"
    "${REWARD_CONFIG[@]}"
    "${TRAINER_CONFIG[@]}"
)

mkdir -p "$REPORT_DIR"

printf 'Using model path: %s\n' "${MODEL}"
printf 'Logging to %s\n' "${LOG_FILE}"
"${PYTHON_BIN}" -m verl.trainer.main_ppo "${ARGS[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
