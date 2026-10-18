#!/usr/bin/env bash
set -xeuo pipefail

if (( $# != 0 )); then
    printf 'This script does not accept Hydra overrides; edit an explicit test input instead.\n' >&2
    exit 2
fi

# Frameworks default to the prepared image; individual roots support development overrides.
RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-/opt/ascend-training}"
VERL_ROOT="${VERL_ROOT:-${DEPENDENCY_ROOT}/third_party/verl}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${DEPENDENCY_ROOT}/third_party/Megatron-LM}"
MINDSPEED_ROOT="${MINDSPEED_ROOT:-${DEPENDENCY_ROOT}/third_party/MindSpeed}"
MINDSPEED_LLM_ROOT="${MINDSPEED_LLM_ROOT:-${DEPENDENCY_ROOT}/third_party/MindSpeed-LLM}"
MBRIDGE_ROOT="${MBRIDGE_ROOT:-${DEPENDENCY_ROOT}/third_party/mbridge}"
VLLM_ROOT="${VLLM_ROOT:-${DEPENDENCY_ROOT}/third_party/vllm}"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${DEPENDENCY_ROOT}/third_party/vllm-ascend}"

export RECIPE_ROOT DEPENDENCY_ROOT VERL_ROOT MINDSPEED_ROOT MINDSPEED_LLM_ROOT MBRIDGE_ROOT MEGATRON_ROOT VLLM_ROOT VLLM_ASCEND_ROOT
export V41_CONFIG_SOURCE="${V41_CONFIG_SOURCE:-$VLLM_ASCEND_ROOT/vllm_ascend/deepseek_v41_config.py}"
export PYTHONPATH="$RECIPE_ROOT:${PYTHONPATH:-}"

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
# Log and error messages
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# INT8 QLI V3 selection, including cross-layer candidate blocks.
export V41_RMSNORM_BACKEND="${V41_RMSNORM_BACKEND:-native}"
export V41_ROPE_BACKWARD="${V41_ROPE_BACKWARD:-explicit}"
export V41_CSA2_METADATA_CACHE="${V41_CSA2_METADATA_CACHE:-1}"
export V41_ROPE_CACHE="${V41_ROPE_CACHE:-1}"
export V41_MHC_PRE_BACKEND="${V41_MHC_PRE_BACKEND:-premix}"
export V41_INDEXER_BACKEND="${V41_INDEXER_BACKEND:-lightning}"

# CSA2 tuning; explicitly propagated to Ray workers below.
export V41_CSA2_FORWARD_BACKEND="${V41_CSA2_FORWARD_BACKEND:-native}"
export V41_CSA2_BACKWARD_BACKEND="${V41_CSA2_BACKWARD_BACKEND:-native}"
export V41_CSA2_FORWARD_BK="${V41_CSA2_FORWARD_BK:-16}"
export V41_CSA2_BACKWARD_BK="${V41_CSA2_BACKWARD_BK:-16}"
export V41_CSA2_TASK_ORDER="${V41_CSA2_TASK_ORDER:-query}"
export V41_CSA2_SKIP_EMPTY="${V41_CSA2_SKIP_EMPTY:-2}"
# Record effective values, including explicit environment overrides.
printf 'V41 backends: mhc=%s indexer=%s csa2_forward=%s csa2_backward=%s rmsnorm=%s rope_backward=%s rope_cache=%s metadata_cache=%s\n' \
    "$V41_MHC_PRE_BACKEND" "$V41_INDEXER_BACKEND" "$V41_CSA2_FORWARD_BACKEND" \
    "$V41_CSA2_BACKWARD_BACKEND" "$V41_RMSNORM_BACKEND" "$V41_ROPE_BACKWARD" \
    "$V41_ROPE_CACHE" "$V41_CSA2_METADATA_CACHE"
# Ray memory and communication
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-250}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.95}"
export RAY_max_grpc_message_size=5368709120
# HuggingFace offline mode
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Pytorch NPU memory allocation and caching
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Vllm
export VLLM_USE_V1=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Vllm ascend
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_NZ=0
export TASK_QUEUE_ENABLE=0
export VLLM_ASCEND_TASK_QUEUE_ENABLE=0
# HCCL communication
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_BUFFSIZE=200
export GROUP_HCCL_BUFFER_SIZE=200
export P2P_HCCL_BUFFSIZE=0
# HCCL and Cuda parameters
export HCCL_OP_EXPANSION_MODE=AI_CPU
export NCCL_ALGO=Ring
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# CPU affinity and threading
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Unknown; used in dsv4_flash_verl.sh and dsv4_flash_mopd.sh.
export HYBRID_IGNORE_LOAD_CHECK=1
export USE_MAX_V2=0

# ------------------------- Paths -------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
export ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')}"
ROOT="${WORK_DIR:-${RECIPE_ROOT}}"
ENGRAM_HASH_SOURCE="${ENGRAM_HASH_SOURCE:-${MINDSPEED_LLM_ROOT}/mindspeed_llm/tasks/models/transformer/deepseek41/engram_hash.py}"
if [[ ! -f "$ENGRAM_HASH_SOURCE" ]]; then
    printf 'Missing Engram hash implementation: %s; use a MindSpeed-LLM checkout with V4.1 Engram support, or set ENGRAM_HASH_SOURCE.\n' "$ENGRAM_HASH_SOURCE" >&2
    exit 1
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# Preserve rollout loader paths before adding training-only native vendors.
# The image also prepends Premix globally; exclude that known training vendor.
v41_rollout_loader_path() {
    local entry result=""
    local -a entries
    [[ -n "$1" ]] || return 0
    IFS=: read -r -a entries <<< "$1"
    for entry in ${entries[@]+"${entries[@]}"}; do
        case "$entry" in
            ""|*/mhc-premix/*|/opt/dsv41-image/mhc-vendor|/opt/dsv41-image/mhc-vendor/*) continue ;;
        esac
        result="${result:+$result:}$entry"
    done
    printf '%s' "$result"
}
export V41_CSA2_ROLLOUT_LD_LIBRARY_PATH="$(v41_rollout_loader_path "${V41_CSA2_ROLLOUT_LD_LIBRARY_PATH-${LD_LIBRARY_PATH:-}}")"
export V41_CSA2_ROLLOUT_CUSTOM_OPP_PATH="$(v41_rollout_loader_path "${V41_CSA2_ROLLOUT_CUSTOM_OPP_PATH-${ASCEND_CUSTOM_OPP_PATH:-}}")"
export V41_CSA2_ISOLATE_ROLLOUT_ENV=1
unset -f v41_rollout_loader_path
# Prepare native QLI/CSA2 before Ray captures the driver environment.
if [[ "$V41_INDEXER_BACKEND" == lightning || "$V41_CSA2_FORWARD_BACKEND" == native || "$V41_CSA2_BACKWARD_BACKEND" == native ]]; then
    source "${V41_NATIVE_OPERATOR_ENV:-$RECIPE_ROOT/tools/native_operator_env.sh}"
fi
# The job submitter may replace image PYTHONPATH. Restore the additive vendor
# before Ray captures runtime_env; do not import/compile NPU operators here.
if [[ "$V41_MHC_PRE_BACKEND" == premix ]]; then
    mhc_build_root="${V41_MHC_BUILD_ROOT:-${DEPENDENCY_ROOT}/workspace/rollout/dsv41/build-src/mhc-premix}"
    if [[ ! -f "$mhc_build_root/env.sh" ]]; then
        printf 'Premix environment missing: %s/env.sh; use a Premix-enabled image or build it first.\n' "$mhc_build_root" >&2
        exit 1
    fi
    source "$mhc_build_root/env.sh"
    "$PYTHON_BIN" -c 'import importlib.util; assert importlib.util.find_spec("cann_ops_transformer_custom") is not None, "Premix Python package missing from training Python"'
fi
PROJECT_NAME="${PROJECT_NAME:-DeepSeek-V41-Flash}"
EXP_NAME="${EXP_NAME:-DeepSeek-V41-Flash-16-nodes}"
MODEL_PATH="${MODEL_PATH:-/path/to/deepseek-v4.1-flash-0910-BF16}"
LOG_DIR="${LOG_DIR:-/path/to/log}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/dsv41_flash_save_run_hpu}"
TRAIN_FILES="${TRAIN_FILES:-/path/to/rl_train.parquet}"
VAL_FILES="${VAL_FILES:-/path/to/rl_valid.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-/path/to/rl_reward.py}"
if [[ -z "${ENGRAM_CROP+x}" ]]; then
    export ENGRAM_CROP=""
    [[ ! -d "$MODEL_PATH/engram-grpo-crop" ]] || ENGRAM_CROP="$MODEL_PATH/engram-grpo-crop"
fi
export ENGRAM_TABLE_TRAINABLE="${ENGRAM_TABLE_TRAINABLE:-True}"
case "$ENGRAM_TABLE_TRAINABLE" in
    1|true|True) ENGRAM_TABLE_TRAINABLE=True ;;
    0|false|False) ENGRAM_TABLE_TRAINABLE=False ;;
    *) echo 'ENGRAM_TABLE_TRAINABLE must be 0/1 or True/False' >&2; exit 2 ;;
esac
if [[ -n "$ENGRAM_CROP" ]]; then
    export ENGRAM_SOURCE="${ENGRAM_SOURCE:-/path/to/deepseek-v4.1-flash-0910-BF16}"
else
    export ENGRAM_SOURCE="${ENGRAM_SOURCE:-$MODEL_PATH}"
fi
export V41_NATIVE_TRAINER=1
# Native training imports verl rollout classes too; resolve the base V4 packages.
# V41Replica separately prepends the pinned V4.1 sources for rollout servers.

MODEL_PATH="${MODEL_PATH%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
LOG_DIR="${LOG_DIR%/}"

# ------------------------- Cluster -------------------------
NNODES="${NNODES:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPUS_PER_NODE:-16}}"

# ------------------------- Data And Lengths -------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# ------------------------- Algorithm / Optimization ------------------
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
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0}"
SWAP_OPTIMIZER="${SWAP_OPTIMIZER:-True}"
SWAP_OPTIMIZER_TIMES="${SWAP_OPTIMIZER_TIMES:-64}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
BYPASS_RECOMPUTING_LOGPROBS="${BYPASS_RECOMPUTING_LOGPROBS:-False}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
INDEXER_LOSS_COEFF="${INDEXER_LOSS_COEFF:-0.0}"

# ------------------------- Parallelism -------------------------
ACTOR_TP=${ACTOR_TP:-8}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_CP=${ACTOR_CP:-16}
ACTOR_EP=${ACTOR_EP:-128}
ACTOR_ETP=${ACTOR_ETP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_DP=${ROLLOUT_DP:-16}
ROLLOUT_EP=${ROLLOUT_EP:-128}
ROLLOUT_DCP=${ROLLOUT_DCP:-1}
# Full Engram table owners: reuse EP; integers create row groups, 0 uses TP.
export ROLLOUT_ENGRAM_SHARDS="${ROLLOUT_ENGRAM_SHARDS:-ep}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_MODEL_LEN + ACTOR_CP - 1) / ACTOR_CP))}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_MODEL_LEN + ACTOR_CP - 1) / ACTOR_CP))}"

# ------------------------- Rollout / Trainer -------------------------
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-120}"
LOGGER="${LOGGER:-[\"console\", \"tensorboard\", \"wandb\"]}"
SEED="${SEED:-42}"


# ------------------------- Configuration -------------------------
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
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.tokenizer_path="$ENGRAM_SOURCE"
    actor_rollout_ref.model.external_lib=verl.utils.deepseek_v41_config
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
    ++algorithm.rollout_correction.bypass_mode=${BYPASS_RECOMPUTING_LOGPROBS}
    algorithm.rollout_correction.loss_type=ppo_clip
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.mindspeed.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.mindspeed.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.mindspeed.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.mindspeed.sequence_parallel=True
    actor_rollout_ref.actor.mindspeed.use_distributed_optimizer=True
    actor_rollout_ref.actor.mindspeed.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.mindspeed.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.mindspeed.optimizer_offload=False
    actor_rollout_ref.actor.mindspeed.use_dist_checkpointing=True
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    +actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=False
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]
    actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}"
    actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD}"
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD}
    ++actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER}
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    ++actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.free_cache_engine=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer=${SWAP_OPTIMIZER}
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer_times=${SWAP_OPTIMIZER_TIMES}
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_crop="$ENGRAM_CROP"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_table_trainable="$ENGRAM_TABLE_TRAINABLE"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_source="$ENGRAM_SOURCE"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v41_engram_hash_source="$ENGRAM_HASH_SOURCE"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.indexer_loss_coeff="${INDEXER_LOSS_COEFF}"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length="$MAX_MODEL_LEN"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.transformer_impl=local
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.normalization=RMSNorm
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_shared_storage=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_flash_attn=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_type=PretrainedFromHF
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_name_or_path="${MODEL_PATH}"
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size="${ROLLOUT_DCP}"
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=${N_SAMPLES}
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${TOP_P}"
    actor_rollout_ref.rollout.top_k="${TOP_K}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${WEIGHT_BUCKET_MB:-3072}"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.safetensors_load_strategy=lazy
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_engram=False
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0
)

TRAINER_CONFIG=(
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXP_NAME}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.device=npu
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.val_before_train=False
    trainer.test_freq="${TEST_FREQ}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.use_legacy_worker_impl=disable
)

# Optional native profiler: disabled unless explicitly requested.
if [[ -n "${V41_PROFILE_STEPS:-}" ]]; then
    TRAINER_CONFIG+=(
        global_profiler.tool=npu
        "global_profiler.steps=${V41_PROFILE_STEPS}"
        "global_profiler.save_path=${V41_PROFILE_PATH:-$OUTPUT_DIR/profile}"
        actor_rollout_ref.actor.profiler.enable=True
        # Rollout inherits actor profiling unless explicitly disabled.
        actor_rollout_ref.rollout.profiler.enable=False
        actor_rollout_ref.actor.profiler.all_ranks=False
        "actor_rollout_ref.actor.profiler.ranks=${V41_PROFILE_RANKS:-[0]}"
        'actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu]'
        actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
        actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True
        actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True
    )
fi

REWARD_CONFIG=(
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
)

EXTRA_CONFIG=(
    --config-name=ppo_trainer.yaml
    model_engine=mindspeed
    +seed="${SEED}"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE="'${HCCL_BUFFSIZE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE="'${GROUP_HCCL_BUFFER_SIZE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE="'${P2P_HCCL_BUFFSIZE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_FLASHCOMM1="'${VLLM_ASCEND_ENABLE_FLASHCOMM1}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ROLLOUT_ENGRAM_SHARDS="'${ROLLOUT_ENGRAM_SHARDS}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_NATIVE_TRAINER="'${V41_NATIVE_TRAINER}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ENGRAM_CROP="'${ENGRAM_CROP}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ENGRAM_SOURCE="'${ENGRAM_SOURCE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_RMSNORM_BACKEND="'${V41_RMSNORM_BACKEND}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_ROPE_BACKWARD="'${V41_ROPE_BACKWARD}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_METADATA_CACHE="'${V41_CSA2_METADATA_CACHE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_ROPE_CACHE="'${V41_ROPE_CACHE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_MHC_PRE_BACKEND="'${V41_MHC_PRE_BACKEND}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_INDEXER_BACKEND="'${V41_INDEXER_BACKEND}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_FORWARD_BACKEND="'${V41_CSA2_FORWARD_BACKEND}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_BACKWARD_BACKEND="'${V41_CSA2_BACKWARD_BACKEND}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_ISOLATE_ROLLOUT_ENV="'${V41_CSA2_ISOLATE_ROLLOUT_ENV:-0}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_ROLLOUT_LD_LIBRARY_PATH="'${V41_CSA2_ROLLOUT_LD_LIBRARY_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_ROLLOUT_CUSTOM_OPP_PATH="'${V41_CSA2_ROLLOUT_CUSTOM_OPP_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_NATIVE_LIBRARY="'${V41_CSA2_NATIVE_LIBRARY:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_CUSTOM_OPP_PATH="'${ASCEND_CUSTOM_OPP_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="'${LD_LIBRARY_PATH:-}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_FORWARD_BK="'${V41_CSA2_FORWARD_BK}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_BACKWARD_BK="'${V41_CSA2_BACKWARD_BK}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_TASK_ORDER="'${V41_CSA2_TASK_ORDER}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CSA2_SKIP_EMPTY="'${V41_CSA2_SKIP_EMPTY}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE="'${WANDB_MODE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'${PYTHONPATH}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.DEPENDENCY_ROOT="'${DEPENDENCY_ROOT}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.MINDSPEED_LLM_ROOT="'${MINDSPEED_LLM_ROOT}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.MBRIDGE_ROOT="'${MBRIDGE_ROOT}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROOT="'${VERL_ROOT}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ROOT="'${VLLM_ROOT}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ROOT="'${VLLM_ASCEND_ROOT}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.V41_CONFIG_SOURCE="'${V41_CONFIG_SOURCE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_VERBOSITY="'${TRANSFORMERS_VERBOSITY}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_NPU_SOCKET_PORT_RANGE="'${HCCL_NPU_SOCKET_PORT_RANGE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_HOST_SOCKET_PORT_RANGE="'${HCCL_HOST_SOCKET_PORT_RANGE}'"
)

# ------------------------- Launch -------------------------
mkdir -p "${LOG_DIR}"
START_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${START_TIME}_${EXP_NAME}_verl_grpo_${NNODES}N.log"

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

mkdir -p "$OUTPUT_DIR"

printf 'Using model path: %s\n' "${MODEL_PATH}"
printf 'Logging to %s\n' "${LOG_FILE}"
"${PYTHON_BIN}" -m verl.trainer.main_ppo "${ARGS[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
