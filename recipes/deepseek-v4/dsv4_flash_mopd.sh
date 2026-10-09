#!/usr/bin/env bash
# Full DeepSeek-V4-Flash MOPD training.
#
# Resources:
#   - student actor/ref + colocated rollout: 8 nodes x 16 NPUs = 128 NPUs
#   - one full-vocabulary teacher:           4 nodes x 16 NPUs =  64 NPUs
#
# The validated full-model RL script remains the single source of truth for
# actor/MindSpeed and rollout/vLLM configuration.  This wrapper pins the formal
# job profile and appends only the MOPD-specific overrides.

set -xeuo pipefail

if (( $# != 0 )); then
    printf 'This production script does not accept overrides; edit the script explicitly.\n' >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=python3

TRANSFER_QUEUE_VERSION="$(
    "${PYTHON_BIN}" -c \
        'from importlib.metadata import version; import transfer_queue; print(version("TransferQueue"))' \
        2>/dev/null || true
)"
if [[ "${TRANSFER_QUEUE_VERSION}" != "0.1.7" ]]; then
    echo "ERROR: full-vocabulary MOPD requires TransferQueue==0.1.7, found '${TRANSFER_QUEUE_VERSION:-not installed}'." >&2
    exit 1
fi

# Formal data/model/output paths.
export PROJECT_NAME=DeepSeek-V4-Flash-MOPD
export EXP_NAME=DeepSeek-V4-Flash-MOPD-8N-student-4N-teacher-full-vocab
export MODEL_PATH=/path/to/deepseek-v4-flash-0731-bf16
export TRAIN_FILES=/path/to/k26_rl_train.parquet
export VAL_FILES=/path/to/k26_rl_valid.parquet
export REWARD_FN_PATH=/path/to/k26_rl_reward.py
export OUTPUT_DIR=/path/to/dsv4_flash_save_run_hpu
export LOG_DIR=/path/to/log

# Student pool: exactly 8 nodes x 16 NPUs.  These are the known-good full RL
# settings; do not let inherited environment variables change the topology.
export NNODES=8
export GPUS_PER_NODE=16
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=64
export N_SAMPLES=8
export MAX_PROMPT_LENGTH=32768
export MAX_RESPONSE_LENGTH=1024
export MAX_MODEL_LEN=33792
export ACTOR_TP=4
export ACTOR_PP=2
export ACTOR_CP=8
export ACTOR_EP=64
export ACTOR_ETP=1
export ROLLOUT_TP=8
export ROLLOUT_DP=16
export ROLLOUT_EP=128
export ROLLOUT_DCP=1
export ACTOR_MAX_TOKEN_LEN_PER_GPU=4224
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4224

# Keep the optimizer/offload mode identical to the rerun that passed full RL.
export ALL_OFFLOAD=True
export OPTIMIZER_CPU_OFFLOAD=True
export USE_PRECISION_AWARE_OPTIMIZER=False
export OPTIMIZER_OFFLOAD_FRACTION=0
export SWAP_OPTIMIZER=True
export SWAP_OPTIMIZER_TIMES=64
export USE_DYNAMIC_BSZ=True
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MS_MAX_BATCH_SIZE=1
export MS_MANUAL_GC_INTERVAL=10
export MOE_SHARED_EXPERT_OVERLAP=False
export CONTEXT_PARALLEL_ALGO=kvallgather_cp_algo

# Known-good shared-card rollout behavior.  Level-1 sleep is selected in verl
# for the non-LoRA actor so weights are retained and only KV cache is released.
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
export ROLLOUT_MAX_NUM_SEQS=8
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_ASCEND_ENABLE_NZ=0
export TASK_QUEUE_ENABLE=0
export VLLM_ASCEND_TASK_QUEUE_ENABLE=0
export VLLM_USE_V1=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VERL_TEACHER_DEDICATED_RACK=1
export VERL_MOPD_DEBUG=0
export VERL_FULL_VOCAB_EXPERIMENT_NAME="${VERL_FULL_VOCAB_EXPERIMENT_NAME:-${EXP_NAME}}"

# Training settings are intentionally pinned for a reproducible formal job.
export ADV_ESTIMATOR=grpo
export USE_KL_IN_REWARD=False
export KL_COEF=0.0
export USE_KL_LOSS=False
export KL_LOSS_COEF=0.0
export BYPASS_RECOMPUTING_LOGPROBS=False
export LR=1e-6
export WEIGHT_DECAY=1e-2
export CLIP_GRAD=1.0
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export SAVE_FREQ=10
export TEST_FREQ=-1
export TOTAL_EPOCHS=1
export LOGGER='["console", "tensorboard", "wandb"]'
export SEED=42

# Pool arithmetic checked before the expensive Ray startup.
(( NNODES * GPUS_PER_NODE == 128 ))
(( ACTOR_TP * ACTOR_PP * ACTOR_CP == 64 ))
(( 128 % (ACTOR_TP * ACTOR_PP * ACTOR_CP) == 0 ))
(( ROLLOUT_TP * ROLLOUT_DP == 128 ))
(( ROLLOUT_EP == 128 ))

exec bash "${SCRIPT_DIR}/dsv4_flash_verl.sh" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    +actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=False \
    actor_rollout_ref.actor.mindspeed.use_dist_checkpointing=True \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
    'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    distillation.enabled=True \
    distillation.share_gpu_group=False \
    distillation.n_gpus_per_node=16 \
    distillation.nnodes=4 \
    distillation.teacher_key=data_source \
    distillation.distillation_loss.loss_mode=forward_kl_full_vocab \
    distillation.distillation_loss.full_vocab_chunk_tokens=128 \
    distillation.distillation_loss.full_vocab_lm_head_layer=head.weight \
    distillation.distillation_loss.full_vocab_experiment_name="${VERL_FULL_VOCAB_EXPERIMENT_NAME}" \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_task_rewards=True \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.loss_max_clamp=2.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    distillation.distillation_loss.distillation_loss_coef=0.5 \
    +distillation.teacher_models.gsm8k.key=openai/gsm8k \
    +distillation.teacher_models.gsm8k.model_path="${MODEL_PATH}" \
    +distillation.teacher_models.gsm8k.num_replicas=1 \
    +distillation.teacher_models.gsm8k.inference.name=vllm \
    +distillation.teacher_models.gsm8k.inference.data_parallel_size=8 \
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=8 \
    +distillation.teacher_models.gsm8k.inference.expert_parallel_size=64 \
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.75 \
    +distillation.teacher_models.gsm8k.inference.max_model_len=33800 \
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=33800 \
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=1 \
    +distillation.teacher_models.gsm8k.inference.enforce_eager=False \
    ++distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8]' \
    +distillation.teacher_models.gsm8k.inference.enable_chunked_prefill=False \
    ++distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.additional_config.enable_dsa_cp=True \
    +distillation.teacher_models.gsm8k.inference.enable_prefix_caching=False \
    +distillation.teacher_models.gsm8k.inference.free_cache_engine=False \
    +distillation.teacher_models.gsm8k.inference.enable_sleep_mode=True \
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.prefill_context_parallel_size=1 \
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.kv_cache_dtype=auto \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_TEACHER_DEDICATED_RACK='1'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MOPD_DEBUG='0'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FULL_VOCAB_EXPERIMENT_NAME='${VERL_FULL_VOCAB_EXPERIMENT_NAME}'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
