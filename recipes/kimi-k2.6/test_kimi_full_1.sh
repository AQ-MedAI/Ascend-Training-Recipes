#!/usr/bin/env bash
# Kimi-K2.6 GRPO RL with verl 0.8.0 + Megatron (hybrid engine)
#
# 传递链路(代码已核实):
#   rollout.engine_config        -> vLLM AsyncEngineArgs (online_vllm_backend.py, 改名7个+同名透传)
#   remote_megatron_config       -> Megatron args (megatron_config.py set_args 按名透传, bridge路从HF config推断结构)
#   allocation_mode              -> dp/tp/pp/cp/ep (alloc_mode.py)
#   gconfig                      -> 采样 (convert_sample_keys: max_new_tokens->max_tokens)
#   wrap_policy                  -> PPO (disable_value->GRPO, reward_norm group, recompute_logp->rollout logp)
#
# 硬性问题(非映射问题):
#   [B] rollout EP=128 与 verl "EP>1需EP=TP*DP(=40)" 冲突, 本脚本不设EP(默认1)。
#   [D] NPU 上 attention_backend 需按 vllm-ascend 实际支持调整。
# =====================================================

export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256"     # PYTORCH_NPU_ALLOC_CONF: "max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export HYBRID_IGNORE_LOAD_CHECK=1
export USE_MAX_V2=0
export HCCL_NPU_SOCKET_PORT_RANGE=62000-62050
export HCCL_HOST_SOCKET_PORT_RANGE=62100-62200
export NCCL_ALGO=Ring
# export OMP_NUM_THREADS=1   # [fix] vllm engine core 子进程 torch OpenMP 线程池崩溃(Invalid thread pool/堆损坏),锁 1 线程
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1


set -xeuo pipefail

########################### Quick Config ###########################
#   alloc_mode.py: d=dp p=pp t=tp c=cp e=ep
#   actor  : DP=1 TP=8 PP=10 CP=8 EP=64
#   rollout: DP=5 TP=8 PP=1 EP=128
TP=${TP:-8}
PP=${PP:-10}
CP=${CP:-8}
EP=${EP:-64}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}
ROLLOUT_DP=${ROLLOUT_DP:-16}
ROLLOUT_EP=${ROLLOUT_EP:-128}

rollout_name="vllm"
exp_name='kimi_k26'
adv_estimator=grpo

HF_MODEL_PATH=/path/to/Kimi-K2.6_bf16
TRAIN_PATH=/path/to/k26_rl_train.parquet
TEST_PATH=/path/to/k26_rl_valid.parquet
DEFAULT_LOCAL_DIR=/path/to/ckpt

N_SAMPLES=1
max_prompt_length=$((1024 * 64))
max_response_length=$((1024 * 8))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=$((actor_ppo_max_token_len / CP))
infer_ppo_max_token_len=$((infer_ppo_max_token_len / CP))
MAX_MODEL_LEN=$((max_prompt_length + max_response_length + 1))
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=1000000


#####TEACHER######
TEACHER_NNODES=16
NGPUS_PER_NODE=16
TEACHER_MODEL=/path/to/Kimi-K2.6_bf16
teacher_tp=8
teacher_dp=16
teacher_ep=128
teacher_gpu_mem_util=0.7
distillation_loss_mode=forward_kl_topk
distillation_topk=64
use_policy_gradient=False

########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${TRAIN_PATH}
    data.val_files=${TEST_PATH}
    data.train_batch_size=64
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=5.0e-7
    actor_rollout_ref.actor.optim.weight_decay=1.0e-2
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=adam
    actor_rollout_ref.actor.optim.betas=[0.9,0.95]
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.min_lr=0.0
    actor_rollout_ref.actor.optim.lr_warmup_steps=0.03
    # ---- PPO batch ----
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
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
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.multi_latent_attention=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_mask_type=causal
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${CP}
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_score_function=sigmoid
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_topk_scaling_factor=2.5
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=3
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=2
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    # ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    # +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    # +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    # +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    # +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=${N_SAMPLES}
    actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=1024
    actor_rollout_ref.rollout.scheduling_policy=fcfs
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TOP_P}
    actor_rollout_ref.rollout.top_k=${TOP_K}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cudagraph_capture_sizes=[1,4,8]
    actor_rollout_ref.rollout.calculate_log_probs=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_dtype=float32
    # +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN
    # +actor_rollout_ref.rollout.engine_kwargs.vllm.triton_attention_num_kv_splits=16
    # +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_overlap_schedule=False
)

REF=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=True
)


ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.gamma=1.0
    algorithm.lam=1.0
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=16
    trainer.nnodes=40
    trainer.save_freq=10
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.total_epochs=1
    trainer.total_training_steps=1145
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}
)

REWARD=(
    reward.custom_reward_function.path=/path/to/k26_rl_reward.py   # orz math reward: 提 <answer>/\boxed{} 对 reward_model.ground_truth
    reward.custom_reward_function.name=compute_score
)

EXTRA=(
    model_engine=megatron
    +seed=42
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE='200'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE='200'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE='AIV'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RPC_TIMEOUT='1800000'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_TEACHER_DEDICATED_RACK='1'"
)


TEACHER=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${NGPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.data_parallel_size=${teacher_dp}
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.expert_parallel_size=${teacher_ep}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=1024
    distillation.teacher_models.teacher_model.inference.max_num_seqs=32
    distillation.teacher_models.teacher_model.inference.enforce_eager=False
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=False
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.cudagraph_capture_sizes=[1,4,8]
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.kv_cache_dtype=auto
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.mamba_cache_dtype=float32
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)
########################### Launch ###########################
start_time=$(date +%Y%m%d)_$(date +%H%M%S)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "${TEACHER[@]}" \
    "$@" 2>&1 | tee /path/to/logs${start_time}_kimi_k26_verl_grpo_40N.log
