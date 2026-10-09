#!/usr/bin/env bash
# On-policy distillation | MULTI-teacher (gsm8k + math) | FULL-VOCAB KL | vLLM rollout | Megatron | Ascend NPU
# 全层 Kimi-K2.6 学生 + 2 全层 Kimi-K2.6 教师 (按 data_source 路由), forward_kl_full_vocab loss.
#
# 合并自两份已跑通脚本:
#   - test_kimi_full_1.sh                          (全层 OPD 单教师 topk, actor TP8PP10CP8EP64=640die/40机 已跑通)
#   - run_k26_4layers_mopd_2teacher_4node_fullvocab.sh (减层 MOPD 双教师 fullvocab, fullvocab 处理全套)
# 改动要点:
#   1. actor 并行保留 OPD (TP8 PP10 CP8 EP64, 40机×16die=640die)
#   2. teacher 切分保留 OPD (tp8/dp16/ep128, per_replica=128die), 双教师 num_replicas=1 each
#      → teacher pool = 128+128 = 256die = 16机×16die (跟 OPD 单教师总die持平, 拆2教师)
#      → 改 num_replicas=2 则 teacher 翻倍 512die=32机, 总集群 72机
#   3. loss 换 forward_kl_full_vocab + full_vocab_chunk_tokens=2048 (减层 fullvocab 验证值)
#   4. teacher 强制 fullvocab 约束 (代码 distillation.py:221-248 强制, 否则 __post_init__ raise):
#        max_num_seqs=1 + enable_chunked_prefill=false + max_num_batched_tokens>=max_model_len
#   5. 模型换全层 Kimi-K2.6 (学生+2教师同模型同vocab, 满足 fullvocab teacher_vocab<=student_vocab 且被 student TP 整除)
#   6. VERL_ROOT 用 full-vocab fork (forward_kl_full_vocab loss 只在此 fork: verl_full_vocab_0806_no_reverse_kl)
#   7. 数据 gsm8k+math 双源 (按 data_source 路由 2 教师, key 匹配 parquet 实际 data_source 值)
# 集群总: actor 40机 + teacher 16机 = 56机. Layout disjoint (main 40机 + teacher 16机 物理隔离).
#
# PRE-REQ (脚本管不了, 必须先做):
#   1. ray 集群已起: head `ray start --head --port=6379`; 其余机 `ray start --address=<head>:6379`;
#      `ray status` 看到 56 节点 x 16 die (40 main + 16 teacher).
#   2. 56 机共享存储同路径挂载 (模型/数据/ckpt + VERL_ROOT 源码 + HF_HOME 缓存, 56 机都能 ls 到同一份).
#      verl 未 pip install, worker 靠透传 PYTHONPATH=${VERL_ROOT} import 源码, 路径必须共享可见.
#   3. 56 机 CANN/torch_npu/vllm-ascend 版本一致; 防火墙放行 ray 端口 + 下方两个 HCCL 端口段.
#   4. HF_HOME 共享缓存已建好 (含 KimiK25 custom code 全部 .py, 含 modeling_*):
#      /path/to/<model> (ray worker import 链不断)
#   5. 在 head 上执行本脚本. 拉起后 npu-smi info 确认 teacher 256die 与 main 640die 不重叠.
set -xeuo pipefail

# [fullvocab] HF trust_remote_code 缓存: KimiK25 custom code (modeling/configuration_kimi_k25) 需复制到
# ~/.cache/huggingface/modules/transformers_modules/<model>/ 才能被 ray worker import. 新环境首次跑 HF 自动
# 复制时漏 modeling_deepseek.py/modeling_kimi_k25.py -> 缓存不全 + worker 节点缓存空 -> ray 反序列化时
# ModuleNotFoundError configuration_kimi_k25. 解法: 完整 custom code 预置到共享存储, 设 HF_HOME 指向它.
#export HF_HOME=${HF_HOME:-/path/to/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60200
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61200
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=0
# [OOM 权衡] 256 时 HCCL 能拿 1.56GB 分块, step4 PyTorch optimizer alloc 碎片化崩; 128 让 PyTorch 切碎 NPU
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
# [HCCL OOM] 容器默认 HCCL_BUFFSIZE=2048(2GiB)->4GiB double-buffer 会 OOM. 多机跨机通信更易触发, 必须设小.
export HCCL_BUFFSIZE=512
export GROUP_HCCL_BUFFER_SIZE=256
export P2P_HCCL_BUFFSIZE=0
# [HCCL 通信模式] 默认 FFTS+ 遮蔽真实错误 (只报 HcclAllreduce RuntimeError, 栈不准). AIV 暴露 dataType 不一致
# 等细节 (EI0005). AIV 是诊断工具也是通信模式选择. 减层 fullvocab 验证值.
export HCCL_OP_EXPANSION_MODE=AIV
# vllm engine core 子进程 torch OpenMP 线程池崩溃 (Invalid thread pool/堆损坏), 锁线程.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# All 16 dies (0..15) healthy. 若有坏 die, 去掉其 id 并降 pool 使 NGPUS_PER_NODE + TEACHER_GPUS_PER_NODE <= len.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# full-vocab fork (forward_kl_full_vocab loss 只在此 fork). 减层 fullvocab 同 VERL_ROOT.
VERL_ROOT=${VERL_ROOT:-/path/to/verl}
cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

# ---- user-adjustable ----
# 全层 Kimi-K2.6 (学生 + 2 教师同模型, fullvocab 要求同 tokenizer/vocab). OPD test_kimi_full_1.sh 同路径.
HF_MODEL_PATH=/path/to/Kimi-K2.6_bf16
STUDENT_MODEL=${HF_MODEL_PATH}
GSM8K_TEACHER_MODEL=${HF_MODEL_PATH}
MATH_TEACHER_MODEL=${HF_MODEL_PATH}

# ---- actor 并行 (保留 OPD test_kimi_full_1.sh): TP8 PP10 CP8 EP64 = 640die = 40机×16die ----
# TP8: 砍 fullvocab lm_head/dense grad 半 (TP 分片), 治 fullvocab actor OOM 根因.
# PP10: K2.6 全层 61 层, 10 stage (first=3 last=2 中间8×7), OPD 验证 pipeline 稳.
# CP8: 切序列, ppo_max_token_len=(2k+2k)/8=512/卡.
# EP64: 全层 384 experts 切 64 组, 砍 npu_grouped_matmul alloc 治碎片化 (vs 减层 EP16, 全层 die 多空间宽裕).
NNODES=${NNODES:-40}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
TP=${TP:-8}
PP=${PP:-10}
CP=${CP:-8}
EP=${EP:-64}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}            # rollout vllm tensor_parallel_size
ROLLOUT_DP=${ROLLOUT_DP:-16}   # rollout vllm data_parallel (OPD 验证值; 40机大集群 16路并发)
ROLLOUT_EP=${ROLLOUT_EP:-128}  # rollout vllm expert_parallel (解耦 actor EP, vllm 内部切 expert)
OFFLOAD=${OFFLOAD:-True}       # megatron param/optimizer offload

# ---- teacher 切分 (保留 OPD): tp8/dp16/ep128, per_replica=128die ----
# 双教师 num_replicas=1 each → pool = (1+1)×8×16 = 256die = 16机×16die
#   (== distillation.nnodes×n_gpus_per_node 严格校验, verl distillation.py:344-350)
# fullvocab teacher max_num_seqs=1 单请求 forward, 吞吐靠 dp16 (16路 vllM 并发/教师) 不靠 replica, 1 replica 够.
# 改 num_replicas=2 则 teacher 翻倍 512die=32机, 总集群 72机 (actor40+teacher32).
TEACHER_NNODES=${TEACHER_NNODES:-16}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-16}
TEACHER_TP=${TEACHER_TP:-8}
TEACHER_DP=${TEACHER_DP:-2}
TEACHER_PCP=${TEACHER_PCP:-8}
TEACHER_EP=${TEACHER_EP:-128}
TEACHER_NUM_REPLICAS_GSM8K=${TEACHER_NUM_REPLICAS_GSM8K:-1}
TEACHER_NUM_REPLICAS_MATH=${TEACHER_NUM_REPLICAS_MATH:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.7}


# ---- loss: fullvocab (减层 fullvocab 验证值) ----
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-reverse_kl_full_vocab}
full_vocab_chunk_tokens=${FULL_VOCAB_CHUNK_TOKENS:-4096}   # fullvocab readme 默认值 (一次 chunk 处理多 token). OOM 再砍 2048.
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}   # fullvocab 不用 topk (use_topk=False), 字段保留无害

# ---- batch + token (OPD 64k/8k) ----
#   - teacher_tp8 (vocab 152k/tp8=19k 行, V/tp shard 小) + full_vocab_chunk_tokens 4096 分 chunk 循环
#   - teacher max_num_seqs=1 单请求 (fullvocab 强制), 无并发, 显存压力=单条 72k
# 全层 K2.6 lm_head 大 (vocab≈152k), full_vocab_chunk_tokens 用 4096 (fullvocab readme 默认值, 一次 chunk 处理多 token)
# 若 actor step4 fullvocab 碎片化 OOM 再砍 4096->2048 (减层验证值).
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 64))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
N_SAMPLES=${N_SAMPLES:-1}   # MOPD GRPO 需 n>=2 (减层 4; OPD 单教师 n=1 无 advantage). batch64×n4=256 traj/step
actor_ppo_max_token_len=$(( max_prompt_length + max_response_length ))
infer_ppo_max_token_len=$(( max_prompt_length + max_response_length ))
actor_ppo_max_token_len=$(( actor_ppo_max_token_len / CP ))
infer_ppo_max_token_len=$(( infer_ppo_max_token_len / CP ))
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$actor_ppo_max_token_len}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-2048}   # student rollout prefill 上限 (baseline, 学生无 fullvocab 约束)

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-1000000}
actor_lr=${ACTOR_LR:-5.0e-7}   # OPD 全层验证值 (大模型小 lr; 减层用 1e-6)

rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}   # 减层 fullvocab 验证值
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-$max_num_tokens}   # 4097, fullvocab 要求 max_num_batched_tokens>=此值

total_epochs=${TOTAL_EPOCHS:-1}   # smoke
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-50}
project_name=${PROJECT_NAME:-verl_distill_mopd_full}
experiment_name=${EXPERIMENT_NAME:-k26_full_mopd_2teacher_fullvocab}
reward_fn_path=${REWARD_FN_PATH:-/path/to/orz_math_service_reward.py}

# 数据 (减层 fullvocab 双源, 按 data_source 路由 2 教师)
# Dataset A: gsm8k  (data_source = "openai/gsm8k") -> teacher gsm8k
# Dataset B: math   (data_source = "math", orz_math_train.parquet 实测) -> teacher math
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
    # use_task_rewards=True 时 use_kl_in_reward 生效, 给 task reward 加 -kl_coef*KL(pi||ref) 防 RL drift (减层 fullvocab 保留)
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
    # [smoke] 截断样本数加速数据加载. 注意 gsm8k(64k) 在前 math 在后, train_max_samples=5000 大概率全 gsm8k.
    # 跑全量 export TRAIN_MAX_SAMPLES=-1.
    data.train_max_samples=${TRAIN_MAX_SAMPLES:-5000}
    data.val_max_samples=${VAL_MAX_SAMPLES:-1000}
    # KimiK25 含 custom code (modeling_kimi_k25/configuration_kimi_k25, config.json auto_map), 需 trust_remote_code.
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    # [megatron] 不用 FSDP 的 enable_gradient_checkpointing (KimiK25 不支持). megatron 用 recompute_method=uniform.
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR=(
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.use_torch_compile=False
    # ---- optim (OPD 全层验证值) ----
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=1.0e-2
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=adam
    actor_rollout_ref.actor.optim.betas=[0.9,0.95]
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.min_lr=0.0
    actor_rollout_ref.actor.optim.lr_warmup_steps=0.03
    # [MindSpeed AdamW OOM] 减层 fullvocab 验证: 大 batch 触发 NPU 建 optimizer state (exp_avg/exp_avg_sq) OOM,
    # 用 override_optimizer_config 3 件套强制 state 走 CPU. 不开 use_precision_aware_optimizer
    # (会引入 grad dataType 不一致 -> HCCL AllReduce EI0005: local 4 fp32 vs remote 11 bf16).
    # OPD 全层 topk 单教师 n=1 没开也能跑; fullvocab 双教师 n=4 batch 大 (256 traj) 开 3 件套保险
    # (CPU offload 不影响正确性, 只多 CPU 内存; 不开 use_precision_aware 则不回 OOM 也不回 AllReduce).
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
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
    # ---- KL loss: 不用 KL loss (已有 distill loss + use_kl_in_reward) ----
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    # ---- megatron-bridge 并行 (OPD TP8PP10CP8EP64) ----
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
    actor_rollout_ref.actor.megatron.grad_offload=True   # OPD 全层验证值
    # ---- override_transformer_config (OPD 全层 + 减层 sequence_parallel/moe_router_dtype) ----
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
    # OPD 全层 PP10 不均 (K2.6 61层/10stage): first=3 last=2, 中间 8stage×7层 = 3+56+2=61
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=3
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=2
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
    # [AllReduce fix] MoE router 默认 bf16 与其他参数 fp32 grad 在 get_grad_norm AllReduce 时 dataType 不一致.
    # 官方 grpo_megatron MoE 示例配套设 moe_router_dtype=fp32.
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    # n=4: GRPO 需 >1 samples/prompt for group-relative advantage. batch64×n4=256 rollout.
    actor_rollout_ref.rollout.n=${N_SAMPLES}
    # student rollout 无 fullvocab 约束 (hidden 导出是 teacher 的事, 学生只管生成). 配 32 baseline 加速.
    actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.scheduling_policy=fcfs
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TOP_P}
    actor_rollout_ref.rollout.top_k=${TOP_K}
    # student rollout 无 fullvocab 约束, 保持 baseline (图模式 + cudagraph). 沿用 OPD/减层.
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cudagraph_capture_sizes=[1,4,8]
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # NPU vllm-ascend engine_kwargs (减层验证值)
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

# [megatron] 构建 CANN env 透传数组 (head 上动态读当前 env 值, 不硬编码路径).
# PYTHONPATH 特殊: verl 源码前缀 + 保留 head 原 PYTHONPATH (含 CANN python/site-packages/mindspeed 等).
# ray worker 是子进程不继承 head env, megatron/mindspeed 要 import acl (CANN python/site-packages) + 找 libascendcl.so.
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
# Each teacher key MUST equal the data_source value in its parquet (实测 pandas.read_parquet 确认).
# 别用 teacher_model 当 key (单数会被 _resolve_teacher_models pop + 自动算 num_replicas, 多教师场景必须显式两 key).
EXTRA=(
    # [megatron] 切后端: FSDP -> Megatron (KimiK25 VLM 外壳改类名, FSDP wrap policy 找不到 DeepseekDecoderLayer -> OOM)
    model_engine=megatron
    +seed=42
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    # --- teacher A: gsm8k (全层 K2.6, key 匹配 parquet data_source="openai/gsm8k") ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="$GSM8K_TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas=${TEACHER_NUM_REPLICAS_GSM8K}
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${TEACHER_TP}
    +distillation.teacher_models.gsm8k.inference.data_parallel_size=${TEACHER_DP}
    +distillation.teacher_models.gsm8k.inference.expert_parallel_size=${TEACHER_EP}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    +distillation.teacher_models.gsm8k.inference.enforce_eager=False                              # teacher yaml 默认 True, fullvocab 单 forward 用不着 cudagraph
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.cudagraph_capture_sizes=[1]
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.prefill_context_parallel_size=${TEACHER_PCP}
    +distillation.teacher_models.gsm8k.inference.enable_chunked_prefill=False                    # fullvocab 强制 (distillation.py:230)
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=1                                   # fullvocab 强制 (distillation.py:224): 单请求/forward 保 hidden 对齐
    +distillation.teacher_models.gsm8k.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=${teacher_max_model_len} # >= max_model_len, fullvocab 强制 (distillation.py:236): prefill 单 forward
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.kv_cache_dtype=auto
    +distillation.teacher_models.gsm8k.inference.engine_kwargs.vllm.mamba_cache_dtype=float32
    # --- teacher B: math (全层 K2.6, key 匹配 orz_math parquet data_source="math") ---
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
    # fullvocab: teacher 导出全词表 hidden + lm_head, 对全 V 做 forward KL (vs topk 只对 top-k).
    # full_vocab_chunk_tokens=2048: chunk 循环砍 lm_head [chunk, V/tp] fp32 buffer, 治 step4 碎片化 OOM.
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.full_vocab_chunk_tokens=${full_vocab_chunk_tokens}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=2.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.distillation_loss_coef=0.5
    # [multi-node] NPU env 透传到所有 ray worker (否则 worker 拿不到 HCCL_BUFFSIZE 等 -> 用容器默认 2048 -> 跨机 OOM).
    # head 的 export 只对 driver 生效. 数值须与上方 export 段一致.
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:256'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE='200'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE='200'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_TEACHER_DEDICATED_RACK='1'"
    # 透传 CANN 全套 env 给 ray worker (上方 CANN_ENV_PROPAGATE 数组动态读 head 当前值).
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
