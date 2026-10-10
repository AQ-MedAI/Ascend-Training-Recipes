#!/usr/bin/env bash
# On-policy distillation | MULTI-teacher (gsm8k + mbpp) | vLLM rollout | Megatron | Ascend NPU
# 8-NODE env (8 boxes x 16 die = 128 die; 1 卡=2 die, 8 卡/机), 但 main 只占 4 台机满 (4x16=64 die).
# Megatron backend. 治 4 机版 (32 die) fullvocab+use_kl_in_reward 的 OOM.
# Layout (disjoint, 物理隔离):
#   - Main pool (actor+rollout+ref): NNODES=4, NGPUS_PER_NODE=16 -> 64 die (4 台机满, 16 die/机)
#     megatron: TP8 x PP4 x CP2 x EP16 = 64 die
#       TP8 (vs 4机版TP4) 砍 fullvocab lm_head/dense grad 半, 治 actor PyTorch OOM
#       EP16 砍 npu_grouped_matmul alloc 2.63->1.31, 治 actor 碎片化 OOM (8x8 布局时)
#       EP16 机内 (16 die/机=1 机内) 消除跨机 HCCL ReduceScatter buffer (1.67GB) OOM
#   - Teacher pool: TEACHER_NNODES=1, 2 teacher x teacher_tp=4 -> 8 die, 单独 1 台机 (物理隔离)
#   - 8 机环境里占 5 台 (4 main + 1 teacher), 剩 3 台空闲.
#   - GRPO: train_batch_size*n = 64*4 = 256, world_size=64 -> 256/64=4 OK.
# PRE-REQ (脚本管不了, 必须先做):
#   1. ray 集群已起: head `ray start --head --port=6379`; 其余机 `ray start --address=<head>:6379`;
#      `ray status` 看到 4 节点 x 16 die.
#   2. 8 机共享存储同路径挂载 (模型/数据/ckpt 目录 + VERL_ROOT 源码目录, 8 机都能 ls 到同一份).
#      verl 未 pip install, worker 靠透传的 PYTHONPATH=${VERL_ROOT} import 源码, 路径必须共享可见.
#   3. 8 机 CANN/torch_npu/vllm-ascend/verl 版本一致; 防火墙放行 ray 端口 + 下方两个 HCCL 端口段.
#   4. 在 head 上执行本脚本. 拉起后 npu-smi info 确认 teacher 8 die 与 main 64 die 不重叠.
# Routed by `data_source`. MOPD with task rewards (distillation + RL on correctness).
set -xeuo pipefail

# ---- NPU environment ----
# [4node] 端口段从 50 扩到 200 (4 机 40+ rank 够用); 修原 HCCL_NPU_SOCKET_PORT_RANGE=61000-60050 的 typo.
# [fullvocab][新环境] HF trust_remote_code 缓存: KimiK25 custom code (modeling/configuration_kimi_k25)
# 需复制到 ~/.cache/huggingface/modules/transformers_modules/<model>/ 才能被 ray worker import. 新环境首次跑
# HF 自动复制时漏了 modeling_deepseek.py/modeling_kimi_k25.py (import torch_npu 等依赖时降级跳过) -> 缓存
# 不全 + worker 节点缓存空 -> ray 反序列化 CheckpointEngineWorker 时 ModuleNotFoundError configuration_kimi_k25
# (导入链断在 modeling). 解法: 把完整 custom code 预置到共享存储, 设 HF_HOME 指向它, 4 机 worker 用同一份.
# 共享缓存已在远程建好 (含全部 10 个 .py, 含 modeling_*): /path/to/...
export HF_HOME=${HF_HOME:-/path/to/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60200
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61200
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
# [4node] HCCL buffer: 容器默认 HCCL_BUFFSIZE=2048(2GiB)->4GiB double-buffer 会 OOM (见 test_kimi.sh 注释).
# 多机跨机通信更易触发, 必须设小. 对齐 test_kimi_2node.sh / test_kimi.sh 的验证值.
export HCCL_BUFFSIZE=512   # [HCCL OOM] 800->512: HCCL AllReduce 1.56GB buffer OOM (EL0004, 独立显存不归PyTorch).
# 赌 HCCL 按 BUFFSIZE 分块 (512MB一块, 1.56GB分4块). 4x16+EP16 已过 PyTorch OOM, 这是 HCCL 显存余量问题.
# 风险: 不确定 HCCL 是否分块, 可能无效. 若无效退回 800 走别的解法.
export GROUP_HCCL_BUFFER_SIZE=256   # [HCCL OOM] 400->256 配套降.
export P2P_HCCL_BUFFSIZE=0
# [4node] HCCL 通信模式: 日志提示默认 FFTS+ 限制可用流数+支持的 shape 范围, 建议 AIV 改善通信性能+稳定性.
# grad_norm 的跨 4 机 AllReduce 失败 (无 rank 先死/无 NaN), 疑通信模式不稳, 设 AIV 试.
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# All 16 dies (0..15) healthy. If a die goes bad, drop its id AND lower the pools
# so NGPUS_PER_NODE + TEACHER_WORLD_SIZE <= len(this list).
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

VERL_ROOT=${VERL_ROOT:-/path/to/verl} #/path/to/verl_opd for fullvocab, /path/to/verl_opd
cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

# ---- user-adjustable ----
HF_MODEL_PATH=/path/to/kimi_k26_bf16_4layers_wlp
STUDENT_MODEL=${HF_MODEL_PATH}
# Two 27B domain-expert teachers. MUST share the student's tokenizer/vocab
# (verify Qwen3.5-27B vs Qwen3-8B tokenizer_config.json match, else logprobs misalign).
GSM8K_TEACHER_MODEL=${HF_MODEL_PATH}
ORZ_TEACHER_MODEL=${HF_MODEL_PATH}

NNODES=${NNODES:-4}   # [HCCL OOM] 8->4 机. main pool 64 die 占 4 台机满 (16 die/机). EP16=16 die 全在 1 机内
                     # → ReduceScatter 机内 buffer 小, 消除 EP16 跨 2 机的大 HCCL buffer (1.67GB) OOM.
                     # 8x8 布局时 EP16 跨 2 机, HCCL ReduceScatter alloc 1.67GB 通信 buffer 失败 (EL0004).
                     # 改 4x16: main 仍 64 die (TP8×PP4×CP2×EP16), 但每台 16 die 给 main, EP16 组=16 die=1 机内.
                     # TP8=8 die 仍 1 机内 ✓. teacher 单独 1 台机物理隔离. 剩余 3 台机空闲 (8 机环境里只占 5 台).
# Main pool (student). 16 die/机 × 4 机 = 64 die = TP8×PP4×CP2×EP16. EP16 机内是关键 (消除跨机 HCCL buffer).
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}

# Per-teacher replicas; teacher pool = sum(num_replicas) * teacher_tp
# 27B needs teacher_tp=4 (13.5GB weights/die, comfortable KV; also forces disjoint at 8+8=16).
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_NUM_REPLICAS_GSM8K=${TEACHER_NUM_REPLICAS_GSM8K:-1}
TEACHER_NUM_REPLICAS_MBPP=${TEACHER_NUM_REPLICAS_MBPP:-1}
teacher_tp=${TEACHER_TP:-4}
TEACHER_WORLD_SIZE=$(( (TEACHER_NUM_REPLICAS_GSM8K + TEACHER_NUM_REPLICAS_MBPP) * teacher_tp ))

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-forward_kl_full_vocab}
full_vocab_chunk_tokens=${FULL_VOCAB_CHUNK_TOKENS:-2048}   # [step4 OOM] 4096->2048: step4 optimizer 阶段 _copy_model_grads_to_main_grads alloc 114MB 失败 (碎片化). fullvocab chunk buffer [chunk, V/tp] fp32 砍半, PyTorch allocator 占用降, 给 optimizer main_grad + HCCL 连续块腾空间. 偏离 readme (写4096) 但 OOM 破例. 代价: chunk 循环多一倍 (kernel launch 多), 略慢.
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# Real training: larger batch + real sequence lengths.
# Length sizing standard:
#   - max_response_length caps the STUDENT rollout (not the teacher). Base it on the
#     teachers' ACTUAL response distribution, not their training cap.
#     mbpp teacher converged to resp mean ~800, max mostly <2k (cap was 11776, unused);
#     gsm8k teacher <2k. So 2048 covers mean x2.5 + student (8B) margin.
#   - max_prompt=2048 matches the gsm8k teacher's training prompt cap (actual prompts
#     are ~70-200 tokens, so this is pure safety margin).
#   - Teacher max_model_len auto = max_num_tokens (4097), well within 27B native 32k+.
#   - After launch, watch response_length/clip_ratio; if >0.2, raise max_response.
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
# ---- megatron parallelism ----
# [8node] TP8 x PP4 x CP2 x EP8 = 64 die (4 台 × 16 die). 8 机 128 die 空间宽裕, 不必为省显存上 EP16.
# TP8 (vs 4机版 TP4) 砍 fullvocab lm_head 587->294MB + dense grad 半 (TP 分片), 治 fullvocab OOM 根因.
# CP2 不变 (切序列, ppo_max_token_len=2048).
# [8node OOM 碎片化] EP8->16: 8机 actor 45.66GB, free 2.67 > alloc 2.63 但 OOM -> 纯碎片化 (0.64GB pooled
# 碎片挡连续 alloc, expandable_segments 治不了因 vLLM sleep mode CaMemAllocator 硬冲突 expandable).
# EP16 砍 npu_grouped_matmul alloc 2.63->1.31 (按 expert 数缩, 4机验证过), 1.31 < free 即使碎片也放得下.
# EP16=16 die/stage, PP4 时每 stage 16 die (64/4) ✓ 放得下. 代价: MoE AlltoAll 跨 2 机 (EP8 机内->EP16 跨机),
# 但 AlltoAll 数据量小. rollout EP 解耦保 8 (worker 8 die 做不了 EP16), weight sync actor EP16->rollout EP8
# 重分片 (8 整除 16 合法, 但 verl 未验证 reshard, 是本次主要风险点).
# TP8=8 die=1 机内 ✓; CP2 跨机小数据 (hidden×2048); PP4 跨 4 机.
TP=${TP:-8}
PP=${PP:-4}
CP=${CP:-2}
EP=${EP:-16}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}          # rollout vllm tensor_parallel_size (=rollout_tp)
ROLLOUT_EP=${ROLLOUT_EP:-8}  # rollout vllm expert_parallel_size. 解耦 actor EP: rollout worker 仅 8 die
                              # (GEN_TP=8), 做 EP16 装不下; actor EP=16 (砍 alloc 治碎片) 但 rollout 保 EP8,
                              # weight sync 时 verl 重分片 (actor 16 片 -> rollout 8 片, 8 整除 16 合法).
ROLLOUT_DP=${ROLLOUT_DP:-1}  # rollout data_parallel. >1 有 zmq bug (见 test_kimi_2node 注释), 保持 1
OFFLOAD=${OFFLOAD:-True}     # megatron param/optimizer offload

# token 预算: megatron dynamic_bsz 按 (prompt+resp)/CP 切 (CP 把一条样本分到多卡, 单卡预算可缩).
# 对齐 test_kimi_2node.sh:71-74. max_prompt=2048+max_resp=2048=4096, /CP=2048.
actor_ppo_max_token_len=$(( max_prompt_length + max_response_length ))
infer_ppo_max_token_len=$(( max_prompt_length + max_response_length ))
actor_ppo_max_token_len=$(( actor_ppo_max_token_len / CP ))
infer_ppo_max_token_len=$(( infer_ppo_max_token_len / CP ))
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$actor_ppo_max_token_len}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-2048}

# 采样参数 (rollout vllm 用)
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-1000000}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-8}
# rollout vLLM 与 actor 同 die 共置. util 调过 0.25/0.35 都没解决 actor bwd OOM (日志: non-PyTorch
# retain ~12GB 不随 util 变, KV pool 不是大头), 说明 OOM 根因是 fullvocab 在 actor 侧多 retain ~1.5GB
# 时 KV pool 5.64GB, 4层MLA KV 极小, 够用). OOM 改由学生 max_num_batched_tokens 2048->1024 压 prefill 激活 pool 治.
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.5}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-100}   # 10->5: capture the peak ckpt (reward peaks ~step 27 then crashes)
# Real steps are long (longer seqs + 27B teachers); validate less often to avoid long stalls.
test_freq=${TEST_FREQ:-50}

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
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-$max_num_tokens}
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    # KL anchor to ref (use_task_rewards=True 时 use_kl_in_reward 生效, 给 task reward 加
    # -kl_coef*KL(pi||ref) 防 RL drift). 保留 (用户要求). OOM 不靠关它, 改由学生 rollout
    # max_num_batched_tokens 2048->1024 压 vLLM prefill 激活 pool 解决 (见 ROLLOUT 块).
    algorithm.use_kl_in_reward=True
    algorithm.kl_penalty=kl
    +algorithm.kl_coef=0.001
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    # filter_overlong_prompts 默认 num_proc=1, 64351 条要 1 分钟. 多进程过滤秒过.
    data.filter_overlong_prompts_workers=${FILTER_WORKERS:-8}
    data.truncation='error'
    data.shuffle=True
    # [smoke] 截断样本数加速数据加载. 默认 -1 全量; 设正数则合并后两 parquet 取前 N.
    # 注意: gsm8k(64351) 在前 mbpp 在后, train_max_samples=10000 大概率全是 gsm8k.
    # 跑全量时 export TRAIN_MAX_SAMPLES=-1 即可恢复.
    data.train_max_samples=${TRAIN_MAX_SAMPLES:-5000}
    data.val_max_samples=${VAL_MAX_SAMPLES:-1000}
    # KimiK25 含 custom code (modeling_kimi_k25 / configuration_kimi_k25, config.json auto_map).
    # main_ppo.py:270 读 data.trust_remote_code (默认 False) -> hf_tokenizer -> resolve_trust_remote_code
    # 不设会抛 ValueError "contains custom code ... pass trust_remote_code=True". 对齐 test_kimi.sh:91.
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    # [megatron] 不用 FSDP 的 enable_gradient_checkpointing (KimiK25 不支持).
    # megatron 用 override_transformer_config.recompute_method=uniform 代替 (见 ACTOR).
    # KimiK25 custom code: actor/ref/rollout 加载权重时需 trust_remote_code. 对齐 test_kimi.sh:98.
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR=(
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=1.0e-2
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=adam
    actor_rollout_ref.actor.optim.betas=[0.9,0.999]
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.min_lr=1.0e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    # [OOM fix] MindSpeed AdamW (mindspeed/core/optimizer/adamw.py:142) 首次 step 时在 NPU 上
    # 建 exp_avg/exp_avg_sq (zeros_like(p), p 在 NPU) -> 42.8GB 已占时 OOM (大 batch n=6 触发).
    # 用 override_optimizer_config 强制 optimizer state 走 CPU (hybrid optimizer).
    # [AllReduce fix] 关 use_precision_aware_optimizer: 它让参数/grad 保 bf16, 与其他 rank fp32 grad
    # 在 get_grad_norm AllReduce 时 dataType 不一致 (EI0005: local 4 fp32 vs remote 11 bf16).
    # 参照 test_kimi_2node_mopd.sh 能跑通: 它 override_optimizer_config={} (4件套全没开), 因 batch 小
    # 不需 offload. 我们 batch 大需 offload, 但只开 offload 不开 precision_aware -> state 在 CPU fp32,
    # NPU 显存不增 (不回 OOM), grad dtype 统一 (解 AllReduce).
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False  # [step2/4 OOM] True->False: optimizer 阶段 _copy_model_grads_to_main_grads alloc 114MB 碎片化崩 (forward peak 46.37 + optimizer ~2.4 = 48.8GB, NPU 贴满). overlap 模式 d2h/h2d 有 prefetch buffer 占 NPU, 关了同步 offload 省 buffer. 代价: optimizer step 慢 (d2h 不 overlap). 保留 optimizer_cpu_offload=True + offload_fraction=1 (state 仍走 CPU).
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
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
    # [AllReduce fix] MoE + use_precision_aware_optimizer 下, router 默认 bf16 与其他参数 fp32 的
    # grad 在 get_grad_norm_fp32 AllReduce 时 dataType 不一致 (EI0005: local 4 fp32 vs remote 11 bf16).
    # 官方 grpo_megatron MoE 示例 (run_qwen3_vl_30b_a3b_megatron.sh:92) 都配套设 moe_router_dtype=fp32.
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}  # 解耦: rollout 8 die/worker 只能 EP8, 不跟 actor EP16 (砍 alloc 治碎片)
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    # n=8: GRPO needs >1 samples per prompt for group-relative advantage. n=1 gives ~0
    # advantage (no in-group contrast) -> task reward produces no gradient. With batch=64,
    # n=8 => 64 prompts x 8 samples = 512 rollouts. (test_kimi 参照是 n=1, n=8+megatron 未验证)
    actor_rollout_ref.rollout.n=4
    # fullvocab 只要求 teacher max_num_seqs=1 (代码强制, 见下方 teacher 块). 学生 rollout 无 full_vocab
    # 约束 (hidden 导出是 teacher 的事, 学生只管生成). 配 32 (baseline) 加速 rollout.
    actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.scheduling_policy=fcfs
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TOP_P}
    actor_rollout_ref.rollout.top_k=${TOP_K}
    # enforce_eager=True / chunked_prefill=True, 日志证实对 vLLM retain 无效 (vLLM v1 本就按
    # 学生 max_num_batched_tokens 2048->1024 压 prefill 激活 pool 治.
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
    trainer.default_local_dir=/path/to/ckpt
    # keep ALL ckpts (don't delete the peak ~step 25-30). Crash is unavoidable (external
    # teacher), so we early-stop at the peak and keep every ckpt to pick the best.
    trainer.max_actor_ckpt_to_keep=30
)

REWARD=(
    reward.custom_reward_function.path=${reward_fn_path}
    reward.custom_reward_function.name=compute_score
)

# [4node][megatron] 构建 CANN env 透传数组 (在 head 上动态读当前 env 值, 不硬编码路径).
# PYTHONPATH 特殊: verl 源码前缀 + 保留 head 原 PYTHONPATH (含 CANN python/site-packages/mindspeed 等).
# 其余原值透传. 空值的变量跳过 (避免传空 LD_LIBRARY_PATH 覆盖 worker 默认).
# 不透传 PATH (超长且会覆盖 worker 自己的 PATH, CANN bin 已在 LD_LIBRARY_PATH/ASCEND_HOME_PATH 间接覆盖).
CANN_ENV_PROPAGATE=()
CANN_ENV_NAMES=(PYTHONPATH LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH \
    ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_TOOLKIT_LATEST_HOME \
    ATB_HOME_PATH TOOLCHAIN_HOME CMAKE_PREFIX_PATH ASCEND_RUNTIME_OPTIONS \
    HCCL_OP_EXPANSION_MODE HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE)
for _v in "${CANN_ENV_NAMES[@]}"; do
    _val="${!_v}"
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
    # --- teacher A: gsm8k-expert (27B) ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="$GSM8K_TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas=${TEACHER_NUM_REPLICAS_GSM8K}
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.enforce_eager=True
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    # +distillation.teacher_models.gsm8k.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.gsm8k.inference.enable_chunked_prefill=False  # 开fullvocab
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=1  # fullvocab: teacher 单请求/forward (代码强制 distillation.py:222, readme 漏列)
    +distillation.teacher_models.gsm8k.inference.max_model_len=$((max_prompt_length + max_response_length + 1))
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=$((max_prompt_length + max_response_length + 1))
    # --- teacher B: math-expert (27B) ---
    # 数据 orz_math_train.parquet 的 data_source='math' (实测), teacher key 必须匹配 -> key="math".
    # (原脚本沿用 "mbpp" 旧称, 与实际 math 数据对不上 -> No teacher configured for 'math')
    +distillation.teacher_models.math.key="math"
    +distillation.teacher_models.math.model_path="$ORZ_TEACHER_MODEL"
    +distillation.teacher_models.math.num_replicas=${TEACHER_NUM_REPLICAS_MBPP}
    +distillation.teacher_models.math.inference.name=vllm
    +distillation.teacher_models.math.inference.enforce_eager=True
    +distillation.teacher_models.math.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.math.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    # +distillation.teacher_models.math.inference.max_model_len=${teacher_max_model_len}
    +distillation.teacher_models.math.inference.enable_chunked_prefill=False  # 开fullvocab
    +distillation.teacher_models.math.inference.max_num_seqs=1  # fullvocab: teacher 单请求/forward (代码强制 distillation.py:222, readme 漏列)
    +distillation.teacher_models.math.inference.max_model_len=$((max_prompt_length + max_response_length + 1))
    +distillation.teacher_models.math.inference.max_num_batched_tokens=$((max_prompt_length + max_response_length + 1))
    # --- loss: task-reward RL + distillation regularizer ---
    # L = policy_loss(task correctness) + distillation_loss_coef * distill_loss(k1)
    # task reward (gsm8k answer / mbpp code) is the main driver; distill keeps student
    # near teacher. k1 + use_policy_gradient=True (PG OPD); clamp=2 caps distill advantage.
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.full_vocab_chunk_tokens=${full_vocab_chunk_tokens} # 开fullvocab
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=2.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.distillation_loss_coef=0.5
    # [4node] multi-node: 把 NPU env 透传到所有 ray worker (否则 worker 侧拿不到 HCCL_BUFFSIZE 等
    # -> 仍用容器默认 2048 -> 跨机 HCCL OOM). 上方 head 的 export 只对 driver 生效.
    # 数值须与上方 export 段一致. 对齐 test_kimi_2node.sh:234-238 的写法.
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:256'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE='512'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GROUP_HCCL_BUFFER_SIZE='256'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE='0'"
    # [4node][megatron] 透传 CANN 全套 env 给 ray worker. head 登录时 source 了 set_env.sh,
    # PYTHONPATH/LD_LIBRARY_PATH/ASCEND_* 等都设好了, 但 ray worker 是子进程不继承.
    # megatron/mindspeed 后端要 import acl (CANN python/site-packages) + 找 libascendcl.so 等.
    # 下方 ENV_TO_PROPAGATE 列表在 head 上动态读当前值透传 (不硬编码路径, 自动跟随 CANN 版本/位置).
    # 前提: 4 机 CANN 装在同路径 (同一镜像). PYTHONPATH 把 verl 源码拼在 head 原值前面 (保留 CANN 路径).
    "${CANN_ENV_PROPAGATE[@]}"
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
    "$@" 2>&1 | tee /path/to/k26_cutlayer_4H_fullvocab${start_time}_k26_cutlayer_mopd_2teacher_fullvocab.log
