# Kimi-K2.6

正式 RL 入口为 [`kimi_k26_verl.sh`](kimi_k26_verl.sh)，使用 verl + Megatron + vLLM 进行 GRPO。OPD/MOPD 脚本保留为独立蒸馏入口。

## 正式 RL 运行

准备匹配的 verl/Megatron/MindSpeed/vLLM-Ascend 环境和 Ray 集群，确保所有节点可访问模型、数据与 reward 文件。脚本通过当前 Python 环境导入 verl，不会自动切换或安装框架。

从仓库根目录运行，替换下列共享路径：

```bash
MODEL_PATH=/path/to/Kimi-K2.6_bf16 \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/valid.parquet \
REWARD_FN_PATH=/path/to/reward.py \
OUTPUT_DIR=/path/to/checkpoints \
LOG_DIR=/path/to/logs \
WITH_TRACKING=false \
bash recipes/kimi-k2.6/kimi_k26_verl.sh
```

默认 `NNODES=40`、`GPUS_PER_NODE=16`，actor TP8/PP10/CP8/EP64/ETP1，rollout TP8/DP16；`ROLLOUT_EP` 默认不显式指定。prompt/response 上限默认 65536/8192，训练步数 1145，保存间隔 10。以上均为配置默认值，修改规模时需同步核对并行布局和内存需求。

模型、数据、reward、输出路径及上述配置支持环境变量覆盖；额外命令行参数透传为 Hydra overrides。默认开启 tensorboard/wandb（wandb offline），示例使用 `WITH_TRACKING=false` 只启用 console。脚本创建日志目录并真正启动训练，没有专门的 `--dry-run`。

正式入口指仓库采用的标准 RL 脚本，不代表已完成 NPU 训练验收。

## 脚本选择

| 文件 | 训练配置 |
| --- | --- |
| [kimi_k26_verl.sh](kimi_k26_verl.sh) | **正式 RL（GRPO）入口**，支持环境变量及 Hydra 参数覆盖 |
| [test_kimi_full_1.sh](test_kimi_full_1.sh) | 全层单教师 OPD，实际 loss 为 forward_kl_topk |
| [run_k26_4layers_mopd_2teacher_4node_fullvocab.sh](run_k26_4layers_mopd_2teacher_4node_fullvocab.sh) | 4 层、4 节点双教师 full-vocab MOPD 配置 |
| [run_k26_full_mopd_2teacher_fullvocab.sh](run_k26_full_mopd_2teacher_fullvocab.sh) | 全层双教师配置，默认 actor 40 节点、teacher 16 节点；实际默认 loss 为 reverse_kl_full_vocab，可用 DISTILLATION_LOSS_MODE 覆盖 |
| [test_kimi_full_mopd.sh](test_kimi_full_mopd.sh) | 根目录全层双教师 reverse-KL 配置，每教师 1 个 replica，保存间隔 10、总步数 1145 |
| [test_kimi_full_mopd_collate_teacher.sh](test_kimi_full_mopd_collate_teacher.sh) | 共享教师资源变体：share_gpu_group=True、max_awake_teachers=1，每教师 2 个 replica，保存间隔 100、总步数 30 |

部分脚本头部仍写 GRPO 或 forward KL，实际启用蒸馏的配置和 loss 参数才决定行为。共享教师变体默认 30 步小于保存间隔 100，不能以此配置宣称覆盖了周期 checkpoint 保存恢复。

## 环境与输入

正式 RL 使用上述训练环境；以下要求针对 OPD/MOPD。蒸馏需要含相应蒸馏实现的 verl、Megatron/MindSpeed 训练环境及匹配的 vLLM-Ascend 教师/rollout 后端。full-vocab KL、教师共享和 offload 能力需对应源码支持。模型 custom code 必须能被所有 Ray worker 导入。

运行前逐项配置：

1. BF16 模型和 tokenizer；学生与教师的 vocab 约束须符合 loss 实现。
2. GSM8K/math parquet 文件及其真实 `data_source`，与 teacher 路由 key 一致。
3. reward 函数路径，checkpoint 和日志目录。
4. actor、rollout、teacher 的并行度、replica 数和资源池；不要把所有并行维度简单相乘计算设备数。
5. Ray 集群、共享文件系统、HCCL 端口、节点环境一致性。

部分路径是固定赋值，外部 export 不能覆盖；先修改本仓库副本再运行。`run_k26_*` 启动脚本自身使用 VERL_ROOT 切换目录；`test_kimi_*` 使用当前已安装/通过 PYTHONPATH 提供的 verl。

## 运行方式

从本仓库根目录，在已准备的数据和 NPU 集群上选择其中一个：

```bash
# 单教师 OPD：先配置 PYTHONPATH 或安装匹配的 verl
bash recipes/kimi-k2.6/test_kimi_full_1.sh
# 裁层双教师 MOPD
VERL_ROOT=/path/to/patched-verl bash recipes/kimi-k2.6/run_k26_4layers_mopd_2teacher_4node_fullvocab.sh
# 全层双教师 MOPD
VERL_ROOT=/path/to/patched-verl bash recipes/kimi-k2.6/run_k26_full_mopd_2teacher_fullvocab.sh
```

## 状态

尚未完成 NPU 端到端验收；CPU 检查范围见 [验证说明](../../docs/VALIDATION.md)。RL 正式入口已确定，蒸馏脚本是独立候选配置；全层 OPD、裁层 MOPD 的历史通过记录不证明全层双教师或共享教师变体已通过。资源需求以最终配置和调度校验为准。

## 依赖分支核查

当前 RL/MOPD 使用已有 `aq/k26-integration` verl 分支；`vanilla_mbridge=False` 实际选择 Megatron-Bridge，不是 AQ mbridge。MindSpeed 容器基线尚未锁定，请勿直接套用 V4/V4.1 依赖组合。详见 [K2.6 分支核查](../../docs/DEPENDENCIES.md)。
