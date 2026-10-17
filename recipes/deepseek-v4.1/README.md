# DeepSeek-V4.1-Flash

## 入口

| 文件 | 用途与当前默认值 |
| --- | --- |
| [dsv41_flash_rl.sh](dsv41_flash_rl.sh) | 多节点 RL，默认 16 节点、每节点 16 NPU、120 个训练步；使用 MODEL_PATH |
| [dsv41_flash_main_ppo_16npu.sh](dsv41_flash_main_ppo_16npu.sh) | 单机原生 main_ppo 验收候选；使用 MODEL，默认裁剪模型、response 长度 1、5 个训练步 |
| [dsv41_flash_rl_16npu.sh](dsv41_flash_rl_16npu.sh) | 委托 main_ppo 单机入口的兼容别名 |

默认值描述的是配置，不表示对应规模已经验收。多节点脚本与单机脚本并非同一配置的别名。

## 环境与源码

使用 [V4.1 框架镜像](../../docker/v41/README.md)，在各 Ray 节点的同一路径执行：

```bash
bash tools/setup_sources.sh v41
bash tools/setup_environment.sh v41
source tools/activate.sh v41
```

Megatron-LM、vLLM/vLLM-Ascend、rollout Python 和编译算子位于镜像的 `/opt/ascend-training`；四个源码依赖 和配套 runtime 位于 recipe。`DEPENDENCY_ROOT` 指镜像环境，`WORK_DIR` 是运行目录（默认 recipe 根目录）。不再要求原开发依赖仓库。完整镜像尚待依赖锁定和 NPU 验收。

## 单机运行示例

在已准备的 NPU 环境和 Ray 配置下，从本仓库根目录执行。以下 `/path/to/` 都需替换成真实的共享路径。

```bash
source tools/activate.sh v41
export MODEL=/path/to/prepared-v41-model
export TRAIN_FILES=/path/to/train.parquet
export VAL_FILES=/path/to/valid.parquet
export REWARD_FN_PATH=/path/to/reward.py
export REPORT_DIR=/path/to/new-report-directory
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export ROLLOUT_PYTHON="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
bash recipes/deepseek-v4.1/dsv41_flash_main_ppo_16npu.sh
```

裁剪 Engram 时另行配置 `ENGRAM_CROP` 和 `ENGRAM_SOURCE`。正式长度和步数使用脚本实际支持的 `MAX_RESPONSE_LENGTH`、`TOTAL_TRAINING_STEPS` 等变量。多节点入口使用 `MODEL_PATH`，并应核对 `NNODES`、`GPUS_PER_NODE`、原生 backend 和输出路径；不能直接套用单机的 MODEL 变量。

## 验证边界

尚未完成 NPU 验收。历史裁剪模型在线测试、独立同步测试及用户运行反馈，不能替代当前入口的端到端报告。发布前需分别验证当前多节点配置、单机在线更新、派生权重刷新、Engram、保存恢复和资源峰值。训练/rollout logprob 差异单独报告，不能将流程完成当作数值等价。

## Engram hash 实现

Engram hash 随 MindSpeed-LLM 的 V4.1 模型源码提供：`mindspeed_llm/tasks/models/transformer/deepseek41/engram_hash.py`。设置 `MINDSPEED_LLM_ROOT` 指向对应源码根目录；两个入口默认从该目录加载，不再依赖外部 `workspace/inference/engram.py`。自定义实现通过 `ENGRAM_HASH_SOURCE=/shared/path/engram.py` 指定，所有 Ray 节点必须能读取同一路径。

裁剪/小表模型会校验 hash 源码的 SHA256。随附实现保持原字节内容以兼容已有模型；修改实现或仅格式化该文件也会改变 SHA256，不应绕过一致性检查。运行需要 numpy、torch、sympy、tokenizers（以及模型加载所需的 transformers）。

### 默认框架目录

先按[主 README](../../README.md#拉取源码)运行 `tools/setup_sources.sh` 初始化四个子模块。两个 V4.1 启动脚本默认从仓库的 `third_party/` 加载四个源码依赖；`DEPENDENCY_ROOT` 用于镜像中的框架和算子，四个框架路径可分别通过 `*_ROOT` 覆盖。MindSpeed-LLM 已固定包含 Engram 的提交，详见[依赖说明](../../docs/DEPENDENCIES.md#子模块与版本更新)。
