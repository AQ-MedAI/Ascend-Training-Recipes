# DeepSeek-V4.1-Flash

## 入口

| 文件 | 用途与当前默认值 |
| --- | --- |
| [dsv41_flash_rl.sh](dsv41_flash_rl.sh) | 多节点 RL，默认 16 节点、每节点 16 NPU、120 个训练步；使用 MODEL_PATH |
| [dsv41_flash_main_ppo_16npu.sh](dsv41_flash_main_ppo_16npu.sh) | 单机原生 main_ppo 验收候选；使用 MODEL，默认裁剪模型、response 长度 1、5 个训练步 |
| [dsv41_flash_rl_16npu.sh](dsv41_flash_rl_16npu.sh) | 委托 main_ppo 单机入口的兼容别名 |

默认值描述的是配置，不表示对应规模已经验收。多节点脚本与单机脚本并非同一配置的别名。

## 外部依赖

需要额外准备兼容的框架、V4.1 runtime 和原生算子；完整依赖包的独立发布尚未完成，见 [依赖说明](../../docs/DEPENDENCIES.md)。除框架目录外，以下内容仍来自该依赖仓库，未复制进本 recipe：

- `workspace/rollout/dsv41/`：固定 rollout 源码、runtime 和隔离 Python 环境。
- `workspace/tests/dsv41/native_operator_env.sh`：多节点入口按所选 backend 加载的算子环境。
- `workspace/rollout/dsv41/build-src/`：原生算子、Premix 等实际构建产物。

复制后的两个启动脚本使用 `ROOT=${WORK_DIR:-${DEPENDENCY_ROOT}}`，不再从 recipe 所在位置推导依赖根目录。`DEPENDENCY_ROOT` 指向完整依赖仓库；如果覆盖 `WORK_DIR`，该路径也必须包含上述 workspace 内容。

## 单机运行示例

在已准备的 NPU 环境和 Ray 配置下，从本仓库根目录执行。以下 `/path/to/` 都需替换成真实的共享路径。

```bash
export DEPENDENCY_ROOT=/path/to/dsv4_flash_npu_dependency
export MODEL=/path/to/prepared-v41-model
export TRAIN_FILES=/path/to/train.parquet
export VAL_FILES=/path/to/valid.parquet
export REWARD_FN_PATH=/path/to/reward.py
export REPORT_DIR=/path/to/new-report-directory
export ROLLOUT_PYTHON="$DEPENDENCY_ROOT/workspace/rollout/dsv41/.venv/bin/python"
bash recipes/deepseek-v4.1/dsv41_flash_main_ppo_16npu.sh
```

裁剪 Engram 时另行配置 `ENGRAM_CROP` 和 `ENGRAM_SOURCE`。正式长度和步数使用脚本实际支持的 `MAX_RESPONSE_LENGTH`、`TOTAL_TRAINING_STEPS` 等变量。多节点入口使用 `MODEL_PATH`，并应核对 `NNODES`、`GPUS_PER_NODE`、原生 backend 和输出路径；不能直接套用单机的 MODEL 变量。

## 验证边界

尚未完成 NPU 验收。历史裁剪模型在线测试、独立同步测试及用户运行反馈，不能替代当前入口的端到端报告。发布前需分别验证当前多节点配置、单机在线更新、派生权重刷新、Engram、保存恢复和资源峰值。训练/rollout logprob 差异单独报告，不能将流程完成当作数值等价。

## Engram hash 实现

Engram hash 随 MindSpeed-LLM 的 V4.1 模型源码提供：`mindspeed_llm/tasks/models/transformer/deepseek41/engram_hash.py`。设置 `MINDSPEED_LLM_ROOT` 指向对应源码根目录；两个入口默认从该目录加载，不再依赖外部 `workspace/inference/engram.py`。自定义实现通过 `ENGRAM_HASH_SOURCE=/shared/path/engram.py` 指定，所有 Ray 节点必须能读取同一路径。

裁剪/小表模型会校验 hash 源码的 SHA256。随附实现保持原字节内容以兼容已有模型；修改实现或仅格式化该文件也会改变 SHA256，不应绕过一致性检查。运行需要 numpy、torch、sympy、tokenizers（以及模型加载所需的 transformers）。

### 默认框架目录

先按[主 README](../../README.md#拉取源码)运行 `tools/setup_sources.sh` 初始化四个子模块。两个 V4.1 启动脚本默认从仓库的 `third_party/` 加载四个 fork；`DEPENDENCY_ROOT` 用于其余外部依赖，四个框架路径可分别通过 `*_ROOT` 覆盖。MindSpeed-LLM 已固定包含 Engram 的提交，详见[依赖说明](../../docs/DEPENDENCIES.md#子模块与版本更新)。
