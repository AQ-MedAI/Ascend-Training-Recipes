# DeepSeek-V4-Flash

## 脚本

| 文件 | 用途 |
| --- | --- |
| [dsv4_flash_verl.sh](dsv4_flash_verl.sh) | 完整模型 GRPO 基础入口，也是完整 MOPD 的委托脚本 |
| [dsv4_flash_mopd.sh](dsv4_flash_mopd.sh) | 完整 MOPD：默认学生 8 节点 × 16 NPU、教师 4 节点 × 16 NPU |
| [dsv4_flash_rl_8npu_cutlayer.sh](dsv4_flash_rl_8npu_cutlayer.sh) | 8 NPU 裁层 RL，委托同目录裁层脚本 |
| [dsv4_flash_mopd_16npu_cutlayer.sh](dsv4_flash_mopd_16npu_cutlayer.sh) | 裁层训练实现，含模型准备和蒸馏配置 |
| [prepare_dsv4_flash_bf16_cut_model.py](prepare_dsv4_flash_bf16_cut_model.py) | 准备裁层 BF16 checkpoint，需要 torch 和 safetensors |
| [inspect_dsv4_checkpoint_metadata.py](inspect_dsv4_checkpoint_metadata.py) / [inspect_dsv4_flash_shapes.py](inspect_dsv4_flash_shapes.py) | checkpoint 元数据和形状检查 |
| [diagnose_dsv4_flash_vllm_prefill.sh](diagnose_dsv4_flash_vllm_prefill.sh) / [Python 实现](diagnose_dsv4_flash_vllm_prefill.py) | vLLM prefill 诊断 |

## 环境和运行

需要匹配的 verl、MindSpeed、MindSpeed-LLM、Megatron-LM、mbridge、vLLM、vLLM-Ascend；使用包含现有适配的环境，不能假定任意上游最新版兼容。完整 MOPD 脚本显式检查 `TransferQueue==0.1.7`。

先启动所需 Ray 集群，确认模型、parquet、reward 文件、输出目录和资源布局。完整 MOPD 包装脚本固定拓扑并拒绝命令行覆盖；模型、数据、输出路径及 `PYTHON_BIN` 可通过环境变量设置。基础 GRPO 和裁层脚本的环境变量以实际赋值为准。

从仓库根目录执行以下所选入口（不是顺序执行全部命令）：

```bash
# 完整模型 GRPO
bash recipes/deepseek-v4/dsv4_flash_verl.sh
# 完整模型 MOPD
bash recipes/deepseek-v4/dsv4_flash_mopd.sh
# 裁层 RL；可能先写出模型文件
bash recipes/deepseek-v4/dsv4_flash_rl_8npu_cutlayer.sh
```

保持这些脚本在同一目录，包装脚本使用 `SCRIPT_DIR` 查找被调用脚本和模型准备工具。诊断脚本运行方式及模型准备参数请查看文件；Python 工具的 `--help` 需要相应导入依赖。

## 状态

本目录提供 V4 完整模型和裁层配置。NPU 端到端验收待完成；CPU 检查范围见 [验证说明](../../docs/VALIDATION.md)，来源见 [来源说明](../../docs/PROVENANCE.md)。

## 依赖布局

V4 入口统一通过 `tools/v4_environment.sh` 初始化依赖：四个 fork 默认来自本仓库
`third_party`，Megatron-LM 和 vLLM/vLLM-Ascend 来自镜像。默认镜像根目录为
`/opt/ascend-training`，vLLM 路径复用镜像的 `workspace/rollout/dsv41/build-src` 布局；
目录名称不代表 V4 已完成兼容性验证。使用旧 V4 镜像时显式指定实际源码目录：

```bash
export DEPENDENCY_ROOT=/path/to/prepared-dependencies
export VLLM_ROOT="$DEPENDENCY_ROOT/vllm"
export VLLM_ASCEND_ROOT="$DEPENDENCY_ROOT/vllm-ascend"
source tools/activate.sh v4
```

运行前按 V4 依赖清单准备子模块；仅执行 `activate.sh v4` 不会切换 Git 提交。
在每个节点保持源码和动态库路径一致，并从新的终端环境启动，避免沿用 V4.1 的专用配置。
启动脚本会将源码路径、动态库路径传给 Ray；V4 不使用 V4.1 专用 QLI 初始化脚本。
