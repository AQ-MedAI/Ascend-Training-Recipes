# 依赖版本

本仓库提供训练配置与启动脚本。以下是配套框架源码版本，并非完整可安装环境。NPU 端到端验收尚未完成；使用前需补齐本页列出的依赖。

| 配置 | 仓库 | 开发分支 | 固定提交 |
| --- | --- | --- | --- |
| v41 | [MindSpeed](https://github.com/AQ-MedAI/MindSpeed) | `aq/v41-integration` | `28c2945cbbafadbb1888eb5bef53195e06ddf44c` |
| v4 | [MindSpeed](https://github.com/AQ-MedAI/MindSpeed) | `aq/v4-integration` | `dcfc6c22b04a4c966d05e128421959f60e0e80c2` |
| v41 | [MindSpeed-LLM](https://github.com/AQ-MedAI/MindSpeed-LLM) | `aq/v41-integration` | `7ffccdc3fdea5f8eb25b5ce88aa0d45f9faa5042` |
| v4 | [MindSpeed-LLM](https://github.com/AQ-MedAI/MindSpeed-LLM) | `aq/v41-integration` | `7ffccdc3fdea5f8eb25b5ce88aa0d45f9faa5042` |
| v41 | [mbridge](https://github.com/AQ-MedAI/mbridge) | `aq/v41-integration` | `7b0035f9e9b9aea540f0fb39861468a9cf6ff9e8` |
| v4 | [mbridge](https://github.com/AQ-MedAI/mbridge) | `aq/v41-integration` | `7b0035f9e9b9aea540f0fb39861468a9cf6ff9e8` |
| v41 | [verl](https://github.com/AQ-MedAI/verl) | `aq/v41-integration` | `c73f779e4d6ebbf6172c3b020e47b3fd2f346d52` |
| v4 | [verl](https://github.com/AQ-MedAI/verl) | `aq/v4-integration` | `e001090e88532bda2d4264bb806bc914fe7c6822` |
| k26 | [verl](https://github.com/AQ-MedAI/verl) | `aq/k26-integration` | `be9a6dc3a01cadea0035fcb3e105eb665209d05e` |

复现实验应固定 commit，而非跟随分支最新版本。机器可读版本见 [registry.json](../dependencies/registry.json)。

## V4 与 V4.1

MindSpeed-LLM、mbridge 共用版本。MindSpeed 与 verl 分开固定：V4.1 使用新的 Megatron rank-owned optimizer 接口和 swap checkpoint 格式，需要与匹配的 Megatron-LM 同时部署；不能仅替换其中一个包。

V4.1 rollout 源码：

- [vLLM](https://github.com/vllm-project/vllm)：`a97dacb7106ee49f39f3d1fc6ae1800ff724e01d`。
- [vLLM-Ascend](https://github.com/GDzhu01/vllm-ascend)：`e67ab649500fcb21fd2139da17d0e566f7364301`。

仅安装这两个包不足以运行 V4.1；还需权重同步 runtime、Engram 实现、原生算子及其构建环境。这些组件的独立分发和安装方式尚未齐备。V4 rollout 的完整源码/补丁发布也待补齐。

## K2.6

RL 与 MOPD 共用 verl 的 aq/k26-integration，蒸馏默认关闭。当前配置使用 model_engine=megatron、vanilla_mbridge=False，实际选择 Megatron-Bridge，不是 ISEEKYAN/mbridge；MindSpeed-LLM 也不是当前配置选择的模型入口。

K2.6 的 MindSpeed、Megatron-LM、Megatron-Bridge 与 rollout 配套版本尚未完整固定，不能直接套用 V4/V4.1 的组合。需要确认 Kimi 模型支持、CP、HF 导出接口以及 checkpoint 保存恢复能力。

## 安装和验证

在满足驱动、CANN、PyTorch/torch-npu 兼容性的环境中克隆对应框架并检出表中提交，再按各框架安装说明安装。所有 Ray 节点须使用一致的源码和包版本。本仓库暂不提供完整环境安装器或可直接运行的公共镜像。

补齐版本后，分别验证：模型加载、训练更新、rollout 权重同步、生成、checkpoint 保存与恢复。完整模型、裁层和教师共享配置应分别验收。CPU 检查范围见 [验证说明](VALIDATION.md)。
