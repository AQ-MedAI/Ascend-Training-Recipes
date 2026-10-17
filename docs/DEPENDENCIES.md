# 依赖版本

本仓库提供训练配置与启动脚本。以下是配套框架源码版本，并非完整可安装环境。NPU 端到端验收尚未完成；使用前需补齐本页列出的依赖。

| 配置 | 仓库 | 源码方案 | 固定提交 |
| --- | --- | --- | --- |
| v41 | [MindSpeed](https://github.com/Ascend/MindSpeed) | 上游 + common.patch + v41.patch | `c356513e331e55d53b799cfde51c7b23901d23dc` |
| v4 | [MindSpeed](https://github.com/Ascend/MindSpeed) | 上游 + common.patch + v41.patch | `c356513e331e55d53b799cfde51c7b23901d23dc` |
| v41 | [MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM) | 上游 + common.patch + v41.patch | `53ff1095801f9ce14b14be3e62739e742f8b9067` |
| v4 | [MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM) | 上游 + common.patch + v41.patch | `53ff1095801f9ce14b14be3e62739e742f8b9067` |
| v41 | [mbridge](https://github.com/ISEEKYAN/mbridge) | 上游 v0.15.1 + patch | `0cd4ae23f2425da77a80cb3f517828452fa8e984` |
| v4 | [mbridge](https://github.com/ISEEKYAN/mbridge) | 上游 v0.15.1 + patch | `0cd4ae23f2425da77a80cb3f517828452fa8e984` |
| v41 | [verl](https://github.com/verl-project/verl) | 上游 + v4.patch + v41.patch + entrypoint.patch | `809f2d8f59910dfd3a96745b68119829650250a0` |
| v4 | [verl](https://github.com/verl-project/verl) | 上游 + v4.patch + v41.patch + entrypoint.patch | `809f2d8f59910dfd3a96745b68119829650250a0` |
| k26 | [verl](https://github.com/verl-project/verl) | 上游 + k26.patch | `bee9f6f4bb1de784781e296f165b4c22d6c26cc1` |

复现实验应固定 commit，而非跟随分支最新版本。机器可读版本见 [registry.json](../dependencies/registry.json)。

## 子模块与版本更新

四个依赖均使用上游固定提交加补丁，分别见 [verl](../patches/verl/README.md)、 [MindSpeed](../patches/mindspeed/README.md)、[MindSpeed-LLM](../patches/mindspeed-llm/README.md) 和 [mbridge](../patches/mbridge/README.md)。框架格式遵循上游配置，补丁文件禁止自动格式化。

`third_party/` 中的四个 Git 子模块默认记录 V4.1 基线。`.gitmodules` 保存公开仓库地址，Git 子模块引用保存实际提交；`registry.json` 记录各模型的兼容组合。不要使用 `git submodule update --remote` 复现实验，它会跟随远端分支而绕过记录的版本。

MindSpeed-LLM 两种配置共用完整源码，均应用公共补丁和包含 `engram_hash.py`、Transformers 约束的增量补丁。拉齐四个框架后仍须准备下文的运行时和原生算子。

维护者更新依赖时，需同步上游基线、补丁校验值、完整源码清单、registry 与本页版本；在 recipe 仓库提交 `git add third_party/<仓库名>` 记录的上游引用，不在子模块中提交补丁。`.gitmodules` 与四个子模块引用必须随 recipe 一同提交，其他人初始化源码时才会得到正确版本。

V4 使用本页的 V4 提交组合，并通过各脚本支持的路径配置指向对应 checkout。K2.6 仍有未固定组件，不使用默认 V4.1 组合替代。现有平级开发目录可继续保留；子模块是面向使用者的源码获取入口。

mbridge 的子模块引用记录上游基线，`setup_sources.sh` 自动校验并应用 `patches/mbridge/ascend-training.patch`。补丁后的完整源码清单记录在 `dependencies/mbridge-sources.json`。应用后 `git status` 显示 mbridge 内部修改是正常状态；额外的源码修改或暂存修改会阻止再次初始化。不要在子模块中提交这份补丁，也不要用 `git restore` 丢弃它后直接运行训练。更新补丁时需同步校验值、源码清单和基线引用。

## V4 与 V4.1

V4/V4.1 共用四个依赖的最终源码。verl 先应用 `v4.patch` 再叠加 `v41.patch` 和入口路径修复 `entrypoint.patch`；MindSpeed 和 MindSpeed-LLM 均应用公共补丁及 V4.1 增量。模型架构和训练参数决定执行路径。共享 optimizer/checkpoint 接口仍需与匹配的 Megatron-LM 同时部署。

V4.1 rollout 源码：

- [vLLM](https://github.com/vllm-project/vllm)：`a97dacb7106ee49f39f3d1fc6ae1800ff724e01d`。
- [vLLM-Ascend](https://github.com/GDzhu01/vllm-ascend)：`e67ab649500fcb21fd2139da17d0e566f7364301`。

V4.1 权重同步 runtime 已随本仓库 `workspace/rollout/dsv41/runtime` 提供；原生算子由镜像构建。Engram hash 由 MindSpeed-LLM 提供。使用方式见 [镜像分层说明](../docker/v41/README.md)，Python 依赖锁定及 NPU 验收仍待补齐。V4 rollout 的完整源码/补丁发布也待补齐。

## K2.6

RL 与 MOPD 共用 verl 的 `k26.patch`，蒸馏默认关闭。当前配置使用 model_engine=megatron、vanilla_mbridge=False，实际选择 Megatron-Bridge，不是 ISEEKYAN/mbridge；MindSpeed-LLM 也不是当前配置选择的模型入口。

K2.6 的 MindSpeed、Megatron-LM、Megatron-Bridge 与 rollout 配套版本尚未完整固定，不能直接套用 V4/V4.1 的组合。需要确认 Kimi 模型支持、CP、HF 导出接口以及 checkpoint 保存恢复能力。

## 安装和验证

在满足驱动、CANN、PyTorch/torch-npu 兼容性的环境中克隆对应框架并检出表中提交，再按各框架安装说明安装。所有 Ray 节点须使用一致的源码和包版本。本仓库暂不提供完整环境安装器或可直接运行的公共镜像。

补齐版本后，分别验证：模型加载、训练更新、rollout 权重同步、生成、checkpoint 保存与恢复。完整模型、裁层和教师共享配置应分别验收。CPU 检查范围见 [验证说明](VALIDATION.md)。

MindSpeed-LLM 的子模块同样记录上游基线；`dependencies/mindspeed-llm-sources.json` 分别校验 V4 和 V4.1 的完整源码。使用 `bash tools/setup_sources.sh v4` 或 `bash tools/setup_sources.sh v41` 切换。切换前验证所有依赖；仅撤销与清单完全匹配的已知补丁，额外修改或暂存内容会阻止操作。子模块内部的补丁修改无需单独提交。

MindSpeed 共用固定上游基线，V4 与 V4.1 均应用 `patches/mindspeed/common.patch` 和 `v41.patch`。完整清单在 `dependencies/mindspeed-sources.json`；初始化与切换同样由 `setup_sources.sh` 校验和管理。V4.1 增量包含 optimizer/checkpoint 接口差异和 NumPy 1.26.4 约束，不能省略后直接运行 V4.1。

使用 `bash tools/setup_sources.sh k26` 仅准备 K2.6 的 verl，不切换其他三个依赖，也不代表完整 K2.6 环境已就绪。V4/V4.1 共用层叠补丁，与 K2.6 的 verl 补丁互斥；脚本校验并撤销当前已知补丁后切换到目标基线。
