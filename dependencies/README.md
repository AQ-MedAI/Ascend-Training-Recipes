# 框架依赖

[registry.json](registry.json) 记录框架仓库、分支和固定提交，不是完整环境 lockfile。[依赖说明](../docs/DEPENDENCIES.md)列出各模型的组合、兼容性限制与待补齐组件。

框架源码在独立 fork 中维护，本仓库通过 `third_party/` 下的 Git 子模块引用四个 fork 的固定提交；初始化命令见[主 README](../README.md#拉取源码)。框架修改在各自 fork 中提交，recipe 只更新子模块引用；格式化仍使用各框架自己的配置。V4.1 rollout 使用固定的 vLLM/vLLM-Ascend 源码和额外运行时；K2.6 选择 Megatron-Bridge，其版本仍待固定。
