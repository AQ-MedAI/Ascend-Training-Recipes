# 框架依赖

[registry.json](registry.json) 记录框架仓库、分支和固定提交，不是完整环境 lockfile。[依赖说明](../docs/DEPENDENCIES.md)列出各模型的组合、兼容性限制与待补齐组件。

框架源码在独立 fork 中维护，本仓库不内置第三方框架源码。V4.1 rollout 使用固定的 vLLM/vLLM-Ascend 源码和额外运行时；K2.6 选择 Megatron-Bridge，其版本仍待固定。
