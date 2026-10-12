# 来源与第三方声明

- DeepSeek V4 recipe 参考 [verl-ascend-recipe](https://github.com/verl-project/verl-ascend-recipe) 的训练和模型准备脚本；本仓库维护选定的脚本，verl-ascend-recipe 本身不是额外的 Python 运行时依赖。
- Kimi K2.6 recipe 基于 [verl](https://github.com/verl-project/verl) 训练入口，由 AQ-MedAI 维护 RL/MOPD 配置。
- DeepSeek V4.1 recipe 和配套集成配置由 AQ-MedAI 维护。

保留源文件已有版权与许可证声明；随附 licenses/ 中的第三方许可证。公开副本的默认输入/输出路径已替换为示例路径。框架改造在对应 fork 中维护，版本见 [依赖说明](DEPENDENCIES.md)。

V4.1 的原生算子由镜像构建，第三方归档的来源与许可证需随最终镜像提供。本仓库许可证不替代模型或数据许可。

V4.1 runtime 随本 recipe 维护，模块路径与固定 verl 入口兼容；迁移调整了框架路径解析。镜像的 Megatron-LM 基线、补丁和许可证见 [镜像目录](../docker/v41/README.md)。原生构建辅助代码与下载材料分别维护，第三方材料保留各自来源及许可证要求。
