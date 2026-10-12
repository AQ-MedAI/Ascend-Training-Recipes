# V4.1 recipe runtime

本目录随 recipe 发布，负责 rollout worker、权重同步和 Engram 传输。`workspace.rollout.dsv41.runtime` 是固定 verl 提交使用的模块路径，因此保留该包名。

四个框架从 recipe 的 `third_party/` 加载；Megatron-LM、vLLM、vLLM-Ascend 与原生算子从镜像加载。先运行 `bash tools/setup_environment.sh v41`，再 `source tools/activate.sh v41`。所有 Ray 节点需要相同源码版本及可访问的目录。

迁移只调整源码路径，尚未完成新布局的 NPU 端到端验收。此目录不包含模型、数据、编译产物或原环境日志。
