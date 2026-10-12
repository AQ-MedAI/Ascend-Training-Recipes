# Ascend-Training-Recipes

DeepSeek V4、V4.1 与 Kimi K2.6 在 Ascend NPU 上的 GRPO、OPD 和 MOPD 训练配置与启动脚本。

**当前状态：开发预览。** 已进行限定范围的 CPU 和静态检查，尚未完成独立环境的 NPU 端到端验收。部分框架、算子和 rollout 运行时的配套版本及安装方式仍待补齐，当前不是开箱即用的训练套件。

## 选择 recipe

| 模型 | 内容 | 使用说明 |
| --- | --- | --- |
| DeepSeek V4 Flash | GRPO、MOPD、裁层准备与诊断 | [V4](recipes/deepseek-v4/README.md) |
| DeepSeek V4.1 Flash | 多节点 RL、单机验证配置 | [V4.1](recipes/deepseek-v4.1/README.md) |
| Kimi K2.6 | GRPO、OPD、双教师 MOPD、教师共享 | [K2.6](recipes/kimi-k2.6/README.md) |

## 拉取源码

新环境克隆 recipe 后，运行源码初始化脚本即可自动拉齐 AQ-MedAI 的四个 fork：

```bash
git clone https://github.com/AQ-MedAI/Ascend-Training-Recipes.git
cd Ascend-Training-Recipes
bash tools/setup_sources.sh
```

已有 clone 或更新 recipe 后执行：

```bash
git pull --ff-only
bash tools/setup_sources.sh
```

四个 fork 位于 `third_party/{MindSpeed,MindSpeed-LLM,mbridge,verl}`，按 recipe 记录的提交检出，不跟随分支最新版本。有未提交改动时，初始化脚本会停止更新。V4.1 启动脚本默认使用这些目录，也可分别设置 `MINDSPEED_ROOT`、`MINDSPEED_LLM_ROOT`、`MBRIDGE_ROOT`、`VERL_ROOT` 使用其他 checkout。

Git 不会在普通 clone 后自动执行仓库脚本。也可用 `git clone --recurse-submodules`，但它还会下载框架内嵌的上游子模块（包括 GitCode 仓库）；这些依赖的版本不代表已通过本 recipe 验证。推荐上述初始化脚本，只拉取四个已登记的 fork。

各模型的仓库和固定提交记录在 [dependencies/registry.json](dependencies/registry.json)，它不是完整环境 lockfile。默认源码组合面向 V4.1。V4、K2.6 的版本差异见[依赖说明](docs/DEPENDENCIES.md)；克隆源码不会安装软件包、编译算子或下载模型。

## 使用前准备

1. 查看 [依赖版本与缺失组件](docs/DEPENDENCIES.md)，准备匹配的 Ascend 软件环境与框架。
2. 准备模型、tokenizer、parquet 数据和 reward 函数。所有 Ray 节点必须能访问相同的输入和源码。
3. 将脚本中的 `/path/to/...` 示例路径替换为实际共享路径。支持 `${VAR:-default}` 的配置可以通过环境变量覆盖；固定赋值需编辑脚本。
4. 根据集群调整节点数、并行度、batch 和教师资源。默认规模只是配置，不代表对应规模已验收。
5. 按模型 README 启动 Ray 与训练。`bash` 启动命令会实际运行训练，裁层入口可能写入模型文件；没有统一的 `--dry-run` 参数。

```text
recipes/       # 模型配置、启动与诊断脚本
dependencies/  # registry.json：各模型的依赖版本登记
third_party/   # 四个 fork 的 Git 子模块（默认 V4.1 基线）
docs/          # 依赖、验证与来源说明
tools/         # CPU 数值与启动参数检查
licenses/      # 第三方许可证
```

## 验证与限制

见 [验证说明](docs/VALIDATION.md)。CPU 通过不代表 NPU 内核、多节点训练、数值收敛或 checkpoint 恢复已经通过。公开数据/reward 示例、完整环境锁定和 NPU 验收报告仍待补齐。

## 来源与许可证

源码来源见 [来源说明](docs/PROVENANCE.md)。本仓库许可证见 [LICENSE](LICENSE)，第三方声明见 [verl 许可证](licenses/verl-LICENSE) 和 [verl-ascend-recipe 许可证](licenses/verl-ascend-recipe-LICENSE)。框架、模型与数据遵循各自许可证。
