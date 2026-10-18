# Ascend-Training-Recipes

DeepSeek V4、V4.1 与 Kimi K2.6 在 Ascend NPU 上的 GRPO、OPD 和 MOPD 训练配置与启动脚本。

**当前状态：开发预览。** 已进行限定范围的 CPU 和静态检查，尚未完成独立环境的 NPU 端到端验收。部分框架、算子和 rollout 运行时的配套版本及安装方式仍待补齐，当前不是开箱即用的训练套件。

## 选择 recipe

| 模型 | 内容 | 使用说明 |
| --- | --- | --- |
| DeepSeek V4 Flash | GRPO、MOPD、裁层准备与诊断 | [V4](recipes/deepseek-v4/README.md) |
| DeepSeek V4.1 Flash | 多节点 RL、单机验证配置 | [V4.1](recipes/deepseek-v4.1/README.md) |
| Kimi K2.6 | GRPO、OPD、双教师 MOPD、教师共享 | [K2.6](recipes/kimi-k2.6/README.md) |

## 使用镜像内 recipe

V4/V4.1 镜像在构建时已准备七个框架和匹配的 recipe。默认工作目录为 `/opt/ascend-training/Ascend-Training-Recipes`，启动时自动加载环境，无需再 clone、初始化子模块、安装框架或打补丁。

进入容器后，先设置模型、数据、reward 和集群参数，再按模型说明运行脚本：

```bash
cd /opt/ascend-training/Ascend-Training-Recipes
# 可选：检查预安装框架与依赖版本
bash tools/setup_environment.sh v41
# 配置好模型、数据和集群参数后启动
bash recipes/deepseek-v4.1/dsv41_flash_rl.sh
```

V4 使用 `recipes/deepseek-v4/dsv4_flash_rl.sh`，同样使用镜像内环境。两个 profile 共用 `/opt/ascend-training/third_party/` 下的七个框架。模型、数据、凭据和训练输出通过运行时配置与挂载提供，镜像不会自动启动训练。全部 Ray 节点应使用相同镜像和 recipe 路径。

构建从当前目录复制经过 Docker ignore 过滤的 recipe，版本记录在 `/opt/ascend-training/manifest/recipe.json`，包含逐文件 SHA-256 与源码快照摘要；可通过 `--build-arg RECIPE_REVISION=<commit>` 附加提交标识。构建和使用见 [镜像说明](docker/v41/README.md)。完整构建及 NPU 验收仍需在目标硬件上完成。

### 开发源码

可将开发 recipe 挂载到另一目录，并设置 `RECIPE_ROOT` 与工作目录。例如：

```bash
docker run --rm -it \
  -v "$PWD:/workspace/Ascend-Training-Recipes" \
  -e RECIPE_ROOT=/workspace/Ascend-Training-Recipes \
  -w /workspace/Ascend-Training-Recipes ascend-training-v41:dev
```

上例展示 recipe 覆盖方式；实际训练还需按机器配置挂载 NPU、驱动、模型和数据。

镜像内 recipe 不带 Git 元数据。需要修改框架时，先在开发目录另行 clone 本仓库，再运行 `bash tools/setup_sources.sh v41` 在 recipe 的 `third_party/` 初始化四个开发子模块，然后显式设置对应的 `VERL_ROOT`、`MINDSPEED_ROOT`、`MINDSPEED_LLM_ROOT` 或 `MBRIDGE_ROOT`。激活脚本尊重这些覆盖。开发源码需要安装时，在目标 Python 中执行 `python -m pip install --no-deps --no-build-isolation -e <checkout>`；`setup_environment.sh` 只做校验，不安装或更新框架。

`setup_sources.sh v4` 和 `v41` 准备相同源码。子模块不会自动覆盖镜像内框架。上游基线、补丁和校验清单见 [依赖说明](docs/DEPENDENCIES.md)；`dependencies/registry.json` 不是完整 Python lockfile。

K2.6 保留独立基线和环境适配流程，见 [K2.6](recipes/kimi-k2.6/README.md)。`setup_sources.sh k26` 只准备它的 verl，不能替代完整 K2.6 环境。

## 使用前准备

1. 查看 [依赖版本与缺失组件](docs/DEPENDENCIES.md)，准备匹配的 Ascend 软件环境与框架。
2. 准备模型、tokenizer、parquet 数据和 reward 函数。所有 Ray 节点必须能访问相同的输入和源码。
3. 将脚本中的 `/path/to/...` 示例路径替换为实际共享路径。支持 `${VAR:-default}` 的配置可以通过环境变量覆盖；固定赋值需编辑脚本。
4. 根据集群调整节点数、并行度、batch 和教师资源。默认规模只是配置，不代表对应规模已验收。
5. 按模型 README 启动 Ray 与训练。`bash` 启动命令会实际运行训练，裁层入口可能写入模型文件；没有统一的 `--dry-run` 参数。

```text
recipes/       # 模型配置、启动与诊断脚本
dependencies/  # recipe 依赖登记、框架基线与补丁校验清单
patches/       # 按框架组织的源码适配补丁
third_party/   # 可选的框架开发子模块；运行默认使用镜像内源码
docs/          # 依赖、验证与来源说明
tools/         # 源码/环境初始化、CPU 与启动参数检查
workspace/     # 与固定 verl 入口兼容的 V4.1 Python runtime
licenses/      # 第三方许可证
```

## 验证与限制

见 [验证说明](docs/VALIDATION.md)。CPU 通过不代表 NPU 内核、多节点训练、数值收敛或 checkpoint 恢复已经通过。公开数据/reward 示例、完整环境锁定和 NPU 验收报告仍待补齐。

## 来源与许可证

源码来源见 [来源说明](docs/PROVENANCE.md)。本仓库许可证见 [LICENSE](LICENSE)，第三方声明见 [verl 许可证](licenses/verl-LICENSE) 和 [verl-ascend-recipe 许可证](licenses/verl-ascend-recipe-LICENSE)。框架、模型与数据遵循各自许可证。

K2.6 使用 `bash tools/setup_sources.sh k26`，仅准备 verl 的 K2.6 基线与补丁；其他依赖和运行环境需另行准备，见 [K2.6](recipes/kimi-k2.6/README.md)。
