# Ascend 镜像与 recipe 分层

**候选构建文件，尚未完成 Docker 构建或 NPU 验收。** 镜像提供稳定框架环境，recipe 提供四个 fork、runtime 和训练配置。

| 层 | 内容 | 使用范围 |
| --- | --- | --- |
| `common` | CANN 9.1.0 A3、Ubuntu 22.04、Python 3.12、torch 2.10.0、torch-npu 2.10.0.post4、Triton Ascend 3.2.2、Triton 3.5.0 | V4/V4.1 的公共构建基础，不是完整训练环境 |
| `v41` | Megatron-LM 基线加补丁、固定 vLLM/vLLM-Ascend、原生算子、训练与 rollout 共用的 Python 依赖 | V4.1；不宣称兼容 V4 |
| recipe | 四个 fork、`workspace/rollout/dsv41/runtime`、启动与初始化脚本 | 运行时克隆，在全部 Ray 节点安装/激活 |

基础标签来自 [verl 上游 Dockerfile](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3_v0.9.1)：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-ubuntu22.04-py3.12`。正式发布仍需固定目标架构 digest。CANN/ATB、编译器及驱动配套须验收。当前构建目标沿用 `ascend910_9382`，与上游示例的 `9392` 不同，需按实际芯片确认。

镜像构建不会克隆、安装四个 AQ-MedAI fork，也不复制 recipe runtime。它只从 registry 读取 vLLM/vLLM-Ascend 的固定提交。框架位于 `/opt/ascend-training/`，版本收据位于 `/opt/v41-image/frameworks.json`。

## 固定框架

- Megatron-LM：0.12.1，上游 `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17` + [补丁](megatron-v41.patch)，来源与八个修改文件的校验值见 [登记](megatron-source.json)，许可证见 [许可证](Megatron-LM-LICENSE)。已验证基线加补丁重建参考源码；不是 NPU 验收。
- vLLM：`a97dacb7106ee49f39f3d1fc6ae1800ff724e01d`。
- vLLM-Ascend：`e67ab649500fcb21fd2139da17d0e566f7364301`。
- Transformers：训练和 rollout 统一为 5.14.1，使用同一 Python 环境；训练兼容性尚需 NPU 验收。

## 构建与使用

Dockerfile 使用 BuildKit 自带的前端，不额外拉取 `docker/dockerfile:1`。构建器需支持 `RUN --mount=type=cache` 和 heredoc；旧版构建器遇到语法错误时应升级 Docker/BuildKit，不要关闭 BuildKit。基础镜像和其他构建材料仍需联网下载。

从 recipe 根目录构建公共基础层：

```bash
docker build --platform linux/arm64 --target common \
  -f docker/v41/Dockerfile -t ascend-training-common:dev .
```

可以直接尝试构建 V4.1 开发镜像；没有完整 runtime.lock 时，按 [固定版本与安装说明](locks/README.md) 安装：

```bash
docker build --platform linux/arm64 --target v41 \
  -f docker/v41/Dockerfile -t ascend-training-v41:dev .
```

容器运行后，在各节点的同一挂载路径准备 recipe：

```bash
git clone https://github.com/AQ-MedAI/Ascend-Training-Recipes.git
cd Ascend-Training-Recipes
bash tools/setup_sources.sh v41
bash tools/setup_environment.sh v41
source tools/activate.sh v41
```

初始化先核对四个 fork 的 commit、镜像框架版本和 Engram 校验值，再执行 `pip install -e --no-deps --no-build-isolation`，最后报告依赖检查结果（声明冲突不阻断初始化）。不自动升级依赖或现场重建镜像算子。所有 Ray 节点需使用一致镜像、recipe 和路径，再按模型 README 启动训练。设备/驱动、共享内存和网络挂载需按机器配置。

V4 可用 `bash tools/setup_sources.sh v4` 选择登记的源码版本，但当前 `setup_environment.sh v4` 会明确停止：V4 的镜像版本组合尚未完整锁定，不能默认复用 V4.1 框架层。切换 profile 后子模块工作区可能显示与默认 V4.1 引用不同，不要把临时选择误提交为默认引用。

## 仍待补齐

1. 构建工具的 `locks/build.lock` 已生成（14 个包，固定版本和 SHA256）；[锁文件说明](locks/README.md) 提供生成命令和源码约束。仍缺 `locks/runtime.lock`：包含完整 Python 依赖、固定版本和 wheel 哈希。统一运行锁包含 uv/构建工具及四个 fork 需要的 Python 包。硬件 bootstrap 目前仅固定版本，cp312 wheel 哈希仍需补齐。 完整运行锁不再是开发构建的前置条件。
2. FastAPI、Starlette、OpenCV、NumPy 及关联包已按固定部署版本配置显式覆盖。依赖声明差异会报告，不阻断安装；必要时使用 `--build-arg PYTHON_DEPENDENCY_MODE=no-deps` 跳过运行依赖解析。
3. 基础镜像 digest、CANN/ATB/ABI 和 NPU 验收；公共层也尚未实测构建。镜像中缺少 Python 3.12 头文件时需补齐对应开发环境，不能安装 Ubuntu 默认的其他版本头文件代替。
4. V4 的完整框架/算子组合与兼容性验收。

runtime 源码已经随 recipe 提供，不再需要外部 runtime 发布包或 `public-inputs/workspace`。原生算子的四份准备/构建辅助脚本放在 `build-support/`，自动获取材料后在镜像中编译；不依赖原开发仓库。

## 自动下载原生依赖

[native-dependencies.json](native-dependencies.json) 登记 14 份材料。构建自动调用 [download_dependencies.py](download_dependencies.py)：12 份通过 HTTPS 下载，2 份 Catlass 归档通过 Git 读取公开仓库的固定提交。Git 模式需要安装 Git，不检出或执行仓库中的代码。下载会校验 SHA256，使用临时文件完成后原子写入缓存；网络错误最多尝试三次，哈希不符立即拒绝使用。已存在的材料也必须通过校验。

可以在构建前预下载，成功的文件会保留，缺失或不匹配的条目会汇总报错：

```bash
python3 docker/v41/download_dependencies.py \
  --workspace /path/to/native-cache/workspace
```

本地默认缓存为 `~/.cache/ascend-training-recipes`，也可用 `--cache /path/to/cache` 指定。Docker 构建使用 BuildKit 缓存；它与宿主缓存不是同一目录，当前 Docker 构建自行下载到 BuildKit 缓存，宿主预下载目录用于独立检查，不会自动复制到镜像。

```bash
# 检查已准备的材料；不访问网络。
python3 docker/v41/download_dependencies.py \
  --workspace /path/to/native-cache/workspace --offline
```

可用 `--group native` 或 `--group premix` 单独处理一组。`--build-arg DEPENDENCY_OFFLINE=1` 仅禁用本下载器联网；Git、APT、pip 仍可能联网，因此它不是完整的离线镜像构建开关。

### 已核对与尚不能确定的项目

- makeself `2.5.0-patch1.0`、nlohmann/json `3.11.3`、abseil `20230802.1`、protobuf `25.1`：公开下载已实测，SHA256 与参考材料一致。
- Premix 的 CMake `master-058` 与 abseil CVE 补丁：公开下载已实测匹配。其余六份 Premix 材料已从现有 manifest 登记 URL/版本，校验值与本地文件一致，尚未逐份联网复验；运行下载器时会强制校验。
- `catlass.tar.gz` 和 `catlass_git.tar.gz`：已从 [MrJVium/workspace](https://gitcode.com/MrJVium/workspace) 的固定提交 `86bf5287c347090e099ac8a36e35dd7f714eba39` 获取并验证，SHA256 与原材料完全一致。raw 端点曾返回 HTTP 418，因此采用 Git 读取固定提交中的两个文件。这里固定的是归档存储仓库的提交；Catlass 源码自身的 commit 仍为 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。Premix 的另一个 Catlass commit 是独立依赖，不能混用。
- Triton Ascend wheel 不在这 14 份源码材料中。基础镜像采用 Python 3.12 后，使用 [hardware-requirements.txt](hardware-requirements.txt) 中的公开 `cp312/aarch64` wheel，并通过 [hardware-constraints.txt](hardware-constraints.txt) 固定四个硬件包版本。公开索引已列出该 wheel，但其内容和 SHA256 尚未独立下载核验；先前 cp311 的校验结果不能用于 cp312。

硬件包在构建初期由 pip 安装并解析依赖；这是固定核心版本的 bootstrap，不是完整的哈希锁文件。正式发布前需记录 cp312 wheel 哈希并锁定全部依赖。后续 `runtime.lock` 安装也受硬件版本约束，冲突会报错；不能自动升级这些包来绕过冲突。

下载器不会上传或发布这些文件。第三方许可和来源说明仍需随最终发布材料提供。

## 仅能通过 OSS 和 Dockerfile 导入的平台

本地镜像构建完成后，使用 [OSS 导出与重建流程](OSS_IMPORT.md)。导出脚本生成 rootfs 归档及包含原镜像运行配置的 Dockerfile；内部平台只需能拉取工具镜像并下载 OSS 文件，无需执行 `docker load`。
