# Ascend 镜像与 recipe 分层

**候选构建文件，尚未完成 Docker 构建或 NPU 验收。** 镜像提供稳定框架环境，recipe 提供四个源码依赖、runtime 和训练配置。

| 层 | 内容 | 使用范围 |
| --- | --- | --- |
| `common` | CANN 9.1.0 A3、Ubuntu 22.04、Python 3.12、torch 2.10.0、torch-npu 2.10.0.post4、Triton Ascend 3.2.2、Triton 3.5.0 | V4/V4.1 的公共构建基础，不是完整训练环境 |
| `v41` | Megatron-LM 基线加补丁、固定 vLLM/vLLM-Ascend、原生算子、训练与 rollout 共用的 Python 依赖 | V4/V4.1 共用源码；两个 profile 均需镜像/NPU 验收 |
| recipe | 四个源码依赖、`workspace/rollout/dsv41/runtime`、启动与初始化脚本 | 运行时克隆，在全部 Ray 节点安装/激活 |

基础标签来自 [verl 上游 Dockerfile](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3_v0.9.1)：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-ubuntu22.04-py3.12`。正式发布仍需固定目标架构 digest。CANN/ATB、编译器及驱动配套须验收。当前构建目标沿用 `ascend910_9382`，与上游示例的 `9392` 不同，需按实际芯片确认。

镜像构建不会克隆、安装verl fork 和上游 MindSpeed、MindSpeed-LLM、mbridge，也不复制 recipe runtime。三个框架统一从 `dependencies/framework-sources.json` 读取公开基线、补丁与逐文件校验值，拉取并还原源码。框架位于 `/opt/ascend-training/`，版本收据位于 `/opt/ascend-training/manifest/frameworks.json`。

## 固定框架

- Megatron-LM：0.12.1，上游 `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17` + [补丁](../../patches/megatron-lm/megatron-lm-local.patch)，来源与逐文件校验值见 [登记](../../dependencies/framework-sources.json)，许可证见 [许可证](Megatron-LM-LICENSE)。已验证基线加补丁重建参考源码；不是 NPU 验收。
- vLLM：`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。
- vLLM-Ascend：`23e286f9f4a7de13aa192c43ba894cf4e2ceabdb`。
- Transformers：训练和 rollout 统一为 5.14.1，使用同一 Python 环境；训练兼容性尚需 NPU 验收。

## 构建与使用

Dockerfile 使用 BuildKit 自带的前端，不额外拉取 `docker/dockerfile:1`。构建器需支持 `RUN --mount=type=cache` 和 heredoc；旧版构建器遇到语法错误时应升级 Docker/BuildKit，不要关闭 BuildKit。基础镜像和其他构建材料仍需联网下载。

从 recipe 根目录构建公共基础层：

```bash
docker build --platform linux/arm64 --target common \
  -f docker/v41/Dockerfile -t ascend-training-common:dev .
```

可以直接尝试构建 V4/V4.1 共用开发镜像（保留 `v41` target 和目录名以兼容原构建命令）；没有完整 runtime.lock 时，按 [固定版本与安装说明](locks/README.md) 安装：

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

初始化先核对四个源码依赖 的 commit、镜像框架版本和 Engram 校验值，再执行 `pip install -e --no-deps --no-build-isolation`，最后报告依赖检查结果（声明冲突不阻断初始化）。不自动升级依赖或现场重建镜像算子。所有 Ray 节点需使用一致镜像、recipe 和路径，再按模型 README 启动训练。设备/驱动、共享内存和网络挂载需按机器配置。

V4 使用同一镜像，运行 `bash tools/setup_sources.sh v4`、`bash tools/setup_environment.sh v4` 和 `source tools/activate.sh v4`。源码和补丁已统一，仍须分别执行 V4/V4.1 冒烟测试；源码校验不代表算子或模型兼容性验收。

## 仍待补齐

1. 构建工具的 `locks/build.lock` 已生成（14 个包，固定版本和 SHA256）；[锁文件说明](locks/README.md) 提供生成命令和源码约束。仍缺 `locks/runtime.lock`：包含完整 Python 依赖、固定版本和 wheel 哈希。统一运行锁包含 uv/构建工具及四个源码依赖 需要的 Python 包。硬件 bootstrap 目前仅固定版本，cp312 wheel 哈希仍需补齐。 完整运行锁不再是开发构建的前置条件。
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

## Framework source layout

The image installs these editable source trees at sibling paths:

```text
/opt/ascend-training/Megatron-LM
/opt/ascend-training/vllm
/opt/ascend-training/vllm-ascend
```

`workspace/rollout/dsv41/build-src` contains auxiliary builds such as MHC Premix,
not the installed vLLM source trees. Recipe forks remain in `third_party` after cloning.
Existing images retain their original layout; rebuild to adopt these paths. For older
images, explicitly set `VLLM_ROOT` and `VLLM_ASCEND_ROOT` to their installed locations.
Moving editable source trees manually does not update their Python installation records.

## Build inputs and runtime records

Build helpers, patches and dependency locks are staged under `/tmp/ascend-image-build`
and removed after the framework build and validation succeed. Downloaded source archives
are removed from the runtime filesystem; BuildKit download caches remain reusable.
`/opt/ascend-training/manifest/` retains framework commits, the Python environment report,
native dependency checksums, the dependency registry and the Megatron license/source record.
Editable framework sources and MHC runtime libraries/environment files remain in place.
This cleans the final filesystem; files copied in earlier Docker layers can still occupy
space in those layers. It does not claim to minimize the image's layered storage size.
For an older image, set `V41_IMAGE_METADATA=/opt/v41-image` when running environment checks.

## Shared framework source reconstruction

All three frameworks are siblings under `/opt/ascend-training`: `Megatron-LM`, `vllm`, and `vllm-ascend`. The build fetches fixed public commits and applies the patches in [sources](../../dependencies/framework-sources.json), checking every source file before installation. It does not require a local dependency checkout in the Docker context. The vLLM baseline is v0.23.0; vLLM-Ascend uses a pinned commit from releases/v0.23.0. These are fixed reconstruction bases, not floating latest branches.

`COMPILE_CUSTOM_KERNELS=1` explicitly compiles vLLM-Ascend. Use `docker build --progress=plain ...` for live compiler output; successful images retain `/opt/ascend-training/manifest/build-logs/vllm-ascend.log`. A failed RUN does not produce an image containing that log, so capture the host build output with `tee` when diagnosing failures. The build checks the native extension and vendor directory, and keeps source patches and receipts under `manifest/dependencies/`, `manifest/patches/`, and `manifest/tools/`.

V4.1 is backported onto the retained V4 baseline using separate `*-v41-backport.patch` files. Reconstruction verifies the V4 snapshot first, applies the backport, and then verifies the complete final inventory. Megatron-LM retains its existing baseline and patch. Transformers is pinned to `5.14.1` in the runtime overrides.

The backport adds V4.1 configuration, model/Engram loading, tokenizer and chat encoding, hybrid cache allocation, and native QLI/mHC/sparse-attention bindings. V4 retains its original model registration and default implementation. V4.1's `HcPreV41` and `SparseFlashMlaV41` native operator names are separate from V4's operators to avoid replacing their CANN ABIs.

The initial V4.1 scope is eager text RL rollout on A3: disable prefix caching and speculative decoding, and use PP=DCP=PCP=1. Tensor parallelism remains available; MoE uses the retained v0.23 sequence-parallel handling. Multimodal inputs are rejected until vision hash routing is ported. The current cache planner requires the released 40-layer topology; arbitrary layer pruning, full graph mode, and speculative decoding are not covered. Unsupported settings fail explicitly during initialization.

Source reconstruction and CPU configuration/encoding/cache checks do not validate NPU compilation or numerical correctness. Before release, build the image and run both V4 and V4.1 rollout plus training smoke tests on A3, checking output/log-probability behavior and synchronization with the actor.

CPU cache regression checks can run in a separate process with Torch, Transformers 5.14.1 and pytest installed:

```bash
python docker/v41/tests/check_v41_cache_cpu.py --framework-root /opt/ascend-training
python docker/v41/tests/check_v41_frontend_cpu.py --framework-root /opt/ascend-training
```

This check loads the real source cache classes with device-facing imports stubbed. It does not load or test the native kernels.
