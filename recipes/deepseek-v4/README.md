# DeepSeek-V4-Flash

## 脚本

| 文件 | 用途 |
| --- | --- |
| [dsv4_flash_rl.sh](dsv4_flash_rl.sh) | 完整模型 GRPO 基础入口，也是完整 MOPD 的委托脚本 |
| [dsv4_flash_mopd.sh](dsv4_flash_mopd.sh) | 完整 MOPD：默认学生 8 节点 × 16 NPU、教师 4 节点 × 16 NPU |
| [dsv4_flash_rl_8npu_cutlayer.sh](dsv4_flash_rl_8npu_cutlayer.sh) | 8 NPU 裁层 RL，委托同目录裁层脚本 |
| [dsv4_flash_mopd_16npu_cutlayer.sh](dsv4_flash_mopd_16npu_cutlayer.sh) | 裁层训练实现，含模型准备和蒸馏配置 |
| [prepare_dsv4_flash_bf16_cut_model.py](prepare_dsv4_flash_bf16_cut_model.py) | 准备裁层 BF16 checkpoint，需要 torch 和 safetensors |
| [inspect_dsv4_checkpoint_metadata.py](inspect_dsv4_checkpoint_metadata.py) / [inspect_dsv4_flash_shapes.py](inspect_dsv4_flash_shapes.py) | checkpoint 元数据和形状检查 |
| [diagnose_dsv4_flash_vllm_prefill.sh](diagnose_dsv4_flash_vllm_prefill.sh) / [Python 实现](diagnose_dsv4_flash_vllm_prefill.py) | vLLM prefill 诊断 |

## 环境和运行

需要匹配的 verl、MindSpeed、MindSpeed-LLM、Megatron-LM、mbridge、vLLM、vLLM-Ascend；使用包含现有适配的环境，不能假定任意上游最新版兼容。完整 MOPD 脚本显式检查 `TransferQueue==0.1.7`。

先启动所需 Ray 集群，确认模型、parquet、reward 文件、输出目录和资源布局。完整 MOPD 包装脚本固定拓扑并拒绝命令行覆盖；模型、数据、输出路径及 `PYTHON_BIN` 可通过环境变量设置。基础 GRPO 和裁层脚本的环境变量以实际赋值为准。

从仓库根目录执行以下所选入口（不是顺序执行全部命令）：

```bash
# 完整模型 GRPO
bash recipes/deepseek-v4/dsv4_flash_rl.sh
# 完整模型 MOPD
bash recipes/deepseek-v4/dsv4_flash_mopd.sh
# 裁层 RL；可能先写出模型文件
bash recipes/deepseek-v4/dsv4_flash_rl_8npu_cutlayer.sh
```

保持这些脚本在同一目录，包装脚本使用 `SCRIPT_DIR` 查找被调用脚本和模型准备工具。诊断脚本运行方式及模型准备参数请查看文件；Python 工具的 `--help` 需要相应导入依赖。

## 状态

本目录提供 V4 完整模型和裁层配置。NPU 端到端验收待完成；CPU 检查范围见 [验证说明](../../docs/VALIDATION.md)，来源见 [来源说明](../../docs/PROVENANCE.md)。

## 依赖布局

V4 入口通过 `tools/v4_environment.sh` 初始化七个预安装依赖，默认都在镜像的
`/opt/ascend-training/third_party/` 下。V4/V4.1 共用源码，无需初始化 recipe 子模块。
开发时可通过各 `*_ROOT` 显式覆盖；例如改变镜像依赖根目录：

```bash
export DEPENDENCY_ROOT=/path/to/prepared-dependencies
export VLLM_ROOT="$DEPENDENCY_ROOT/third_party/vllm"
export VLLM_ASCEND_ROOT="$DEPENDENCY_ROOT/third_party/vllm-ascend"
source tools/activate.sh v4
```

`activate.sh v4` 只设置运行环境，不执行下载、安装或 Git 切换。
在每个节点保持源码和动态库路径一致，并从新的终端环境启动，避免沿用 V4.1 的专用配置。
启动脚本会将源码路径、动态库路径传给 Ray；V4 不使用 V4.1 专用 QLI 初始化脚本。

## MTP training

V4 MTP training is opt-in. Prepare the shared source patches with
`bash tools/setup_sources.sh v4`, and use the same sources on every node.
Both V4 and V4.1 profiles include the guarded V4 MTP changes; K2.6 uses its own
verl source profile. Rebuild the image to include these changes in a distributable runtime.

```bash
# Full-model RL; the checkpoint must retain official mtp.<index> weights.
MTP_ENABLE=True bash recipes/deepseek-v4/dsv4_flash_rl.sh
# Four-layer RL; builds a separate BF16 crop retaining MTP weights.
FULL_MODEL_PATH=/path/to/DeepSeek-V4-Flash MTP_ENABLE=True \
  bash recipes/deepseek-v4/dsv4_flash_rl_8npu_cutlayer.sh
# Four-layer MOPD with MTP on the student.
FULL_MODEL_PATH=/path/to/DeepSeek-V4-Flash MTP_ENABLE=True \
  bash recipes/deepseek-v4/dsv4_flash_mopd_16npu_cutlayer.sh
```

`MTP_ENABLE_TRAIN` defaults to `MTP_ENABLE`; `MTP_LOSS_SCALING_FACTOR` defaults to
`0.1`. Training requires MTP parameter loading enabled. Rollout MTP remains
disabled, so this change does not enable speculative decoding.
The engine requires mHC, static CP, padded input, the standard pipeline schedule
and `moe_fb_overlap=False`.

The crop helper accepts `--keep-mtp`. It retains all official MTP weights,
dequantizes them to BF16 and preserves their own compression-ratio settings.
The default four-layer output uses a separate `_mtp` suffix when MTP is retained;
existing crops without MTP are not reused for MTP training. `CUT_KEEP_MTP`
defaults to `MTP_ENABLE`. Prebuilt model paths must contain the retained MTP weights.

CPU tests cover mapping/export, TP projection, mHC-stream gradients, CP token
boundaries, auxiliary-loss backpropagation and real safetensors cropping. NPU
training, checkpoint resume and numerical comparisons still require runtime validation.

## DeepSeek V4 Flash 0731 (backbone RL/MOPD)

The official 0731 index has the same non-`mtp.*` tensor names as the earlier V4
Flash checkpoint. No backbone tensor renaming is needed. Its `mtp.0/1/2` blocks
belong to the DSpark prediction module: `main_proj/main_norm`, Markov and
confidence heads do not map to legacy `e_proj/h_proj/enorm/hnorm` parameters.
This recipe supports backbone training only; DSpark head training/speculative
decoding is not enabled. Legacy V4 MTP training retains its existing behavior.

Use the prepared BF16 0731 checkpoint with all three model MTP switches disabled:

```bash
MODEL_PATH=/path/to/deepseek-v4-flash-0731-bf16 \
MTP_ENABLE=False MTP_ENABLE_TRAIN=False \
bash recipes/deepseek-v4/dsv4_flash_rl.sh
```

The same settings apply to `dsv4_flash_mopd.sh` (configure its teacher paths as
usual). The scripts already set `model.mtp.enable_rollout=False`. The in-memory
model config and vLLM HF overrides disable prediction heads and retain only the
backbone compression ratios. A model-only HF export advertises no MTP/DSpark
heads, because these heads were not trained or exported. The source checkpoint
is never edited. An explicit request to enable a DSpark head fails early.

For a smoke crop, omit `--keep-mtp`. The helper removes all `mtp.*` tensors and
DSpark settings, clips compression ratios to the retained backbone, and rejects
stale crops with the old 46-entry ratio list. `--keep-mtp` is supported for legacy
V4 checkpoints, but rejected for 0731/DSpark before changing the output directory.

Source contract: [official index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/model.safetensors.index.json)
and [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json).
Run `python -m unittest discover -s tests -p 'test_v4_0731.py'` for CPU checks.
These checks verify names/config and tiny safetensors conversion; they do not
verify full-checkpoint tensor shapes, NPU execution or rollout numerical parity.
