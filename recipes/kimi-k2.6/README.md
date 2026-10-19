# Kimi K2.6

## Entrypoints

| Script | Purpose |
| --- | --- |
| [kimi_k26_rl.sh](kimi_k26_rl.sh) | Full-model GRPO; environment and Hydra overrides supported |
| [kimi_k26_rl_8npu_cutlayer.sh](kimi_k26_rl_8npu_cutlayer.sh) | Two-layer RL smoke test, one node with eight NPUs, two steps |
| [kimi_k26_mopd.sh](kimi_k26_mopd.sh) | Full-model, two-teacher full-vocabulary MOPD; reverse KL by default |
| [kimi_k26_mopd_16npu_cutlayer.sh](kimi_k26_mopd_16npu_cutlayer.sh) | Two-layer, two-teacher full-vocabulary MOPD smoke test on one 16-NPU node |
| [kimi_k26_opd.sh](kimi_k26_opd.sh) | Full-model, single-teacher forward-KL top-k OPD |
| [kimi_k26_mopd_reverse_kl.sh](kimi_k26_mopd_reverse_kl.sh) | Alternative full-model two-teacher reverse-KL profile |
| [kimi_k26_mopd_shared_teacher.sh](kimi_k26_mopd_shared_teacher.sh) | Shared teacher pool; at most one awake teacher |

Names follow the V4 pattern: model prefix, training mode, and resource/cut-layer suffix.
The former `test_kimi_full_rl.sh` enabled distillation and is now named `kimi_k26_opd.sh`.
Distillation profiles retain their existing hyperparameters. Configure their model, data,
reward, log and shared-cache paths before use; some profiles still set paths directly.

## Two-layer RL smoke test

Run `bash tools/setup_sources.sh k26` to prepare the pinned upstream verl with `patches/verl/k26.patch`. This selects only verl; provide a compatible K26 runtime and its other dependencies on every node.
The RL entry defaults to `third_party/verl`; override `VERL_ROOT` when needed.
`vanilla_mbridge=False` selects Megatron-Bridge, not AQ mbridge. Do not assume that the
V4/V4.1 MindSpeed, Megatron or native operator combination is compatible with K26.

The smoke entry automatically creates a two-layer checkpoint when `MODEL_PATH` is
missing. Set `FULL_MODEL_PATH` to an unquantized BF16/FP16/FP32 Kimi K2.6 checkpoint.
The helper keeps decoder layers 0–1, embeddings, final norms, output heads, vision
weights and tokenizer/custom-code assets; MTP weights are excluded. It handles sharded
or single-file safetensors and nested `text_config`. Quantized sources must be converted
beforehand. A temporary sibling directory is published only after conversion succeeds.

The default output is `${TMPDIR:-/tmp}/kimi_k26_bf16_2layers_model`. Existing
four-layer directories are not modified. If `MODEL_PATH` or `CUT_MODEL_PATH` is
still set to a four-layer output, unset it or choose a new two-layer path.
Only decoder depth is reduced; expert counts, hidden sizes and vocabulary remain
unchanged, and lower NPU memory use still requires runtime validation.

Existing outputs are validated and reused, even without `FULL_MODEL_PATH`. Invalid or
incomplete existing directories cause an error and are never automatically overwritten.
Use a shared `MODEL_PATH` visible on every node; the default is a local temporary path.
Conversion needs CPU memory for one selected source shard and disk space for the retained
weights. Source checkpoints are not modified.

From the repository root:

```bash
FULL_MODEL_PATH=/path/to/Kimi-K2.6_bf16 \
MODEL_PATH=/path/to/Kimi-K2.6-2layers \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/valid.parquet \
REWARD_FN_PATH=/path/to/reward.py \
OUTPUT_DIR=/path/to/output \
LOG_DIR=/path/to/logs \
bash recipes/kimi-k2.6/kimi_k26_rl_8npu_cutlayer.sh
```

This profile defaults to TP2/PP2/CP2/EP4/ETP1, a single TP8 rollout replica, eight
prompts and two samples per prompt. Actor parallelism can be overridden through
`ACTOR_TP`, `ACTOR_PP`, `ACTOR_CP`, `ACTOR_EP`, and `ACTOR_ETP`. Other smoke settings
remain fixed and Hydra overrides are rejected. It disables distillation and initial
validation, and requests a checkpoint every training step. Success requires two
training steps, including rollout, backward/optimizer execution and subsequent
rollout weight updates. NPU execution and peak memory remain to be validated for
each topology.

`pipeline_layout.py` reads the actual text decoder depth before framework startup.
A two-layer checkpoint with PP=2 uses an even 1+1 split; no full-model endpoint
layer counts are injected. The established 61-layer/PP10 profile retains first=3,
last=2. Explicit `FIRST_PIPELINE_LAYERS` / `LAST_PIPELINE_LAYERS` values are validated;
use `none` to leave an endpoint unspecified. Remove stale endpoint overrides when
switching from a full model to a cut model.

The default actor decomposition is equivalent to the following explicit settings;
CP/PP communication still requires runtime validation:

```bash
ACTOR_TP=2 ACTOR_PP=2 ACTOR_CP=2 ACTOR_EP=4 ACTOR_ETP=1 \
bash recipes/kimi-k2.6/kimi_k26_rl_8npu_cutlayer.sh
```

The smoke token budgets default to the full prompt-plus-response length (8193),
and can be overridden through `ACTOR_MAX_TOKEN_LEN_PER_GPU` and
`LOG_PROB_MAX_TOKEN_LEN_PER_GPU`. They no longer use a hard-coded division by two.

For full-model RL, use `kimi_k26_rl.sh`. Multi-node jobs require a running Ray cluster
and identical source, model, data and runtime paths on every node. Teacher models used
for full-vocabulary KL must have compatible tokenizers and vocabularies. Shared teacher
profiles additionally depend on the corresponding scheduling support in the K26 patch.

See [dependencies](../../docs/DEPENDENCIES.md) and [validation](../../docs/VALIDATION.md).

The checkpoint can also be prepared independently:

```bash
python3 recipes/kimi-k2.6/prepare_kimi_k26_cut_model.py \
  --input-path /path/to/Kimi-K2.6_bf16 \
  --output-path /path/to/Kimi-K2.6-2layers --num-layers 2
```

The helper requires `torch` and `safetensors` in `PYTHON_BIN`'s environment.

## Two-layer MOPD smoke test

`kimi_k26_mopd_16npu_cutlayer.sh` uses the same student defaults:
`ACTOR_TP=2 ACTOR_PP=2 ACTOR_CP=2 ACTOR_EP=4 ACTOR_ETP=1`.
These settings cover tensor, pipeline, context and expert parallelism; expert tensor
parallelism remains at one. A two-layer student uses one layer per pipeline stage.
The student uses eight NPUs. Two TP4 teacher replicas use another eight NPUs,
so this test still requires 16 NPUs in total; rollout remains TP8.

The launcher prepares and validates the same two-layer checkpoint as RL, using
`MODEL_PATH` / `CUT_MODEL_PATH` and `FULL_MODEL_PATH`; both teachers reuse it.
Configure the data, reward and cache paths in the launcher before running:

```bash
bash recipes/kimi-k2.6/kimi_k26_mopd_16npu_cutlayer.sh
```

Override student parallelism with `ACTOR_*`. Legacy `TP`, `PP`, `CP`, `EP` and `ETP`
variables remain supported, with `ACTOR_*` taking precedence. Actor and log-prob
budgets cover the complete sequence before CP partitioning and accept
`ACTOR_MAX_TOKEN_LEN_PER_GPU` and `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` overrides.
The actor budget also accepts the legacy `PPO_MAX_TOKEN_LEN_PER_GPU` variable.
These defaults do not establish that the topology has passed NPU validation.
Full-model RL and MOPD defaults are unchanged.

## Image preparation

See [K26 image adaptation](../../docker/k26/README.md). The public Core/Bridge
baselines are pinned, but the new image is currently a source-only candidate.
RL and the 16-NPU MOPD smoke launcher validate the selected Core/Bridge paths
through `runtime_env.sh`; set `MEGATRON_ROOT` and `MEGATRON_BRIDGE_ROOT` when
using an existing compatible K26 environment.

For the external K2.6 Megatron baseline, apply the [optional TE compatibility patch](../../patches/megatron-lm/k26/README.md) to `/opt/ascend-training/Megatron-LM` before sourcing `runtime_env.sh`. The patch enables import with an incomplete TE shim for non-FP8 execution; it does not provide FP8 support.
