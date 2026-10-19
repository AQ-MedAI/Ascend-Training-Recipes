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

For checkpoint crash diagnosis only, set `SAVE_FREQ=-1` to skip saving while
checking the two-step training loop. The default remains `SAVE_FREQ=1`. A run
without saving does not validate checkpoint save/resume.

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

## Diagnosing serialization crashes

Disabling checkpoint saves does not disable serialization used by Ray batch
transfers. The default DataProto path uses `torch.save` with an in-memory buffer.
For comparison, run the RL smoke entry with
`VERL_DATAPROTO_SERIALIZATION_METHOD=numpy SAVE_FREQ=-1`. This selects verl's
existing NumPy transport for DataProto; it is not a global change to torch.save,
TensorDict serialization, model-weight transport or checkpoint formats. The
NumPy path requires CPU tensors, as provided by the usual worker output path.

The launcher passes the selected method and `PYTHONFAULTHANDLER` (default `1`)
to Ray workers explicitly. Synchronize the recipe and restart the affected job's
workers before comparing runs. A crash persisting with NumPy needs the Python
faulthandler stack or a core backtrace to identify the actual caller. Do not
interpret the CRC32 frame alone as proof that the CRC implementation is faulty.

## Runtime policy alignment

RL uses the cropped MOPD memory policy: `optimizer_cpu_offload=True`,
`optimizer_offload_fraction=1`, `overlap_cpu_optimizer_d2h_h2d=False` and
`use_precision_aware_optimizer=False`. These are Core optimizer options, distinct
from verl's phase-level `megatron.optimizer_offload`. Override them through
`OPTIMIZER_CPU_OFFLOAD`, `OPTIMIZER_OFFLOAD_FRACTION`,
`OVERLAP_CPU_OPTIMIZER_D2H_H2D` and `USE_PRECISION_AWARE_OPTIMIZER`.
The selected Core/MindSpeed CPU optimizer path still requires NPU verification.

`SEQUENCE_PARALLEL` defaults to `True`, matching MoE training with TP greater
than one. For TP=1, set it to `False` if required by the selected Core config.
Allocator, HCCL and thread settings are passed to all Ray workers explicitly.
RL retains HCCL buffer defaults 200/200 MiB; the MOPD smoke profile retains its
512/256 MiB policy. Configure them for the actual topology rather than assuming
that a multi-node placement has the same memory cost as a one-node smoke test.

The full-vocabulary KL, TQ artifact transport/cleanup, hidden export, teacher
sleep controller and rack placement implementations are retained in the K26
source patch. Source equivalence does not validate a different Core, Bridge or
vLLM runtime. Two-layer checkpoints and public-runtime compatibility patches
remain intentional adaptations. NumPy DataProto serialization remains opt-in
for diagnosis; the default protocol matches the upstream/reference implementation.

## Isolating checkpoint serialization failures

`SAVE_FREQ=-1` succeeding while the default save fails isolates a save-dependent
failure. `CHECKPOINT_SAVE_CONTENTS='[model]'` saves only the model;
`'[optimizer,extra]'` isolates optimizer/RNG/scheduler state. The default remains
`'[model,optimizer,extra]'`. `CHECKPOINT_LOAD_CONTENTS` remains the full resume
set by default; partial diagnostic checkpoints cannot restore full training state.
RL explicitly uses synchronous checkpoint saves.

For missing Python stacks in a native serialization crash, apply the opt-in
`patches/verl/k26-checkpoint-trace.patch` after the normal K26 source patches and
restart workers. `K26_CHECKPOINT_TRACE=1` then logs each torch.save call made during
save_checkpoint, including its Python caller and bounded tensor shape/dtype/device
metadata. It logs no tensor values and leaves CRC32 and the payload unchanged.
The trace requires synchronous save and restores torch.save on return/error.
It is diagnostic instrumentation, not a checkpoint correctness fix.

## Synchronous checkpoint writer on Ascend

RL and the MOPD smoke entry default to `ASCEND_CHECKPOINT_WRITER=inprocess`,
with `checkpoint.async_save=False`. Apply the additional
[Core checkpoint patch](../../patches/megatron-lm/k26/README.md#in-process-synchronous-checkpoint-writer)
and restart workers. This mode keeps bucket serialization in the worker process
rather than forking a child that reads offloaded host memory. The native serializer,
CRC32, fsync and metadata/result protocol are retained. The exact cause of the
observed child-process memory fault remains unproven; save/resume needs NPU testing.

The startup checker requires the patched writer when this mode is enabled.
`ASCEND_CHECKPOINT_WRITER=multiprocess` is available only for comparison with the
previous Core path. A successful save should print `[checkpoint-writer] ... mode=inprocess`.
CPU tests cover FP32/BF16 round trips, multiple buckets and write-error propagation.


For shared storage that rejects native safetensors file operations during HF
export, apply [the Bridge sequential-write patch](../../patches/megatron-bridge/README.md#shared-filesystem-safetensors-export).
This matches the V4/V4.1 mbridge policy and keeps the configured output directory.
It requires host RAM for an additional serialized shard; complete save/resume
must be checked on the target filesystem.


For asynchronous NPU errors (for example `aclnnMatmul` reported later during
PP shape communication), set `ASCEND_LAUNCH_BLOCKING=1` for a diagnostic run.
RL and MOPD smoke entries pass this variable to Ray workers explicitly. Keep the
same topology and capture the first operator error plus its full Python stack.
The default is `0`; synchronous launch is a diagnostic setting with a performance
cost, not a fix for the operator failure.
