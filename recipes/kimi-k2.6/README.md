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
Distillation profiles retain their existing hyperparameters. Configure their model, data,
reward, log and shared-cache paths before use; some profiles still set paths directly.

## Two-layer RL smoke test

Run `bash tools/setup_sources.sh k26` to prepare the pinned upstream verl and its K26 patches. This selects only verl; provide a compatible K26 runtime and its other dependencies on every node.
The RL entry defaults to `third_party/verl`; override `VERL_ROOT` when needed.
Both RL and MOPD default to `third_party/MindSpeed`. Set `MINDSPEED_ROOT` to
an existing, K26-adapted MindSpeed checkout when using another location:

```bash
export MINDSPEED_ROOT=/workspace-verl/Ascend-Training-Recipes/third_party/MindSpeed
```

`runtime_env.sh` puts this checkout first in `PYTHONPATH`, validates the selected
package before importing its adapter, and prints its resolved path. The launchers
propagate `MINDSPEED_ROOT` and `PYTHONPATH` to Ray workers. Use the same prepared
source paths on all nodes and restart the training processes after changing them.
This selects sources; it does not apply patches.

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

This profile defaults to TP4/PP1/CP2/EP8/ETP1, a single TP8 rollout replica, eight
prompts and two samples per prompt. Actor parallelism can be overridden through
`ACTOR_TP`, `ACTOR_PP`, `ACTOR_CP`, `ACTOR_EP`, and `ACTOR_ETP`. Smoke context limits can be overridden through `MAX_PROMPT_LENGTH`,
`MAX_RESPONSE_LENGTH`, `MAX_MODEL_LEN` and `ROLLOUT_MAX_NUM_BATCHED_TOKENS`.
Other smoke settings remain fixed and Hydra overrides are rejected. It disables distillation and initial
validation, and requests a checkpoint every training step. Success requires two
training steps, including rollout, backward/optimizer execution and subsequent
rollout weight updates. This RL topology has completed runtime testing; other
topologies and MOPD require separate validation.

The prompt limit is 8192 tokens and the response limit is one token. Prompts use
left truncation and overlong-prompt filtering by default; override these policies
with `DATA_TRUNCATION` and `FILTER_OVERLONG_PROMPTS`.

`pipeline_layout.py` reads the actual text decoder depth before framework startup.
The default PP=1 keeps both layers on one pipeline stage. With PP=2, a two-layer
checkpoint uses an even 1+1 split; no full-model endpoint
layer counts are injected. The established 61-layer/PP10 profile retains first=3,
last=2. Explicit `FIRST_PIPELINE_LAYERS` / `LAST_PIPELINE_LAYERS` values are validated;
use `none` to leave an endpoint unspecified. Remove stale endpoint overrides when
switching from a full model to a cut model.

The default actor decomposition is equivalent to the following explicit settings:

```bash
ACTOR_TP=4 ACTOR_PP=1 ACTOR_CP=2 ACTOR_EP=8 ACTOR_ETP=1 \
bash recipes/kimi-k2.6/kimi_k26_rl_8npu_cutlayer.sh
```

The smoke token budgets default to the full prompt-plus-response length (8193),
and can be overridden through `ACTOR_MAX_TOKEN_LEN_PER_GPU` and
`LOG_PROB_MAX_TOKEN_LEN_PER_GPU`. These budgets cover the sequence before CP partitioning.

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

`kimi_k26_mopd_16npu_cutlayer.sh` prepares the two-layer model, then calls
`kimi_k26_mopd.sh` with smoke settings. It uses the same
student defaults as the validated RL smoke entry:
`ACTOR_TP=4 ACTOR_PP=1 ACTOR_CP=2 ACTOR_EP=8 ACTOR_ETP=1`.
It keeps both decoder layers on one pipeline stage. Eight student NPUs and two
TP4 teacher replicas require one 16-NPU node. By default both teachers reuse
the cropped student checkpoint; override `MATH_TEACHER_MODEL` and
`GSM8K_TEACHER_MODEL` with models sharing its tokenizer and vocabulary.

The launcher creates or validates the two-layer checkpoint using `MODEL_PATH`,
`CUT_MODEL_PATH` and `FULL_MODEL_PATH`. Data files default to
`${DATA_ROOT:-/workspace-verl/data}/{math,gsm8k}/{train,test}.parquet`.

| Dataset | Required `data_source` / teacher key | Task reward |
| --- | --- | --- |
| MATH | `DigitalLearningGmbH/MATH-lighteval` | DAPO boxed-answer matching: correct `1`, incorrect `-1` |
| GSM8K | `openai/gsm8k` | GSM8K strict `####` matching: correct `1`, otherwise `0` |

[multi_reward.py](multi_reward.py) routes by each sample's `data_source` and reads
its final answer from `reward_model.ground_truth`. It rejects unknown sources.
Override `REWARD_FN_PATH` to use a different reward dispatcher. For answers using
`Answer: ...` instead of boxed notation, append
`+reward.custom_reward_function.reward_kwargs.math_strict_box_verify=False`.
The full-model MOPD launcher uses the same datasets, teacher keys and reward dispatcher.

```bash
DATA_ROOT=/workspace-verl/data \
MODEL_PATH=/path/to/Kimi-K2.6-2layers \
OUTPUT_DIR=/path/to/output LOG_DIR=/path/to/logs \
bash recipes/kimi-k2.6/kimi_k26_mopd_16npu_cutlayer.sh
```

Override individual dataset paths with `MATH_TRAIN_FILE`, `MATH_VAL_FILE`,
`GSM8K_TRAIN_FILE` and `GSM8K_VAL_FILE`. Model, data and reward paths must be
readable by all workers. Dataset identifiers must match the table exactly.

The student uses eight prompts, two samples per prompt, 8192 prompt tokens and
one response token, with eager rollout, left truncation and prompt filtering.
Actor/log-prob token budgets default to 8193; rollout model length is 8196 and
the batched-token budget is 4096. CPU optimizer offload, learning rate, gradient
offload, recomputation and synchronous checkpoint settings are configured by
`kimi_k26_mopd.sh` to match the validated RL smoke profile. The run lasts two steps, saves each step and skips validation.
These short responses exercise execution and do not validate answer quality.

`ACTOR_*` overrides and legacy `TP`, `PP`, `CP`, `EP`, `ETP` remain supported.
Token budgets accept `ACTOR_MAX_TOKEN_LEN_PER_GPU` and
`LOG_PROB_MAX_TOKEN_LEN_PER_GPU`; the actor budget also accepts the legacy
`PPO_MAX_TOKEN_LEN_PER_GPU`. Teacher settings and distillation loss have their own
`TEACHER_*`, `DISTILLATION_*`, `FULL_VOCAB_CHUNK_TOKENS` and
`USE_POLICY_GRADIENT` overrides. Default full-vocabulary distillation remains
`forward_kl_full_vocab` with coefficient `0.5` and task rewards enabled.
MOPD requires separate NPU validation; the completed RL run covers the student
configuration only. The full-model MOPD entry retains its separate scale and training defaults.

## Full-model MOPD

`kimi_k26_mopd.sh` uses the same `DATA_ROOT`, per-dataset file overrides,
`multi_reward.py` dispatcher and exact `data_source` values shown above.
`MODEL_PATH` selects the full student checkpoint; `MATH_TEACHER_MODEL` and
`GSM8K_TEACHER_MODEL` override the two teachers independently. Both default to
the student checkpoint and must share its tokenizer and vocabulary.

```bash
DATA_ROOT=/workspace-verl/data \
MODEL_PATH=/path/to/Kimi-K2.6_bf16 \
MATH_TEACHER_MODEL=/path/to/math-teacher \
GSM8K_TEACHER_MODEL=/path/to/gsm8k-teacher \
OUTPUT_DIR=/path/to/output LOG_DIR=/path/to/logs \
bash recipes/kimi-k2.6/kimi_k26_mopd.sh
```

The full profile retains its own scale and training defaults: 40 student nodes,
16 NPUs per node, TP8/PP10/CP8/EP64/ETP1, 64K prompt tokens and 8K response tokens.
Actor parallelism accepts `ACTOR_*` (or legacy `TP`/`PP`/`CP`/`EP`/`ETP`), and
`GPUS_PER_NODE` also accepts the legacy `NGPUS_PER_NODE`. Pipeline layer splits
are resolved from the checkpoint depth: two layers with PP1 use no endpoint
counts, while 61 layers with PP10 use first=3 and last=2.
It defaults to reverse-KL full-vocabulary distillation; the smoke profile uses
forward KL. Both use task rewards, coefficient `0.5`, the K26 runtime checker and
synchronous in-process checkpoint writing. Configure the Ray cluster and dedicated
teacher pool before launching; the one-node smoke placement does not apply here.

Both MOPD entries use the RL algorithm defaults: GRPO, `gamma=1`, `lam=1`,
`norm_adv_by_std_in_grpo=True`, `use_kl_in_reward=False` and
`kl_ctrl.kl_coef=0`. Reference-policy KL is not added to task rewards.
Teacher distillation is configured separately through `distillation.distillation_loss`.

## Image preparation

See [K26 image adaptation](../../docker/k26/README.md). The public Core/Bridge
baselines are pinned, but the new image is currently a source-only candidate.
RL and both MOPD launchers validate the selected Core/Bridge paths
through `runtime_env.sh`; set `MEGATRON_ROOT` and `MEGATRON_BRIDGE_ROOT` when
using an existing compatible K26 environment.

For the external K2.6 Megatron baseline, apply the [optional TE compatibility patch](../../patches/megatron-lm/k26/README.md) to `/opt/ascend-training/Megatron-LM` before sourcing `runtime_env.sh`. The patch enables import with an incomplete TE shim for non-FP8 execution; it does not provide FP8 support.

## Runtime policy

RL defaults to `optimizer_cpu_offload=True`, `optimizer_offload_fraction=1`,
`overlap_cpu_optimizer_d2h_h2d=False` and `use_precision_aware_optimizer=False`.
These Core optimizer options complement verl's phase-level optimizer offload.
Override them through `OPTIMIZER_CPU_OFFLOAD`, `OPTIMIZER_OFFLOAD_FRACTION`,
`OVERLAP_CPU_OPTIMIZER_D2H_H2D` and `USE_PRECISION_AWARE_OPTIMIZER`.

`SEQUENCE_PARALLEL` defaults to `True`. For TP=1, set it to `False` if required
by the selected Core configuration. Allocator, HCCL and thread settings are passed
to Ray workers explicitly. RL and the MOPD smoke profile share HCCL buffer
defaults of 200/200 MiB. Peak memory depends on topology and sequence length.

HCCL host/NPU socket ranges default to `auto` in both RL and MOPD entries.
Explicit `HCCL_HOST_SOCKET_PORT_RANGE`, `HCCL_NPU_SOCKET_PORT_RANGE` and
`HCCL_CONNECT_TIMEOUT` overrides are passed to Ray workers. NPU allocator and
HCCL operation-mode overrides also reach the workers without hard-coded replacements.
Actor/log-prob token budgets cover the complete sequence before CP splitting,
including in full-model profiles. CPU optimizer transfer overlap defaults to
`False`; enable it explicitly only for a separately validated environment.

MOPD accepts explicit `TRAIN_FILES` and `VAL_FILES` lists; when omitted, it builds
them from the four per-dataset file variables. Set `ROLLOUT_EP` to an empty string
to leave the rollout backend's expert-parallel default unspecified.

## Checkpoints

The RL smoke test defaults to `SAVE_FREQ=1`; full-model RL defaults to `10`.
`CHECKPOINT_SAVE_CONTENTS` and `CHECKPOINT_LOAD_CONTENTS` both default to
`[model,optimizer,extra]`. RL uses synchronous saves (`checkpoint.async_save=False`).
A successful training/save run does not establish resume correctness; validate
resume separately before a production run.

RL and both MOPD entries default to `ASCEND_CHECKPOINT_WRITER=inprocess`.
Apply the [Core checkpoint patch](../../patches/megatron-lm/k26/README.md#in-process-synchronous-checkpoint-writer)
before starting workers. It serializes checkpoint buckets in the worker process,
retaining the native payload, CRC32, fsync and metadata protocol. The startup
checker verifies that this writer is available.

For HF export on shared storage, apply the
[Bridge sequential-write patch](../../patches/megatron-bridge/README.md#shared-filesystem-safetensors-export).
It writes safetensors through ordinary sequential file writes and keeps the
configured output directory. Allow host RAM for an additional serialized shard.
