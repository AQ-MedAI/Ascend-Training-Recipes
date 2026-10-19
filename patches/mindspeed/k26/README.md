# K26 MindSpeed compatibility

These incremental patches target the MindSpeed source prepared by this recipe:
upstream commit `c356513e331e55d53b799cfde51c7b23901d23dc` plus the shared
MindSpeed patch series. They are specific to Megatron-LM
`96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472`; do not add them to the shared
V4/V4.1 image profile.

- `ascend-lazy-moe-imports.patch` retains the existing MoE package exports but
  resolves each export only when requested. Importing `WeightGradStore` no longer
  loads all legacy dispatch and pipeline-overlap implementations. Requested
  unsupported modules still raise their original errors.
- `ascend-gmm-api.patch` imports grouped experts independently of the legacy
  dispatcher adaptor collection, forwards explicit `pg_collection` to Core,
  and registers legacy dispatcher preprocessing only when that module exists.
  Selecting `alltoall_seq` without the legacy module raises an explicit error;
  the current K26 recipe selects `alltoall`.

Apply once to the container's prepared MindSpeed checkout:

```bash
cd /workspace-verl/Ascend-Training-Recipes
for patch in ascend-lazy-moe-imports.patch ascend-gmm-api.patch; do
    git -C third_party/MindSpeed apply --check "$PWD/patches/mindspeed/k26/$patch" || break
    git -C third_party/MindSpeed apply "$PWD/patches/mindspeed/k26/$patch" || break
done
```

Also apply the [K26 Core patches](../../megatron-lm/k26/README.md).
Restart the failed training task in fresh processes. `runtime_env.sh` checks
imports; it does not install these patches. The current source setup receipt
covers the shared MindSpeed tree, so these incremental K26 changes remain local
modifications; a shared-profile source update must not overwrite them.

## Review and validation scope

CPU tests cover lazy exports and aliases, routing values/gradients, grouped-expert
process-group forwarding, and dispatcher patch registration. They do not run NPU
kernels, distributed experts, optimizer steps, or checkpoint save/resume.

A static review of Core imports in MindSpeed and Megatron-Bridge also identified
interfaces needing separate validation before enabling these features:

| Feature | Difference | Scope |
| --- | --- | --- |
| Legacy sequence dispatcher | Core removed `legacy_a2a_token_dispatcher` | Do not select `alltoall_seq`; no replacement communication algorithm is supplied here. |
| Pipeline overlap / DualPipe / multi-parameter pipelines | Older callers import P2P free functions now exposed through `P2PCommunicator` | Not exercised by the PP=1 smoke configuration; higher pipeline modes need adaptation. |
| Rank-owned optimizer helper | A MindSpeed optimizer implementation imports `_get_rank_owned_optimizer`, absent in this Core | Its function body needs adaptation if selected; this is not evidence that the default optimizer path invokes it. |
| MindSpore / custom FSDP / old tokenizer | Removed or moved Core symbols | These branches are outside the current PyTorch DDP smoke configuration. |
| Bridge optional integrations | `HyperCommGrid` and compiled dataset helpers are optional imports | Their absence alone does not establish a Kimi model failure. |

No required missing direct Core import was found in the pinned Bridge Python
sources by the static scan. This is an import-level finding, not proof of API,
numerical, or distributed compatibility. Full K26 adaptation remains subject to
container validation: import preflight, 4-layer forward/backward and optimizer
step, then checkpoint save/resume. Do not infer compatibility with V4/V4.1 Core
from sharing the same filesystem layout.

ascend-lazy-moe-imports.patch SHA-256: `da52b82b941ccb020052aa4f444e4565dbf70492f0ab9d2f52a913f8dcadc8e9`.

ascend-gmm-api.patch SHA-256: `4a116eb3080157063fc557f09ab542226da51037ada3c8ea9bc96efbd8e83e6a`.



## Optional multimodal distributed imports

`ascend-lazy-dist-train.patch` removes eager loading of the four implementation
modules in `mindspeed.core.multi_modal.dist_train`. MindSpeed's argument validation
reads `dist_train_config.get_all_config` even when multimodal distributed training
is disabled. Previously that configuration import also loaded
`inner_data_parallel`, `dist_parallel_state`, and legacy pipeline schedules.
The latter implementations require Core APIs removed from the K26 baseline.

This increment keeps the existing package exports available on demand and caches
successful imports. A configuration-only import does not load communication
implementations. Errors in explicitly requested implementations still propagate;
this patch does not implement multimodal distributed training on the K26 Core.
Do not alias `create_hierarchical_parallel_groups` to `create_hierarchical_groups`:
their interfaces and return values need an actual adaptation if this feature is
used in the future.

Apply while retaining the earlier K26 MindSpeed patches:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C third_party/MindSpeed apply --check "$PWD/patches/mindspeed/k26/ascend-lazy-dist-train.patch"
git -C third_party/MindSpeed apply "$PWD/patches/mindspeed/k26/ascend-lazy-dist-train.patch"
```

Tests exercise configuration-only imports with the real configuration module,
legacy package exports, import caching, and error propagation for requested
optional implementations. Complete NPU training remains to be validated.

Patch SHA-256: `5ee40290217a7c0f92a46938b48dfa9c2177502287624b67f05dda0018065f34`.


## MLA explicit process groups

`ascend-mla-process-groups.patch` updates the MindSpeed MLA initializer and its
DotProductAttention initializer wrapper for K26 Core's explicit `pg_collection`
interface. It forwards the group collection to base attention and core attention,
passes CP to rotary embeddings and TP to the output projection, and makes the
MindSpeed DotProductAttention replacement use the supplied TP size. Calls that
do not supply a collection retain the global-group fallback.

The patch also initializes `cache_mla_latents`, aligns MLA attention scaling with
Core's `mscale_all_dim`, and uses `original_max_position_embeddings` for YaRN
rather than the extended maximum sequence length. It retains MindSpeed's NPU
attention implementation; it does not add CUDA/TE kernels or FP8 support.

Apply as an increment, retaining earlier K26 patches:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C third_party/MindSpeed apply --check "$PWD/patches/mindspeed/k26/ascend-mla-process-groups.patch"
git -C third_party/MindSpeed apply "$PWD/patches/mindspeed/k26/ascend-mla-process-groups.patch"
```

Restart training workers. CPU checks cover explicit TP/CP identity propagation,
RoPE and YaRN constructor arguments, the original YaRN context length, MLA
scaling, wrapper keyword compatibility, and TP size selection. They use stubbed
NPU collaborators and do not validate attention outputs or distributed training.

MLA groups patch SHA-256: `936c81b4183be6ede3bb40f40b46f7acbda744e84ed232c09debc01db2eacef5`.


## BF16 model construction and forward interfaces

`ascend-te-model-contracts.patch` is an incremental compatibility patch for the
K26 Core baseline. Keep the previous MLA and optional-import patches applied.

The review covered Core attention/MLA callers, dense/shared MLP and TE grouped
experts, MindSpeed's compatible and performance grouped wrappers, packed-sequence
forwarding, and the BF16 block/recomputation selection used by the K26 RL recipe.

- TE attention receives `pg_collection`, uses its TP group and size, and uses its
  CP group and global ranks when CP is enabled.
- Fused Norm+ColumnLinear accepts explicit TP and stride, forwards the TP group
  through BF16 communication/linear calls and checkpoint metadata, and preserves
  weight/bias partition stride.
- Grouped column/row wrappers forward explicit groups; the bases select expert TP
  for experts and attention TP otherwise. Conflicting explicit groups fail.
- BF16 MLA low-rank projections use Core ColumnParallelLinear, which Core
  explicitly handles by gathering the compressed output. The TE `TELinear` shim
  is not selected for these projections.
- BF16 retains upstream Core `TransformerBlock.forward` and uniform recomputation,
  including padding masks and current quantization-context keyword names. The
  old custom block implementation remains restricted to FP8.
- Packed CP metadata is validated separately instead of forwarding unsupported
  keywords to the attention backend. Dynamic CP layouts differing from the
  configured groups are rejected explicitly; no communication algorithm is
  silently substituted.

Apply this increment in the container:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C third_party/MindSpeed apply --check "$PWD/patches/mindspeed/k26/ascend-te-model-contracts.patch"
git -C third_party/MindSpeed apply "$PWD/patches/mindspeed/k26/ascend-te-model-contracts.patch"
```

The updated K26 `runtime_env.sh` runs `tools/check_k26_runtime_contracts.py`
before Ray starts. It checks the loaded model and optimizer interfaces and reports the actual source
file if an old adapter is still selected. This is an early interface check, not
a model construction or numerical test; feature repatching inside workers still
requires the distributed smoke test.

CPU regression checks compare actual Core call keywords against patched adapter
signatures, exercise TE attention group propagation and packed forwarding, check
expert partition shapes with different attention/expert TP sizes, and verify
BF16 block selection and early rejection of old loaded adapters. All K26
MindSpeed patches were also applied together to a temporary source fixture and
syntax checked.

Validation scope is the current unquantized BF16, text-only, PP=1 smoke recipe
with uniform full recomputation. This is not a completed compatibility claim for
FP8/FP4, dynamic CP, pipeline overlap/DualPipe, MC2, custom FSDP, or delayed weight
gradients. NPU attention values, optimizer steps, rollout weight synchronization,
and checkpoint save/resume remain target-container checks. No local NPU runtime
was available for this review.

TE model patch SHA-256: `b722d959f963d189f6421e38863e7f8dfda159556c30bf84a73571440b92bd6e`.


## AdamW registered through Apex's FusedAdam interface

`ascend-adamw-api.patch` adds the `adam_w_mode` keyword to both MindSpeed AdamW
implementations registered as `apex.optimizers.FusedAdam`. K26 Core identifies
the Apex route and passes `adam_w_mode=config.decoupled_weight_decay`; the
previous MindSpeed signatures rejected this keyword before creating an optimizer.

`adam_w_mode=True` selects the existing AdamW implementation unchanged.
`adam_w_mode=False` raises explicitly because these implementations do not supply
coupled Adam weight decay. Unknown options still raise; unsupported parameters
are not silently discarded. The current K26 recipe uses AdamW semantics.

Apply while retaining earlier MindSpeed patches:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C third_party/MindSpeed apply --check "$PWD/patches/mindspeed/k26/ascend-adamw-api.patch"
git -C third_party/MindSpeed apply "$PWD/patches/mindspeed/k26/ascend-adamw-api.patch"
```

The updated runtime checker also checks that the optimizer actually selected by
Core accepts the Apex keyword before Ray starts. Restart the driver and workers.

Regression checks execute Core's Adam factory branch, including its eager moment
initialization, and the real default MindSpeed AdamW step and fused arithmetic
on CPU (only NPU placement of the step scalar is replaced). Five updates, with
save/restore after the second update, match Torch AdamW parameters and moments.
Additional checks execute MindSpeed chained-step control flow, validate clipping
call keywords against Core, and execute Core DistributedOptimizer's step-metadata
extraction for the default Apex-compatible optimizer.

This validates the current default `fused_adamw` route. Both constructor signatures
are tested, but `fused_torch_adamw`'s per-parameter step state is not validated as
an alternative distributed checkpoint path. Precision-aware optimizer extensions,
coupled Adam, CPU offload, and complete distributed checkpoint reconstruction are
outside this patch's tested scope. NPU kernels and distributed optimizer steps
still require the target-container smoke test.

AdamW API patch SHA-256: `5ee3553ce9420bc4e6e7f716777c68bb29f384bcd70344387eea0a732de7db55`.


## Context-parallel initialization against the K26 Core signature

`ascend-cp-initialization.patch` replaces the CP wrapper's hand-maintained
positional argument list with binding against the actual wrapped Core signature.
The old list included removed split/encoder parameters and omitted new fields;
its final `create_gloo_process_groups=True` landed in `sharp_enabled_group`,
causing a misleading SHARP assertion with `use_sharp=False`. Other options,
including CP size, were also shifted.

The wrapper now validates and forwards arguments using the native signature,
preserves new options and defaults, and uses those bound values when creating
MindSpeed's extra CP groups. Core validation remains active, and unknown options
still raise. No SHARP setting is enabled to work around the error.

The increment also forwards explicit process groups through the CP attention
constructor and non-TE attention/MLA wrappers, selecting explicit CP for Ulysses
when present. Keep earlier MLA process-group patches applied.

For eight ranks, TP=2, PP=2, CP=2 gives attention DP=1. ETP=1, EP=4, PP=2 gives
expert DP=1. The native RankGenerator produces identical pipeline groups for
both decompositions: `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`. This establishes rank-layout
consistency, not numerical compatibility of the CP/PP training implementation.

Apply while retaining existing K26 patches:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C third_party/MindSpeed apply --check "$PWD/patches/mindspeed/k26/ascend-cp-initialization.patch"
git -C third_party/MindSpeed apply "$PWD/patches/mindspeed/k26/ascend-cp-initialization.patch"
```

The updated runtime checker probes argument forwarding without creating process
groups: the probe captures the forwarded values and stops before any CP helper
runs. Restart the driver and workers after synchronizing the updated checker.

Regression checks reproduce the old SHARP error against the actual Core
signature/SHARP validation, verify defaults and new options, compare native rank
groups, and exercise CP constructor/wrapper process-group forwarding. These
checks do not execute HCCL collectives, CP attention kernels, PP scheduling, or
distributed checkpoint save/resume. The earlier PP=1/CP=1 smoke validation cannot
be treated as validation of this new PP=2/CP=2 training mode.

CP initialization patch SHA-256: `8e66104bd342256f214b1587b32cccd61bb8ca2010286954438e929d522b315c`.
