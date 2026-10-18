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
