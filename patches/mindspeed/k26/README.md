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

