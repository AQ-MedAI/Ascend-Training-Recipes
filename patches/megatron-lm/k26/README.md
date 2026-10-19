# K2.6 Megatron-LM optional Transformer Engine compatibility

Upstream: https://github.com/NVIDIA/Megatron-LM.git

Baseline commit: `96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472`.

`ascend-optional-te.patch` treats incomplete optional Transformer Engine imports, including the MindSpeed TE shim, as unavailable. It also treats `fp8=False` and `fp8=None` as disabled in `CheckpointWithoutOutput`. BF16 recomputation uses the existing non-TE path. Requesting FP8 without complete TE support raises a clear error; no fake TE operations are added.

This is specific to the K2.6 baseline, separate from the V4/V4.1 Megatron-LM patch.

Apply once in the container:

```bash
cd /opt/ascend-training/Megatron-LM
git apply --check /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-optional-te.patch
git apply /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-optional-te.patch
```

`ascend-fp8-utils.patch` is an additional patch for `fp8_utils.py` and `fp4_utils.py`. It checks actual FP8 state/tensor imports instead of treating an importable MindSpeed TE shim as a complete backend. FP4 reuses the validated flag. BF16 retains its ordinary context; explicit FP8/FP4 requests without TE raise an error.

If `ascend-optional-te.patch` is already applied, leave it in place and apply only this increment:

```bash
cd /opt/ascend-training/Megatron-LM
git apply --check /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-fp8-utils.patch
git apply /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-fp8-utils.patch
```

Restart the failed training task in fresh processes after applying. The runtime environment script does not apply this patch automatically. This does not complete K2.6 NPU compatibility or provide FP8 support.

Patch SHA-256: `c59e7a1986a791bdbaa88632422cc2cf052b52494ee973d64716da7159da3ed2`.

FP8 utilities patch SHA-256: `4b8a9670e5921c090a7e253193b22f741b1b75dc13685cbc8dabf59f5f374949`.

## Legacy MindSpeed expert checkpoint API

`ascend-expert-checkpoint.patch` restores `expert_dist_ckpt_decorator`, which the pinned MindSpeed grouped-linear modules still import. The implementation preserves the earlier Megatron interface: temporarily select expert tensor-parallel rank, world size and group while the wrapped checkpoint method runs, then restore all three in `finally`, including nested and failed calls. Native K2.6 Megatron checkpoint methods continue using their explicit process groups.

Apply after the optional-TE patches (it touches a separate file):

```bash
cd /opt/ascend-training/Megatron-LM
git apply --check /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-expert-checkpoint.patch
git apply /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-expert-checkpoint.patch
```

CPU tests cover temporary Expert-TP state and restoration on normal, exceptional and nested execution. They do not validate distributed checkpoint files or full NPU save/resume.

Expert checkpoint patch SHA-256: `f081d08714203706d406f70fc7fe84b78f7f15e091258b092e77fc279a5e7d26`.


## Legacy MoE routing API

`ascend-moe-routing.patch` restores `topk_softmax_with_capacity` using the baseline's
`topk_routing_with_score_function` and `apply_router_token_dropping`. It returns
probabilities, the routing map, and **counts before token dropping**, as expected
by MindSpeed. It retains the old argument order, expert bias, group selection,
scaling, and capacity/padding options. The compatibility argument
`deterministic_mode` follows the previous implementation; PyTorch's global
setting controls deterministic execution.

Apply this additional patch without reverting the earlier three:

```bash
git -C /opt/ascend-training/Megatron-LM apply --check /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-moe-routing.patch
git -C /opt/ascend-training/Megatron-LM apply /workspace-verl/Ascend-Training-Recipes/patches/megatron-lm/k26/ascend-moe-routing.patch
```

CPU verification compared outputs and input gradients with the previous routing
implementation across 384 configurations. Repository tests also check the
three-value contract, expert bias, group limits, and counts before capacity
truncation. These checks do not validate distributed NPU execution.

See [MindSpeed K26 compatibility](../../mindspeed/k26/README.md) for the accompanying
import and grouped-expert adaptations and remaining validation scope.

Routing patch SHA-256: `65653945aac928dc1a7e7ffda9b79f4fc65c6d31d04b62d8ef40d796da156df3`.


## Optional TE symbols used as runtime types

`ascend-optional-te-types.patch` guards type checks in Core Attention, MLA,
shared-expert initialization, and the FP8 linear-classification helpers.
An importable Transformer Engine shim does not guarantee that `TELinear` or
other imported names are classes. Calling `isinstance(module, TELinear)` when
that symbol is `None`, a function, or another placeholder raises TypeError even
in BF16 initialization.

The patch requires each optional class symbol to be a type before using it as
`isinstance`'s second argument. It does not disable TE globally, create a dummy
linear implementation, change NPU projection selection, or suppress genuine
model-construction errors. Complete TE classes retain the original checks and
FP8/FP4 input-preservation optimization.

Apply while retaining the previous Core and MindSpeed patches:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C /opt/ascend-training/Megatron-LM apply --check "$PWD/patches/megatron-lm/k26/ascend-optional-te-types.patch"
git -C /opt/ascend-training/Megatron-LM apply "$PWD/patches/megatron-lm/k26/ascend-optional-te-types.patch"
```

Restart workers. Regression checks reproduce the reported failure with the
unpatched Core Attention constructor and execute the complete initialization
chain from Core MLASelfAttention through the patched MindSpeed MLA initializer
into Core Attention. The actual constructor bodies run with lightweight backend
collaborators rather than substituting the inner Attention constructor. Missing,
function-valued, and other non-class TE symbols no longer cause TypeError. Other
checks cover the native MLA projection branch, shared-expert guards, and linear
classification; complete TE-class behavior is retained.

These are constructor/control-flow checks, not NPU kernel or numerical tests.
Full BF16 training, optimizer updates, distributed checkpoint recovery, and
rollout synchronization remain target-container validations. FSDP and optional
quantized checkpoint integrations are outside this patch's validated scope.

Optional TE types patch SHA-256: `a272e370341b345903de3d2d4cf4a79697a44e8033de17a47f5ecf14615463c6`.


## In-process synchronous checkpoint writer

`ascend-checkpoint-inprocess.patch` adds an opt-in `ASCEND_CHECKPOINT_WRITER=inprocess`
mode to the pinned Core filesystem writer. The stock synchronous save still forks
bucket writer processes. Native serialization crashes were traced to such a child
while it read a CPU FP32 tensor after optimizer offload; the precise storage/runtime
cause is not proven. This mode removes that fork boundary and uses the same native
bucket serializer, fsync, write results and metadata in the caller process. It
preserves CRC32 and the checkpoint format. Buckets are written sequentially, which
may reduce I/O throughput.

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C /opt/ascend-training/Megatron-LM apply --check "$PWD/patches/megatron-lm/k26/ascend-checkpoint-inprocess.patch"
git -C /opt/ascend-training/Megatron-LM apply "$PWD/patches/megatron-lm/k26/ascend-checkpoint-inprocess.patch"
```

K26 RL defaults to this mode and synchronous saves. Update the recipe and restart
workers after applying. The runtime checker rejects an unpatched Core when this
mode is selected. `ASCEND_CHECKPOINT_WRITER=multiprocess` restores the previous
Core path for comparison. In-process mode is for synchronous checkpointing only;
it does not make asynchronous background checkpoint processes safe on Ascend.
Target-container save, load and resumed optimizer updates still require validation.
