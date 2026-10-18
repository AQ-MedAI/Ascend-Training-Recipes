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
