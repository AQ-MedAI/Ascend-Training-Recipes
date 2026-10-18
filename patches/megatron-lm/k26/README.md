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

Restart the failed training task in fresh processes after applying. The runtime environment script does not apply this patch automatically. This does not complete K2.6 NPU compatibility or provide FP8 support.

Patch SHA-256: `c59e7a1986a791bdbaa88632422cc2cf052b52494ee973d64716da7159da3ed2`.
