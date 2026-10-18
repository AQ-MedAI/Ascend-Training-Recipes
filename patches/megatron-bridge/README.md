# Megatron-Bridge K2.6 compatibility

Upstream: https://github.com/NVIDIA/Megatron-Bridge.git

Baseline commit: `c15303e05b3d48505e4b1b0047faa0c321368856`.

`ascend-optional-peft.patch` defers LoRA-specific imports until adapter conversion or merging is requested. Ordinary weight conversion no longer imports Transformer Engine through this PEFT import chain. This patch does not implement Ascend LoRA or complete K2.6 compatibility.

Apply in the matching Bridge checkout before sourcing the K2.6 runtime environment:

```bash
RECIPE_ROOT=/workspace/Ascend-Training-Recipes
cd /opt/ascend-training/Megatron-Bridge
git apply --check "$RECIPE_ROOT/patches/megatron-bridge/ascend-optional-peft.patch"
git apply "$RECIPE_ROOT/patches/megatron-bridge/ascend-optional-peft.patch"
```

Use the actual recipe path on your machine. Apply once per checkout; the runtime environment script does not apply patches automatically. Keep patch files excluded from automatic formatting.

Validation: patch applicability and Python syntax checks only; full Bridge import and NPU training remain to be verified in the target environment.

Patch SHA-256: `066767a52d92a6558bcfdd67f1b563019b1b94ede0a551deb3a1636d7a14008f`.
