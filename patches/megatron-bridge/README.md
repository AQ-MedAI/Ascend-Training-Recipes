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


## Kimi conditional-generation checkpoint: text-only RL

`ascend-kimi-text-bridge.patch` registers `KimiK25ForConditionalGeneration` for
text-only RL. The pinned public Bridge has a Kimi K2 provider but no converter for
this architecture; the optional-PEFT patch alone cannot add model support.

The new bridge reads the nested `text_config`, requires a DeepSeek-V3-compatible
unquantized text backbone, and delegates its provider and tensor transforms to
the existing DeepSeek-V3 implementation. It builds a text actor only. It detects
language-model prefixes and the separate output head from the local safetensors
index (or a single safetensors file), remaps both ordinary and gated expert
weights, and retains the original HF names when streaming weights to rollout.
It raises on ambiguous layouts, unmapped actor parameters, and missing mapped
source tensors instead of leaving actor weights randomly initialized.

The composite checkpoint configuration, vision tensors, and rollout model remain
unchanged. This does not implement multimodal actor training or standalone
multimodal HF export. Use the original checkpoint to initialize rollout, and
use distributed training checkpoints for actor save/resume.

Apply this increment while retaining `ascend-optional-peft.patch`:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C /opt/ascend-training/Megatron-Bridge apply --check "$PWD/patches/megatron-bridge/ascend-kimi-text-bridge.patch"
git -C /opt/ascend-training/Megatron-Bridge apply "$PWD/patches/megatron-bridge/ascend-kimi-text-bridge.patch"
```

Restart workers in fresh processes. The existing four-layer composite checkpoint
can be reused; do not rename its architecture or remove its vision weights.
A local checkpoint path must remain in `config.name_or_path` for layout detection.
Unknown text-backbone types or weight layouts fail explicitly and need review.

Validation: patch application to the pinned public baseline and six CPU tests of
registration declarations, nested config isolation, four checkpoint prefixes,
gated mappings, and rejection paths. These tests use lightweight collaborators;
they do not execute the complete Bridge import, distributed tensor transforms,
NPU forward/backward, optimizer steps, rollout synchronization, or checkpoint
save/resume. All of those still require target-container validation.

Text bridge patch SHA-256: `f317a27757ed1f2dfb14705f001c2d2ef923b490755ffc163e2b0cc22208505d`.


## Optional import diagnostics

`ascend-optional-import-logging.patch` changes the caught missing-symbol message
in `safe_import_from` from INFO to DEBUG, matching the existing missing-module
behavior. This avoids a traceback at normal logging levels when the MindSpeed
Apex shim lacks the optional `MixedFusedLayerNorm` symbol. DEBUG logging retains
the traceback. Availability flags, fallbacks, and errors when a missing symbol
is actually used are unchanged; unexpected exceptions still propagate.

Apply this increment while retaining the existing Bridge patches:

```bash
cd /workspace-verl/Ascend-Training-Recipes
git -C /opt/ascend-training/Megatron-Bridge apply --check "$PWD/patches/megatron-bridge/ascend-optional-import-logging.patch"
git -C /opt/ascend-training/Megatron-Bridge apply "$PWD/patches/megatron-bridge/ascend-optional-import-logging.patch"
```

Restart workers to load the patched module. Verification covers INFO/DEBUG
logging behavior, the unavailable-symbol exception, and propagation of unexpected
import failures. This changes diagnostics only; it does not supply Apex kernels.

Logging patch SHA-256: `42ffffbd690a9d1915dc7cae7909b294ac8c21702ff6735a6422816d1416240b`.
