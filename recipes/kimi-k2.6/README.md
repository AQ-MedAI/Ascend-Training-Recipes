# Kimi K2.6

## Entrypoints

| Script | Purpose |
| --- | --- |
| [kimi_k26_rl.sh](kimi_k26_rl.sh) | Full-model GRPO; environment and Hydra overrides supported |
| [kimi_k26_rl_8npu_cutlayer.sh](kimi_k26_rl_8npu_cutlayer.sh) | Four-layer RL smoke test, one node with eight NPUs, two steps |
| [kimi_k26_mopd.sh](kimi_k26_mopd.sh) | Full-model, two-teacher full-vocabulary MOPD; reverse KL by default |
| [kimi_k26_mopd_16npu_cutlayer.sh](kimi_k26_mopd_16npu_cutlayer.sh) | Four-layer, two-teacher full-vocabulary MOPD smoke test on one 16-NPU node |
| [kimi_k26_opd.sh](kimi_k26_opd.sh) | Full-model, single-teacher forward-KL top-k OPD |
| [kimi_k26_mopd_reverse_kl.sh](kimi_k26_mopd_reverse_kl.sh) | Alternative full-model two-teacher reverse-KL profile |
| [kimi_k26_mopd_shared_teacher.sh](kimi_k26_mopd_shared_teacher.sh) | Shared teacher pool; at most one awake teacher |

Names follow the V4 pattern: model prefix, training mode, and resource/cut-layer suffix.
The former `test_kimi_full_rl.sh` enabled distillation and is now named `kimi_k26_opd.sh`.
Distillation profiles retain their existing hyperparameters. Configure their model, data,
reward, log and shared-cache paths before use; some profiles still set paths directly.

## Four-layer RL smoke test

Use the `aq/k26-integration` verl revision and a compatible K26 runtime on every node.
The RL entry defaults to `third_party/verl`; override `VERL_ROOT` when needed.
`vanilla_mbridge=False` selects Megatron-Bridge, not AQ mbridge. Do not assume that the
V4/V4.1 MindSpeed, Megatron or native operator combination is compatible with K26.

The smoke entry automatically creates a four-layer checkpoint when `MODEL_PATH` is
missing. Set `FULL_MODEL_PATH` to an unquantized BF16/FP16/FP32 Kimi K2.6 checkpoint.
The helper keeps decoder layers 0–3, embeddings, final norms, output heads, vision
weights and tokenizer/custom-code assets; MTP weights are excluded. It handles sharded
or single-file safetensors and nested `text_config`. Quantized sources must be converted
beforehand. A temporary sibling directory is published only after conversion succeeds.

Existing outputs are validated and reused, even without `FULL_MODEL_PATH`. Invalid or
incomplete existing directories cause an error and are never automatically overwritten.
Use a shared `MODEL_PATH` visible on every node; the default is a local temporary path.
Conversion needs CPU memory for one selected source shard and disk space for the retained
weights. Source checkpoints are not modified.

From the repository root:

```bash
FULL_MODEL_PATH=/path/to/Kimi-K2.6_bf16 \
MODEL_PATH=/path/to/Kimi-K2.6-4layers \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/valid.parquet \
REWARD_FN_PATH=/path/to/reward.py \
OUTPUT_DIR=/path/to/output \
LOG_DIR=/path/to/logs \
bash recipes/kimi-k2.6/kimi_k26_rl_8npu_cutlayer.sh
```

This profile uses TP8/PP1/CP1/EP8, a single TP8 rollout replica, eight prompts and two
samples per prompt. It disables distillation, initial validation and checkpoint saving.
It omits the full model's uneven pipeline split. Success requires two training steps,
including rollout, backward/optimizer execution and subsequent rollout weight updates.
The wrapper fixes smoke-test settings and rejects Hydra overrides; use the base RL entry
for other topologies. NPU execution and peak memory for this profile remain unvalidated.

For full-model RL, use `kimi_k26_rl.sh`. Multi-node jobs require a running Ray cluster
and identical source, model, data and runtime paths on every node. Teacher models used
for full-vocabulary KL must have compatible tokenizers and vocabularies. Shared teacher
profiles additionally depend on the corresponding scheduling support in the K26 fork.

See [dependencies](../../docs/DEPENDENCIES.md) and [validation](../../docs/VALIDATION.md).

The checkpoint can also be prepared independently:

```bash
python3 recipes/kimi-k2.6/prepare_kimi_k26_cut_model.py \
  --input-path /path/to/Kimi-K2.6_bf16 \
  --output-path /path/to/Kimi-K2.6-4layers --num-layers 4
```

The helper requires `torch` and `safetensors` in `PYTHON_BIN`'s environment.
