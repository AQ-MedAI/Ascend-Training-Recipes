# verl source patches

Upstream: https://github.com/verl-project/verl.git

| Profile | Upstream commit | Patch |
| --- | --- | --- |
| V4 | `809f2d8f59910dfd3a96745b68119829650250a0` | `v4.patch` + `v41.patch` + `entrypoint.patch` + `v4-mtp.patch` |
| V4.1 | `809f2d8f59910dfd3a96745b68119829650250a0` | `v4.patch` + `v41.patch` + `entrypoint.patch` + `v4-mtp.patch` |
| K2.6 RL / MOPD | `bee9f6f4bb1de784781e296f165b4c22d6c26cc1` | `k26.patch` + `k26-engine-registration.patch` + `k26-full-vocab-rpc.patch` |

V4 and V4.1 share the same final source tree: apply `v4.patch` first, then the incremental `v41.patch` and `entrypoint.patch`. `entrypoint.patch` resolves the V4 model provider from the pinned MindSpeed-LLM repository root (`pretrain_deepseek4.py`), which the launchers include in `PYTHONPATH`. Model architecture and recipe settings select the training path. Use the shared MindSpeed optimizer/checkpoint adaptations and matching Megatron-LM alongside this tree. The K2.6 patch is an alternative on its own upstream baseline; never stack it on the shared V4/V4.1 tree.

Run `bash tools/setup_sources.sh v4`, `v41`, or `k26`. K2.6 selects only verl: compatible MindSpeed, Megatron-LM, Megatron-Bridge and runtime dependencies still need to be supplied separately. It does not reuse the V4.1 dependency combination automatically.

Patch hashes and complete before/after source inventories are recorded in `dependencies/verl-sources.json`. Preparation checks all selected dependencies before changing them, removes only a verified managed patch, checks out the requested upstream commit and applies its patch. Additional edits or staged changes stop preparation.

The submodule HEAD records the upstream commit; modified and untracked patch files are expected. Do not commit the adaptation inside the submodule. Commit its upstream gitlink together with the manifest, patches and setup configuration in the recipe repository. Updating patches requires checksum/inventory regeneration and validation of all profiles, including switching across the K2.6 baseline. Keep patches out of automatic formatters.

The source manifest also recognizes the previous V4-only tree for safe migration. It is not an active profile. Extra edits still stop migration.

`k26-engine-registration.patch` retains the first optional engine ImportError and propagates it when the Megatron backend is selected. It never retries a partially initialized MindSpeed module; retrying can duplicate torch.compile patch registration. The runtime preflight initializes MindSpeed before Core/Bridge imports. Missing dependencies still need to be fixed in the selected environment.

`v4-mtp.patch` follows the shared V4/V4.1 stack. It preserves the MindSpeed engine config object during conversion, validates V4 MTP options and uses the native V4 auxiliary loss while returning PPO logits. It is not applied to K2.6.

`k26-full-vocab-rpc.patch` transports captured teacher hidden states as an explicit
dtype/shape/byte payload. vLLM 0.23 utility RPC does not reconstruct untyped Tensor
results with its default serialization settings; treating that descriptor as a
missing capture incorrectly aborts full-vocabulary distillation. The payload uses
native MessagePack values, validates its dtype, shape and byte count, and retains
compatibility with direct Tensor results. It does not enable pickle serialization.
Run `bash tools/setup_sources.sh k26` to update a prepared K26 tree, then restart
the training driver and teacher workers.

### DAPO group filtering

`group-filter.patch` (V4/V4.1) and `group-filter-k26.patch` (K2.6) enable group
filtering in the ordinary `main_ppo` trainer. They follow the semantics in
[upstream verl v0.6.1 DAPO](https://github.com/verl-project/verl/blob/v0.6.1/recipe/dapo/dapo_ray_trainer.py):
retain nonconstant reward groups and singletons, replenish until
`data.train_batch_size` prompts are available, and fail at the generation limit.
The feature is disabled by default. After `bash tools/setup_sources.sh <profile>`,
append these arguments to the Python training command:

```bash
algorithm.filter_groups.enable=True \
algorithm.filter_groups.metric=seq_reward \
algorithm.filter_groups.max_num_gen_batches=10
```

`seq_reward` uses the sum of token scores; `seq_final_reward` uses rewards after
the configured KL penalty. Other names (e.g. `acc`) select scalar reward extra
info. `max_num_gen_batches <= 0` means no generation-batch limit. For group-relative
RL, set `actor_rollout_ref.rollout.n > 1`; singletons are retained as in upstream.

Filtering precedes advantage calculation and optimizer updates. Before replenishment, the existing checkpoint manager restores
the same actor weights because hybrid rollout sleep may discard them at level 2.
This is a weight restore, not an optimizer update. Its cost is reported as
`timing_s/filter_groups/resume_rollout`. Existing teacher
and log-probability computation is retained so KL-based metrics remain correct.
Entire groups are kept even after token balancing. Timings accumulate across
replenishment batches. An incomplete batch at end of data raises an error instead
of silently updating on fewer prompts. Retained responses consume driver memory
while waiting for replenishment; bound the attempts for low-diversity datasets.
Filtering changes the training distribution; it is not enabled automatically for
RL, OPD or MOPD recipes. Metrics use the `filter_groups/` prefix.

`v4-0731-backbone.patch` follows the V4/V4.1 stack. It selects the official 0731
backbone for RL/MOPD, disables DSpark heads, preserves legacy V4 MTP behavior,
and passes the normalized head/ratio settings to vLLM via HF overrides. Source
checkpoints are unchanged; exported backbone-only configs do not advertise
missing DSpark parameters. K2.6 is unaffected.

### V4 / V4.1 full-vocabulary OPD runtime fixes

`v4-opd-runtime.patch` follows the shared V4/V4.1 patch stack and is applied by
`tools/setup_sources.sh` and image source preparation. It fixes three independent
runtime issues in full-vocabulary OPD/MOPD:

- Encode teacher hidden states as validated dtype/shape/byte payloads for untyped
  vLLM utility RPC, and fetch every local DP engine instead of discarding captures
  from engines other than the first. Failed captures now include worker status.
- Validate explicit teacher-to-student response-row mappings when their prompt
  lengths differ. Identical-prompt artifacts remain compatible. A custom teacher
  prompt producer must provide all three fields: `student_sequence_length`,
  `student_row_start`, and `student_row_end`; the teacher `sequence_length` and
  `row_start`/`row_end` retain their original source coordinates. The response
  token sequence must be identical; this does not align different tokenizations.
- Use the module-level `inspect` import in checkpoint saves so HF export works
  when swap optimizer is disabled.

This patch does not add a HealthBench dataset, privileged-prompt worker or OPSD
launcher, and does not change K26. Prepare sources with
`bash tools/setup_sources.sh v4` (or `v41`) and restart all training/teacher
workers; rebuild images to distribute the fix. Previous managed V4/V4.1 trees
are retained in the manifest for verified migration.

CPU regressions reconstruct the managed source stack in a temporary directory:

```bash
python -m unittest discover -s tests -p 'test_v4_opd_*.py' -v
```

The tests cover typed RPC payloads, nonfirst DP-engine captures, forward/reverse
KL values and gradients with different prompt lengths, and both HF save branches.
MessagePack round-trip coverage requires `msgspec`. NPU/multi-node validation is
still required in the target runtime.

### Chunked teacher prefill

`v4-opd-chunked-prefill.patch` follows `v4-opd-runtime.patch` for V4/V4.1.
It accumulates response-prediction hidden rows by absolute token position across
prefill chunks, checks complete coverage, and passes `row_start`, `row_end`, and
`request_id` through the worker RPC. Teacher chunked prefill is supported with
`max_num_seqs=1`; `max_num_batched_tokens` may be smaller than `max_model_len`.
Rebuild the image and restart all teacher workers together so the RPC caller and
worker extension use the same interface. Existing CPU RPC regressions also cover
chunk accumulation and configuration validation. NPU validation remains required.
