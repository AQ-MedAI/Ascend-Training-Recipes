# verl source patches

Upstream: https://github.com/verl-project/verl.git

| Profile | Upstream commit | Patch |
| --- | --- | --- |
| V4 | `809f2d8f59910dfd3a96745b68119829650250a0` | `v4.patch` + `v41.patch` + `entrypoint.patch` + `v4-mtp.patch` |
| V4.1 | `809f2d8f59910dfd3a96745b68119829650250a0` | `v4.patch` + `v41.patch` + `entrypoint.patch` + `v4-mtp.patch` |
| K2.6 RL / MOPD | `bee9f6f4bb1de784781e296f165b4c22d6c26cc1` | `k26.patch` + `k26-engine-registration.patch` |

V4 and V4.1 share the same final source tree: apply `v4.patch` first, then the incremental `v41.patch` and `entrypoint.patch`. `entrypoint.patch` resolves the V4 model provider from the pinned MindSpeed-LLM repository root (`pretrain_deepseek4.py`), which the launchers include in `PYTHONPATH`. Model architecture and recipe settings select the training path. Use the shared MindSpeed optimizer/checkpoint adaptations and matching Megatron-LM alongside this tree. The K2.6 patch is an alternative on its own upstream baseline; never stack it on the shared V4/V4.1 tree.

Run `bash tools/setup_sources.sh v4`, `v41`, or `k26`. K2.6 selects only verl: compatible MindSpeed, Megatron-LM, Megatron-Bridge and runtime dependencies still need to be supplied separately. It does not reuse the V4.1 dependency combination automatically.

Patch hashes and complete before/after source inventories are recorded in `dependencies/verl-sources.json`. Preparation checks all selected dependencies before changing them, removes only a verified managed patch, checks out the requested upstream commit and applies its patch. Additional edits or staged changes stop preparation.

The submodule HEAD records the upstream commit; modified and untracked patch files are expected. Do not commit the adaptation inside the submodule. Commit its upstream gitlink together with the manifest, patches and setup configuration in the recipe repository. Updating patches requires checksum/inventory regeneration and validation of all profiles, including switching across the K2.6 baseline. Keep patches out of automatic formatters.

The source manifest also recognizes the previous V4-only tree for safe migration. It is not an active profile. Extra edits still stop migration.

`k26-engine-registration.patch` retains the first optional engine ImportError and propagates it when the Megatron backend is selected. It never retries a partially initialized MindSpeed module; retrying can duplicate torch.compile patch registration. The runtime preflight initializes MindSpeed before Core/Bridge imports. Missing dependencies still need to be fixed in the selected environment.

`v4-mtp.patch` follows the shared V4/V4.1 stack. It preserves the MindSpeed engine config object during conversion, validates V4 MTP options and uses the native V4 auxiliary loss while returning PPO logits. It is not applied to K2.6.
