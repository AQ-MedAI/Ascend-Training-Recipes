# MindSpeed-LLM source patches

Upstream: https://github.com/Ascend/MindSpeed-LLM.git

Pinned commit: `53ff1095801f9ce14b14be3e62739e742f8b9067`.

- V4 applies `common.patch`: model, sparse-attention and checkpoint adaptations.
- V4.1 applies `common.patch` followed by `v41.patch`: bundled Engram hash implementation, its default path and the Transformers 5.14.1 requirement.

Run `bash tools/setup_sources.sh v4` or `bash tools/setup_sources.sh v41` from the recipe root. The tool verifies patch SHA-256 values and complete source inventories in `dependencies/mindspeed-llm-sources.json`. Switching reverses only a recognized profile's patches, then applies the requested profile. Additional edits or staged changes stop preparation.

The submodule HEAD stays at the upstream commit. Modified and untracked patch files inside it are expected; do not commit these in the submodule or discard them before training. The recipe commit must include the upstream gitlink, manifest, patches and setup tools together.

Upstream nested submodule pins are preserved and verified, but nested repositories are not initialized. Training uses the recipe's explicitly selected dependencies.

When updating a patch, regenerate its SHA-256 and final source inventory, then verify both profiles and their round-trip switching. Keep patch files out of automatic formatters.
