# MindSpeed source patches

Upstream: https://github.com/Ascend/MindSpeed.git

Pinned commit: `c356513e331e55d53b799cfde51c7b23901d23dc`.

- `common.patch` provides swap optimizer parameter mapping, checkpoint synchronization and FP32 state handling, and Megatron feature registration.
- `v41.patch` adds matching optimizer and checkpoint interfaces, parameter grouping, duplicate all-gather handling, and the NumPy 1.26.4 requirement.

Run `bash tools/setup_sources.sh v4` or `bash tools/setup_sources.sh v41` from the recipe root. The tool verifies patch SHA-256 values and complete source inventories in `dependencies/mindspeed-sources.json`. Switching reverses only a recognized profile's patches. Additional source edits or staged changes stop preparation.

The submodule HEAD stays at the upstream commit. Patch modifications in the submodule are expected; do not commit them there or discard them before training. Commit the upstream gitlink, manifest, patches and configuration together in the recipe repository.

Both profiles use the same optimizer interface and must be paired with the shared verl and matching Megatron-LM versions. Updating a patch requires regenerating its checksum and source inventory and validating both profiles and round-trip switching. Patch files must remain excluded from automatic formatting.

Both V4 and V4.1 apply `common.patch` followed by `v41.patch` and select identical final sources. Existing V4-only managed checkouts are recognized for safe upgrade; additional edits are preserved by refusing to proceed.
