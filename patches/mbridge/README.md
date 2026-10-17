# mbridge adaptation

V4 and V4.1 use upstream [ISEEKYAN/mbridge](https://github.com/ISEEKYAN/mbridge), tag `v0.15.1`, commit `0cd4ae23f2425da77a80cb3f517828452fa8e984`, followed by `ascend-training.patch`.

The patch changes eight files for DeepSeek weight loading/export, mappings and safetensors handling. The base and final inventories and patch checksum are recorded in [mbridge-sources.json](../../dependencies/mbridge-sources.json).

```bash
bash tools/setup_sources.sh v41
# Or select the V4 dependency combination:
bash tools/setup_sources.sh v4
```

The upstream submodule remains at `third_party/mbridge`; existing launcher and editable-install paths do not change. The script verifies patches before changing sources. Repeated runs accept exactly the managed patched tree, including ignored build artifacts. Additional source edits or staged changes stop preparation; preserve them before retrying. Do not commit the patch inside the submodule. Commit its upstream gitlink and the patch/manifest together in this recipe repository.

A recursive clone alone only retrieves the unpatched base; always run `setup_sources.sh` afterward. To reconstruct mbridge into an independent empty destination:

```bash
python tools/prepare_framework_sources.py --root /path/to/sources --manifest dependencies/mbridge-sources.json
```

K2.6 currently selects NVIDIA Megatron-Bridge (`vanilla_mbridge=False`), not this package. This conversion does not change or validate the K2.6 environment.
