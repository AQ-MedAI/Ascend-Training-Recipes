# Framework patches

The active source manifest is [dependencies/framework-sources.json](../dependencies/framework-sources.json). It pins each public commit, patch order, patch SHA-256 and source-file inventory. Patch paths are relative to the manifest, independent of the current working directory.

- `megatron-lm/`: Megatron-LM adaptations.
- `vllm/`: V4 baseline adaptations followed by the V4.1 backport.
- `vllm-ascend/`: Ascend adaptations followed by the V4.1 backport.
- `mindspeed-llm/`: pinned upstream source with common V4 and incremental V4.1 patches. See [MindSpeed-LLM](mindspeed-llm/README.md).
- `mbridge/`: upstream v0.15.1 plus a managed adaptation, prepared by `tools/setup_sources.sh`. See [mbridge](mbridge/README.md).
- `megatron-lm/legacy/`: an older patch retained for provenance; not selected by the active manifest or included in the image build. Its receipt is in `dependencies/legacy/`.

## Prepare sources locally

From the recipe repository root:

```bash
python tools/prepare_framework_sources.py --root /path/to/framework-sources
```

The tool creates `Megatron-LM`, `vllm` and `vllm-ascend` below the destination. It fetches fixed commits, validates patch checksums, checks and applies each patch, then verifies source contents and executable modes. For backports, it also verifies the intermediate baseline. An existing source directory is rejected to avoid overwriting local changes. After a failed preparation, inspect and preserve any changes before removing the incomplete destination or choosing another empty destination.

To select another manifest, pass `--manifest /path/to/framework-sources.json`. Source preparation does not install Python packages, compile native operators or initialize the recipe fork submodules. Docker uses the same tool and manifest, then performs installation and compilation.

Changes to a patch require updating its checksum and the affected source inventory. Validate reconstruction before publishing a new manifest; do not update hashes merely to bypass a failed check.
