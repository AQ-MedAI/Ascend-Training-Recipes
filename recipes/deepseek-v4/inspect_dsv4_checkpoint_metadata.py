#!/usr/bin/env python3
"""Inspect a trusted local DCP checkpoint without loading tensor payloads."""

import argparse
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="DCP directory or its parent actor/step directory")
    args = parser.parse_args()
    root = args.checkpoint
    paths = [root] if root.is_file() else sorted(root.rglob(".metadata"))
    if not paths:
        parser.error(f"No .metadata found under {root}")

    import torch
    import megatron.core
    from torch.distributed.checkpoint import FileSystemReader

    print(f"torch={torch.__version__}, mcore={megatron.core.__version__}")
    print(f"mcore_source={megatron.core.__file__}")
    for path in paths:
        print(f"\n[checkpoint] {path}")
        metadata = FileSystemReader(str(path.parent)).read_metadata()
        entries = metadata.state_dict_metadata
        mcore_data = getattr(metadata, "mcore_data", None) or {}
        print(f"metadata_fields={sorted(vars(metadata))}")
        print(f"state_entries={len(entries)}, mcore_entries={len(mcore_data)}")
        print(f"entry_types={dict(Counter(type(v).__name__ for v in entries.values()))}")
        print(f"prefix_counts={dict(Counter('.'.join(k.split('.')[:3]) for k in entries))}")
        selected = [k for k in entries if any(s in k for s in (
            "linear_q_up_proj", "compressor", "param_state", "dp_group_idx"))]
        print(f"\n[related entries] {len(selected)} total; showing at most 40")
        for key in selected[:40]:
            value = entries[key]
            shape = getattr(value, "size", None)
            chunks = getattr(value, "chunks", [])
            print(f"{key}: type={type(value).__name__}, shape={shape}, chunks={len(chunks)}, mcore={mcore_data.get(key)}")
        target = "chained_0.optimizer.state.exp_avg.decoder.layers.2.self_attention.linear_q_up_proj.weight"
        print(f"\n[target] {target}")
        print(f"in_state_metadata={target in entries}, in_mcore_data={target in mcore_data}")
        for filename in ("metadata.json",):
            config = path.parent / filename
            if config.is_file():
                print(f"[{filename}] {config.read_text()}")
        # Storage offsets allow checking truncated/missing shards without reading them.
        required = {}
        for item in (getattr(metadata, "storage_data", None) or {}).values():
            name = getattr(item, "relative_path", None)
            if name is not None:
                required[name] = max(required.get(name, 0), item.offset + item.length)
        bad = []
        for name, end in required.items():
            shard = path.parent / name
            size = shard.stat().st_size if shard.is_file() else -1
            if size < end:
                bad.append((name, size, end))
        print(f"[shard files] checked={len(required)}, missing_or_short={bad}")


if __name__ == "__main__":
    main()
