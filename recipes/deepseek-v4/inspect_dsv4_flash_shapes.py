#!/usr/bin/env python3
"""Inspect DeepSeek-V4-Flash config and safetensor weight/scale shapes.

This is a read-only helper for debugging whether a checkpoint is full HF layout
or already TP/EP-sharded.  It only opens config/index/safetensor files on CPU.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


DEFAULT_KEYS = [
    "embed.weight",
    "head.weight",
    "layers.0.attn.wq_a.weight",
    "layers.0.attn.wq_b.weight",
    "layers.0.attn.wkv.weight",
    "layers.0.attn.wo_a.weight",
    "layers.0.attn.wo_b.weight",
    "layers.0.ffn.gate.weight",
    "layers.0.ffn.gate.bias",
    "layers.0.ffn.experts.0.w1.weight",
    "layers.0.ffn.experts.0.w1.scale",
    "layers.0.ffn.experts.0.w2.weight",
    "layers.0.ffn.experts.0.w2.scale",
    "layers.0.ffn.experts.0.w3.weight",
    "layers.0.ffn.experts.0.w3.scale",
    "layers.0.ffn.experts.1.w1.weight",
    "layers.0.ffn.experts.1.w1.scale",
    "layers.0.ffn.experts.1.w2.weight",
    "layers.0.ffn.experts.1.w2.scale",
    "layers.0.ffn.experts.1.w3.weight",
    "layers.0.ffn.experts.1.w3.scale",
    "layers.0.ffn.experts.128.w1.weight",
    "layers.0.ffn.experts.128.w1.scale",
    "layers.0.ffn.experts.128.w2.weight",
    "layers.0.ffn.experts.128.w2.scale",
    "layers.0.ffn.experts.128.w3.weight",
    "layers.0.ffn.experts.128.w3.scale",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.shared_experts.w2.weight",
    "layers.0.ffn.shared_experts.w3.weight",
]

CONFIG_FIELDS = [
    "architectures",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "ffn_hidden_size",
    "moe_intermediate_size",
    "moe_ffn_hidden_size",
    "num_hidden_layers",
    "num_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "n_routed_experts",
    "num_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "vocab_size",
    "padded_vocab_size",
    "q_lora_rank",
    "o_lora_rank",
    "kv_lora_rank",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tensor_info(model_path: Path, weight_map: dict[str, str], key: str) -> tuple[tuple[int, ...], str, str] | None:
    filename = weight_map.get(key)
    if filename is None:
        return None
    with safe_open(str(model_path / filename), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    return tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""), filename


def print_kv(key: str, value) -> None:
    print(f"{key:<54} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        nargs="?",
        default="/path/to/DeepSeek-V4-Flash",
        help="HF/safetensors model directory. Defaults to the DS4 Flash shared path.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer id to inspect. Default: 0.",
    )
    parser.add_argument(
        "--experts",
        default="0,1,128",
        help="Comma-separated expert ids to inspect. Default: 0,1,128.",
    )
    parser.add_argument(
        "--key",
        action="append",
        default=[],
        help="Additional exact tensor key to inspect. Can be repeated.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"

    print(f"[inspect] model_path: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"model.safetensors.index.json not found: {index_path}")

    cfg = load_json(config_path)
    index = load_json(index_path)
    weight_map = index["weight_map"]

    print("\n[config]")
    for field in CONFIG_FIELDS:
        if field in cfg:
            print_kv(field, cfg[field])

    layer = args.layer
    expert_ids = [int(x.strip()) for x in args.experts.split(",") if x.strip()]

    keys = []
    for key in DEFAULT_KEYS:
        key = key.replace("layers.0.", f"layers.{layer}.")
        keys.append(key)

    for expert_id in expert_ids:
        for suffix in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"):
            keys.append(f"layers.{layer}.ffn.experts.{expert_id}.{suffix}")

    keys.extend(args.key)
    keys = list(dict.fromkeys(keys))

    print("\n[weights]")
    by_file: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        info = tensor_info(model_path, weight_map, key)
        if info is None:
            print_kv(key, "MISSING")
            continue
        shape, dtype, filename = info
        print_kv(key, f"shape={shape}, dtype={dtype}, file={filename}")
        by_file[filename].append(key)

    print("\n[file summary]")
    for filename, file_keys in sorted(by_file.items()):
        print_kv(filename, f"{len(file_keys)} inspected key(s)")

    print("\n[derived checks]")
    hidden = cfg.get("hidden_size")
    moe_intermediate = cfg.get("moe_intermediate_size") or cfg.get("moe_ffn_hidden_size")
    if hidden is not None:
        print_kv("config.hidden_size", hidden)
    if moe_intermediate is not None:
        print_kv("config.moe_intermediate_size", moe_intermediate)

    for expert_id in expert_ids:
        w1 = tensor_info(model_path, weight_map, f"layers.{layer}.ffn.experts.{expert_id}.w1.weight")
        w3 = tensor_info(model_path, weight_map, f"layers.{layer}.ffn.experts.{expert_id}.w3.weight")
        if w1 and w3:
            merged_shape = (w1[0][0] + w3[0][0], w1[0][1])
            print_kv(f"expert {expert_id} cat(w1,w3)", merged_shape)
            if w1[1] == "int8" and w3[1] == "int8":
                unpacked_merged_shape = (w1[0][0] + w3[0][0], w1[0][1] * 2)
                print_kv(f"expert {expert_id} unpacked cat(w1,w3)", unpacked_merged_shape)


if __name__ == "__main__":
    main()
