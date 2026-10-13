#!/usr/bin/env python3
"""Materialize the first decoder layers of an unquantized Kimi K2.6 checkpoint."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

LAYER = re.compile(r"^(?:model\.|language_model\.(?:model\.)?|model\.language_model\.(?:model\.)?)?layers\.(\d+)\.")
MTP = re.compile(r"(?:^|\.)(?:mtp|nextn_predict_layers|nextn_layers)(?:\.|$)")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def layer_id(name):
    match = LAYER.match(name)
    return int(match[1]) if match else None


def weight_map(root):
    index = root / "model.safetensors.index.json"
    if index.is_file():
        result = read_json(index)["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as f:
            result = {key: "model.safetensors" for key in f.keys()}
    if not result:
        raise ValueError("Checkpoint contains no weights")
    for filename in set(result.values()):
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or external weight shard: {filename}")
    return result


def validate(root, count):
    config = read_json(root / "config.json")
    text = config.get("text_config", config)
    if text.get("num_hidden_layers") != count:
        raise ValueError(f"Expected {count} decoder layers in {root}")
    weights = weight_map(root)
    layers = {layer_id(k) for k in weights if layer_id(k) is not None}
    if layers != set(range(count)) or any(MTP.search(k) for k in weights):
        raise ValueError(f"Unexpected decoder/MTP weights in {root}: {sorted(layers)}")
    groups = defaultdict(set)
    for name, filename in weights.items():
        groups[filename].add(name)
    for filename, names in groups.items():
        with safe_open(root / filename, framework="pt", device="cpu") as f:
            if not names.issubset(f.keys()):
                raise ValueError(f"Index references missing tensors in {filename}")


def prepare(src, dst, count=4):
    if count < 1:
        raise ValueError("num-layers must be positive")
    dst = dst.absolute()
    if dst.exists():
        validate(dst, count)
        print(f"Reusing validated checkpoint: {dst}", flush=True)
        return
    if src is None:
        raise ValueError("Output is missing; supply --input-path or FULL_MODEL_PATH")
    src = src.resolve()
    if dst.resolve().is_relative_to(src):
        raise ValueError("Output must be outside the source checkpoint directory")
    config = read_json(src / "config.json")
    text = config.get("text_config", config)
    original = text.get("num_hidden_layers")
    if not isinstance(original, int) or not 0 < count <= original:
        raise ValueError(f"Cannot select {count} layers from {original!r}")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Use an unquantized BF16/FP16/FP32 source; dequantization is not supported")
    weights = weight_map(src)
    groups = defaultdict(list)
    for name, filename in weights.items():
        layer = layer_id(name)
        if not MTP.search(name) and (layer is None or layer < count):
            groups[filename].append(name)
    selected_layers = {layer_id(n) for names in groups.values() for n in names if layer_id(n) is not None}
    if selected_layers != set(range(count)):
        raise ValueError(f"Unrecognized or incomplete decoder weights: {sorted(selected_layers)}")
    # Update only text settings; preserve the vision tower and multimodal projector.
    for key, value in list(text.items()):
        if isinstance(value, list) and len(value) == original:
            text[key] = value[:count]
    text["num_hidden_layers"] = count
    if "num_layers" in text:
        text["num_layers"] = count
    for scope in (config, text):
        for key in ("num_nextn_predict_layers", "mtp_num_layers", "mtp_num_hidden_layers"):
            if key in scope:
                scope[key] = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}.partial-", dir=dst.parent))
    try:
        new_map = {}
        total = 0
        for i, (filename, names) in enumerate(sorted(groups.items()), 1):
            target = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
            tensors = {}
            with safe_open(src / filename, framework="pt", device="cpu") as f:
                for name in names:
                    tensor = f.get_tensor(name)
                    if tensor.is_floating_point() and tensor.element_size() < 2:
                        raise ValueError(f"Quantized tensor requires dequantization: {name}")
                    tensors[name] = tensor
                    total += tensor.numel() * tensor.element_size()
                    new_map[name] = target
                save_file(tensors, staging / target, metadata={"format": "pt"})
            del tensors
            print(f"Wrote {i}/{len(groups)} shards", flush=True)
        # Copy tokenizer/configuration/custom-code assets, never source weight files.
        for item in src.iterdir():
            if item.is_file() and item.suffix in {".json", ".py", ".model", ".txt", ".tiktoken", ".jinja"}:
                if item.name == "config.json" or item.name.endswith(".index.json"):
                    continue
                shutil.copy2(item, staging / item.name)
        write_json(staging / "config.json", config)
        write_json(staging / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": new_map})
        write_json(staging / ".k26_cut_manifest.json", {"source": str(src), "num_layers": count})
        validate(staging, count)
        if dst.exists():
            raise FileExistsError(f"Output appeared during conversion: {dst}")
        staging.rename(dst)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Prepared checkpoint: {dst}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.input_path, args.output_path, args.num_layers)


if __name__ == "__main__":
    main()
