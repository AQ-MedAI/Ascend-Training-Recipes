#!/usr/bin/env python3
"""Prepare a small BF16 DeepSeek-V4-Flash HF checkpoint for smoke tests.

The original DeepSeek-V4-Flash checkpoint stores attention weights in FP8/MXFP8
and routed expert weights in packed MXFP4 int8 tensors.  Atlas A3 test machines
used here only support the BF16 runtime path, so this helper materializes a
cut-layer BF16 HF checkpoint before launching verl/vLLM.

It intentionally converts only the tensors needed by the reduced model:
embeddings, lm head/final norms, and layers [0, num_layers).  Scale tensors and
MTP/full higher-layer tensors are omitted from the output index.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def unpack_mxfloat4_to_fp32(packed_tensor: torch.Tensor) -> torch.Tensor:
    e2m1_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed_tensor.device,
    )
    low_4bits = packed_tensor & 0x0F
    high_4bits = (packed_tensor // 16) & 0x0F
    unpacked = torch.stack([low_4bits, high_4bits], dim=-1)
    fp32_tensor = e2m1_values[unpacked.long()]
    new_shape = list(packed_tensor.shape)
    new_shape[-1] *= 2
    return fp32_tensor.view(*new_shape)


def weight_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    is_mx: bool = False,
) -> torch.Tensor:
    weight = weight.to(torch.float32)
    scale = scale.to(torch.float32)
    if is_mx:
        scale_expanded = scale.repeat_interleave(block_size, dim=1)
    else:
        scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[: weight.shape[0], : weight.shape[1]]
    return (weight * scale_expanded).to(torch.bfloat16)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_layer_id(name: str) -> int | None:
    if name.startswith("model.layers."):
        parts = name.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            return int(parts[2])
    if name.startswith("layers."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
    return None


def should_keep_tensor(name: str, num_layers: int) -> bool:
    if name.endswith(".scale") or name.endswith("_scale_inv"):
        return False
    if name.startswith("mtp.") or name.startswith("model.mtp."):
        return False
    layer_id = parse_layer_id(name)
    if layer_id is not None:
        return layer_id < num_layers
    return True


def scale_name_for(weight_name: str, weight_map: dict[str, str]) -> str | None:
    candidates = [
        weight_name.replace(".weight", ".scale"),
        f"{weight_name}_scale_inv",
    ]
    for candidate in candidates:
        if candidate in weight_map:
            return candidate
    return None


def copy_sidecar_files(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if item.suffix == ".safetensors":
            continue
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item, target_is_directory=item.is_dir())


def is_complete_output(dst: Path, src: Path, num_layers: int) -> bool:
    marker_path = dst / ".verl_bf16_cut_manifest.json"
    config_path = dst / "config.json"
    index_path = dst / "model.safetensors.index.json"
    if not all(path.is_file() for path in (marker_path, config_path, index_path)):
        return False

    try:
        marker = load_json(marker_path)
        config = load_json(config_path)
        index = load_json(index_path)
        weight_map = index["weight_map"]

        if not isinstance(weight_map, dict) or not weight_map:
            return False
        if not all(isinstance(filename, str) and filename for filename in weight_map.values()):
            return False
        if Path(marker["source"]).resolve() != src:
            return False
        if int(marker["num_layers"]) != num_layers:
            return False
        if int(marker["num_tensors"]) != len(weight_map):
            return False
        if config.get("_verl_cut_num_layers") != num_layers:
            return False
        if Path(config["_verl_cut_source"]).resolve() != src:
            return False
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False

    try:
        return all(
            (dst / filename).is_file() and (dst / filename).stat().st_size > 0
            for filename in set(weight_map.values())
        )
    except OSError:
        return False


def prepare_output_dir(dst: Path, src: Path, num_layers: int, force: bool) -> bool:
    if not dst.exists():
        dst.mkdir(parents=True)
        return True
    marker = dst / ".verl_bf16_cut_manifest.json"
    safe_default_overlay = dst.name.startswith("dsv4_flash_bf16_")
    if marker.exists() and not force and is_complete_output(dst, src, num_layers):
        return False
    if marker.exists() or (force and safe_default_overlay):
        shutil.rmtree(dst)
        dst.mkdir(parents=True)
        return True
    raise FileExistsError(
        f"Output directory already exists and was not created by this helper: {dst}. "
        "Set --force to replace it, or choose another --output-path."
    )


def adjust_cut_config(config: dict, num_layers: int) -> None:
    original_num_layers = config.get("num_hidden_layers")
    config["num_hidden_layers"] = num_layers
    config["num_layers"] = num_layers
    config["num_nextn_predict_layers"] = 0
    config["mtp_num_layers"] = 0

    for key, value in list(config.items()):
        if isinstance(value, list) and isinstance(original_num_layers, int) and len(value) == original_num_layers:
            config[key] = value[:num_layers]

    for key in ("num_hash_layers", "n_hash_layers"):
        if isinstance(config.get(key), int):
            config[key] = min(config[key], num_layers)

    config.pop("quantization_config", None)
    config["_verl_cut_num_layers"] = num_layers
    config["_verl_cut_source"] = str(config.get("_verl_cut_source", ""))
    config["_verl_auto_dequant_bf16"] = True


def materialize_bf16_cut(src: Path, dst: Path, num_layers: int, force: bool) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        raise ValueError("Input and output model paths must be different")
    if not src.exists():
        raise FileNotFoundError(f"Input model path does not exist: {src}")

    config_path = src / "config.json"
    index_path = src / "model.safetensors.index.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"model.safetensors.index.json not found: {index_path}")

    if not prepare_output_dir(dst, src=src, num_layers=num_layers, force=force):
        print(f"Reusing complete BF16 cut model: {dst}")
        return

    config = load_json(config_path)
    index = load_json(index_path)
    weight_map: dict[str, str] = index["weight_map"]

    keep_names = [name for name in weight_map if should_keep_tensor(name, num_layers)]
    file_to_names: dict[str, list[str]] = defaultdict(list)
    for name in keep_names:
        file_to_names[weight_map[name]].append(name)

    loaded_files: dict[str, dict[str, torch.Tensor]] = {}

    def get_tensor(name: str) -> torch.Tensor:
        filename = weight_map[name]
        if filename not in loaded_files:
            loaded_files[filename] = load_file(src / filename, device="cpu")
        return loaded_files[filename][name]

    new_weight_map: dict[str, str] = {}
    torch.set_default_dtype(torch.bfloat16)
    try:
        for filename in sorted(file_to_names):
            state_dict = load_file(src / filename, device="cpu")
            loaded_files[filename] = state_dict
            new_state_dict = {}
            for name in file_to_names[filename]:
                weight = state_dict[name]
                scale_name = scale_name_for(name, weight_map)
                if weight.element_size() == 1 and scale_name is not None:
                    scale = get_tensor(scale_name)
                    if weight.dtype == torch.int8:
                        weight = unpack_mxfloat4_to_fp32(weight.view(dtype=torch.uint8))
                        weight = weight_dequant(weight, scale, block_size=32, is_mx=True)
                    else:
                        weight = weight_dequant(weight, scale)
                else:
                    weight = weight.to(torch.bfloat16) if weight.is_floating_point() else weight
                new_state_dict[name] = weight
                new_weight_map[name] = filename

            save_file(new_state_dict, dst / filename, metadata={"format": "pt"})

            if len(loaded_files) > 2:
                for old_filename in list(loaded_files)[:-2]:
                    loaded_files.pop(old_filename, None)
    finally:
        torch.set_default_dtype(torch.float32)

    copy_sidecar_files(src, dst)

    adjust_cut_config(config, num_layers)
    config["_verl_cut_num_layers"] = num_layers
    config["_verl_cut_source"] = str(src)

    dump_json(dst / "config.json", config)
    dump_json(dst / "model.safetensors.index.json", {"metadata": {"format": "pt"}, "weight_map": new_weight_map})
    dump_json(
        dst / ".verl_bf16_cut_manifest.json",
        {
            "source": str(src),
            "num_layers": num_layers,
            "num_tensors": len(new_weight_map),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    materialize_bf16_cut(
        src=Path(args.input_path),
        dst=Path(args.output_path),
        num_layers=args.num_layers,
        force=args.force,
    )
    print(Path(args.output_path).resolve())


if __name__ == "__main__":
    main()
