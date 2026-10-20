"""Reconstruct the four K26 training dependencies without changing inherited rollout sources."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_framework_sources import manifest_sources, patch_specs, prepare


def image_sources(inputs):
    training = json.loads((inputs / "docker/k26/sources.json").read_text())["sources"]
    if set(training) != {"Megatron-LM", "Megatron-Bridge", "MindSpeed"}:
        raise ValueError("K26 source manifest must contain Core, Bridge and MindSpeed")
    verl = manifest_sources(json.loads((inputs / "dependencies/verl-sources.json").read_text()), "k26")["verl"]
    return {**training, "verl": verl}


def prepare_image(root, inputs):
    sources = image_sources(inputs)
    # Check every patch and destination before fetching the first dependency.
    for name, source in sources.items():
        manifest = inputs / ("dependencies/verl-sources.json" if name == "verl" else "docker/k26/sources.json")
        if (root / name).exists():
            raise RuntimeError(f"Refusing to overwrite existing source: {root / name}")
        for filename, expected in patch_specs(source):
            patch = manifest.parent / filename
            if hashlib.sha256(patch.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Patch checksum mismatch: {patch}")
    prepare(root, inputs / "docker/k26/sources.json")
    prepare(root, inputs / "dependencies/verl-sources.json", "k26")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    prepare_image(args.root, args.inputs)
