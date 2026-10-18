"""Prepare the seven pinned dependencies shared by the V4 and V4.1 image."""

import argparse
import json
from pathlib import Path

from prepare_framework_sources import manifest_sources, prepare


def image_sources(inputs):
    """Return framework and training receipts, rejecting divergent profiles."""
    frameworks = json.loads((inputs / "dependencies/framework-sources.json").read_text())
    registry = json.loads((inputs / "dependencies/registry.json").read_text())["components"]
    training = {}
    for name, component in registry.items():
        manifest = json.loads((inputs / component["source_manifest"]).read_text())
        source = manifest_sources(manifest, "v41")[name]
        if source != manifest_sources(manifest, "v4")[name]:
            raise RuntimeError(f"V4/V4.1 image sources differ: {name}")
        if source["repository"] != component["repository"] or any(
            source["commit"] != component["profiles"][profile]["commit"] for profile in ("v4", "v41")
        ):
            raise RuntimeError(f"Registry and source manifest disagree: {name}")
        training[name] = source
    overlap = set(training) & set(frameworks["sources"])
    if overlap:
        raise RuntimeError(f"Duplicate image source destinations: {sorted(overlap)}")
    return frameworks, training


def prepare_image(root, inputs):
    image_sources(inputs)  # Reject incompatible combinations before fetching anything.
    prepare(root, inputs / "dependencies/framework-sources.json")
    registry = json.loads((inputs / "dependencies/registry.json").read_text())["components"]
    for component in registry.values():
        prepare(root, inputs / component["source_manifest"], "v41")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prepare_image(args.root, args.inputs)
