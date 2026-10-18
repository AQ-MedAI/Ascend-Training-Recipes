"""Record the recipe files bundled into the training image without Git metadata."""

import argparse
import hashlib
import json
from pathlib import Path


def record_recipe(root, output, revision="unspecified"):
    root = root.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError("Recipe receipt must be outside the recorded source tree")
    for forbidden in (".git", "third_party"):
        if (root / forbidden).exists():
            raise ValueError(f"Recipe snapshot contains a development checkout: {root / forbidden}")
    for required in (
        "tools/activate.sh",
        "recipes/deepseek-v4/dsv4_flash_rl.sh",
        "recipes/deepseek-v4.1/dsv41_flash_rl.sh",
        "workspace/rollout/dsv41/runtime/worker.py",
    ):
        if not (root / required).is_file():
            raise ValueError(f"Incomplete recipe snapshot: {required}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in recipe snapshot: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executable": bool(path.stat().st_mode & 0o111),
            }
    snapshot = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"schema_version": 1, "revision": revision, "snapshot_sha256": snapshot, "files": files}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Recorded {len(files)} recipe files: {snapshot}", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="unspecified")
    args = parser.parse_args()
    record_recipe(args.root, args.output, args.revision)
