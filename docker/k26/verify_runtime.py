"""Verify image sources and editable installations without importing NPU modules."""

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

INPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(INPUT / "tools"))
from prepare_framework_sources import source_commit, verify  # noqa: E402
from prepare_k26_image_sources import image_sources  # noqa: E402

REPLACED_PACKAGES = {"megatron-core", "megatron-bridge", "mindspeed", "verl", "mindspeed-llm", "mbridge"}
PACKAGES = {
    "Megatron-LM": "megatron-core",
    "Megatron-Bridge": "megatron-bridge",
    "MindSpeed": "mindspeed",
    "verl": "verl",
}


def versions():
    return {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }


def check_base(root, inputs):
    if sys.version_info[:3] != (3, 11, 15):
        raise RuntimeError(f"K26 requires the Python 3.11.15 runtime: {sys.version}")
    installed = versions()
    expected = {
        "torch": "2.10.0",
        "torch-npu": "2.10.0.post4",
        "triton": "3.5.0",
        "triton-ascend": "3.2.2",
        "numpy": "1.26.4",
        "datasets": "5.0.1",
        "pyarrow": "24.0.0",
        "transformers": "5.8.1",
    }
    for name, wanted in expected.items():
        if installed.get(name, "").split("+", 1)[0] != wanted:
            raise RuntimeError(f"Base runtime mismatch: {name}: expected {wanted}, got {installed.get(name)}")
    frameworks = json.loads((inputs / "dependencies/framework-sources.json").read_text())["sources"]
    for name, package in (("vllm", "vllm"), ("vllm-ascend", "vllm_ascend")):
        checkout = root / "third_party" / name
        source = frameworks[name]
        if source_commit(checkout) != source["commit"]:
            raise RuntimeError(f"Inherited rollout source pin differs: {checkout}")
        # Native builds add generated files; every recorded source file must still match.
        for relative, expected_file in source["files"].items():
            path = checkout / relative
            if expected_file.get("kind") == "symlink":
                actual = {"kind": "symlink", "target": os.readlink(path)}
            else:
                actual = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "executable": bool(path.stat().st_mode & 0o111),
                }
            if actual != expected_file:
                raise RuntimeError(f"Inherited rollout source changed: {path}")
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(checkout):
            raise RuntimeError(f"Inherited {package} does not resolve to {checkout}")
    return installed


def check_training(root, inputs):
    sources = image_sources(inputs)
    for name, source in sources.items():
        checkout = root / "third_party" / name
        verify(checkout, source, tracked=True)
        if source_commit(checkout) != source["commit"]:
            raise RuntimeError(f"Training source pin differs: {checkout}")
        distribution = metadata.distribution(PACKAGES[name])
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        location = urlsplit(direct.get("url", ""))
        if (
            location.scheme != "file"
            or Path(unquote(location.path)).resolve() != checkout.resolve()
            or not direct.get("dir_info", {}).get("editable")
        ):
            raise RuntimeError(f"Expected editable {PACKAGES[name]} at {checkout}")
        # Parse adapted Python files without triggering device imports at build time.
        base = inputs / ("dependencies" if name == "verl" else "docker/k26")
        for patch in source["patches"]:
            for line in (base / patch["path"]).read_text().splitlines():
                if line.startswith("+++ b/") and line.endswith(".py"):
                    path = checkout / line[6:]
                    ast.parse(path.read_text(), filename=str(path))
        print(f"[verify] {name}: source inventory, pin, editable installation and syntax passed", flush=True)
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/ascend-training"))
    parser.add_argument("--base", action="store_true")
    args = parser.parse_args()
    root = args.root
    installed = check_base(root, INPUT)
    if args.base:
        (INPUT / "base-package-versions.json").write_text(json.dumps(installed, indent=2) + "\n")
        print("[verify] Reusing the pinned V4/V4.1 Python and compiled rollout runtime", flush=True)
        return
    sources = check_training(root, INPUT)
    before = json.loads((INPUT / "base-package-versions.json").read_text())
    before = {k: v for k, v in before.items() if k not in REPLACED_PACKAGES}
    after = {k: v for k, v in installed.items() if k not in REPLACED_PACKAGES}
    if before != after:
        raise RuntimeError("Inherited package versions changed while installing K26 sources")
    manifest = root / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)
    for name in ("frameworks.json", "runtime-environment.json", "recipe.json"):
        previous = manifest / name
        if previous.exists():
            previous.replace(manifest / ("base-v4-v41-" + name))
    for name in ("dependencies", "patches", "tools", "docker"):
        if (INPUT / name).exists():
            shutil.copytree(INPUT / name, manifest / name, dirs_exist_ok=True)
    shutil.copyfile(INPUT / "base-package-versions.json", manifest / "base-package-versions.json")
    inherited = json.loads((INPUT / "dependencies/framework-sources.json").read_text())["sources"]
    receipt = {
        "profile": "k26",
        "source_root": str(root / "third_party"),
        "sources": {**sources, **{k: inherited[k] for k in ("vllm", "vllm-ascend")}},
        "image_smoke_test": "pending",
    }
    (manifest / "frameworks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (manifest / "runtime-environment.json").open("w") as output:
        subprocess.run([sys.executable, str(INPUT / "collect_environment.py")], check=True, stdout=output)
    print("[verify] K26 image source and environment receipts recorded; run NPU smoke tests after building", flush=True)


if __name__ == "__main__":
    main()
