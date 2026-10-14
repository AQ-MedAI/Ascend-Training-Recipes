"""Check source versions and image inputs without importing NPU backends."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("v41", "v4"), default="v41")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "dependencies/registry.json").read_text())
    for name, component in registry["components"].items():
        checkout = root / "third_party" / name
        expected = component["profiles"][args.profile]["commit"]
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise SystemExit(f"{name}: expected {expected}, found {actual}; run setup_sources.sh {args.profile}")
    if args.profile == "v4":
        raise SystemExit(
            "V4 sources are pinned, but its complete image compatibility contract is not yet validated. "
            "Do not install them into the V4.1 image automatically."
        )
    report_dir = Path(os.environ.get("V41_IMAGE_METADATA", "/opt/ascend-training/manifest"))
    receipt = report_dir / "frameworks.json"
    if not receipt.is_file():
        raise SystemExit(f"Missing built V4.1 framework receipt: {receipt}; use the matching framework image")
    info = json.loads(receipt.read_text())
    if info.get("python_environment") != "single":
        raise SystemExit("Rebuild the V4.1 image with the unified Python environment")
    expected_rollout = registry["rollout"]["v41"]
    for name, key in [("vllm", "vllm"), ("vllm-ascend", "vllm_ascend")]:
        if info[name] != expected_rollout[key]["commit"]:
            raise SystemExit(f"Image {name} commit differs from the recipe")
    expected_megatron = json.loads((root / "docker/v41/megatron-source.json").read_text())
    if info["megatron"] != expected_megatron:
        raise SystemExit("Image Megatron source/patch receipt differs from the recipe")
    for name, version in {
        "torch": "2.10.0",
        "torch-npu": "2.10.0.post4",
        "triton-ascend": "3.2.2",
        "triton": "3.5.0",
        "numpy": "1.26.4",
        "transformers": "5.14.1",
        "fastapi": "0.123.10",
        "starlette": "0.50.0",
        "opencv-python-headless": "5.0.0.93",
        "prometheus-fastapi-instrumentator": "8.1.0",
        "mcp": "2.2.0",
        "model-hosting-container-standards": "0.1.16",
    }.items():
        if metadata.version(name).split("+")[0] != version:
            raise SystemExit(f"{name} must be {version}")
    engram = registry["components"]["MindSpeed-LLM"]["profiles"]["v41"]
    path = root / "third_party/MindSpeed-LLM" / engram["required_files"][0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != engram["engram_hash_sha256"]:
        raise SystemExit("Engram source checksum differs")
    rollout_python = Path(os.environ.get("ROLLOUT_PYTHON", sys.executable))
    rollout_prefix = subprocess.check_output(
        [str(rollout_python), "-c", "import sys; print(sys.prefix)"], text=True
    ).strip()
    if Path(rollout_prefix).resolve() != Path(sys.prefix).resolve():
        raise SystemExit("Training and rollout must use the same Python environment; unset ROLLOUT_PYTHON")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    if result.returncode:
        print("Dependency metadata conflicts reported above; fixed deployment versions are retained.")
    print("Source/image versions checked; NPU runtime acceptance is still required.")


if __name__ == "__main__":
    main()
