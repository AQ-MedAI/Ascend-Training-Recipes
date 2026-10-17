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
    report_dir = Path(os.environ.get("V41_IMAGE_METADATA", "/opt/ascend-training/manifest"))
    receipt = report_dir / "frameworks.json"
    if not receipt.is_file():
        raise SystemExit(f"Missing built V4.1 framework receipt: {receipt}; use the matching framework image")
    info = json.loads(receipt.read_text())
    if info.get("python_environment") != "single":
        raise SystemExit("Rebuild the V4.1 image with the unified Python environment")
    expected_rollout = registry["rollout"][args.profile]
    for name, key in [("vllm", "vllm"), ("vllm-ascend", "vllm_ascend")]:
        if info[name] != expected_rollout[key]["commit"]:
            raise SystemExit(f"Image {name} commit differs from the recipe")
    expected_megatron = json.loads((root / "dependencies/framework-sources.json").read_text())
    if info["megatron"] != expected_megatron["sources"]["Megatron-LM"]:
        raise SystemExit("Image Megatron source/patch receipt differs from the recipe")
    expected_sources = json.loads((root / "dependencies/framework-sources.json").read_text())
    if info.get("sources") != expected_sources:
        raise SystemExit("Image framework sources/patches differ; rebuild the shared image")
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
    if args.profile == "v41":
        dependency_root = Path(os.environ.get("DEPENDENCY_ROOT", "/opt/ascend-training"))
        ascend_root = Path(os.environ.get("VLLM_ASCEND_ROOT", str(dependency_root / "vllm-ascend")))
        for required in (
            "vllm_ascend/deepseek_v41_config.py",
            "vllm_ascend/models/deepseek_v41/engram_hash.py",
            "vllm_ascend/core/deepseek_v41.py",
            "vllm_ascend/attention/dsa_v41.py",
            "vllm_ascend/patch/platform/patch_deepseek_v41_frontend/tokenizer.py",
        ):
            if not (ascend_root / required).is_file():
                raise SystemExit(
                    f"Shared framework sources lack V4.1 adaptation: {ascend_root / required}; "
                    "rebuild the shared image with the V4.1 backport patches"
                )
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
