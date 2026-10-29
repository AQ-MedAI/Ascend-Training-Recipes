"""Reconstruct the patched runtime modules from the managed V4 profile."""
import atexit
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


@lru_cache(maxsize=1)
def source_root():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "dependencies/verl-sources.json").read_text())["profiles"]["v4"]["sources"]["verl"]
    checkout = root / "third_party/verl"
    baseline = subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", source["commit"] + "^{commit}"],
                              capture_output=True)
    if baseline.returncode:
        raise unittest.SkipTest("Initialize the pinned verl checkout before running V4 OPD regressions")
    temporary = tempfile.TemporaryDirectory(prefix="v4-opd-regression-")
    atexit.register(temporary.cleanup)
    target = Path(temporary.name)
    paths = ["verl/workers/config/distillation.py",
             "verl/workers/rollout/vllm_rollout/full_vocab_hidden_export.py",
             "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
             "verl/trainer/distillation/megatron/full_vocab_kl.py",
             "verl/utils/checkpoint/megatron_checkpoint_manager.py"]
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    for path in paths:
        result = subprocess.run(["git", "-C", str(checkout), "show", source["commit"] + ":" + path],
                                capture_output=True)
        if result.returncode == 0:
            file = target / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(result.stdout)
    for patch in source["patches"]:
        subprocess.run(["git", "-C", str(target), "apply", *["--include=" + p for p in paths],
                        str((root / "dependencies" / patch["path"]).resolve())], check=True)
    return target
