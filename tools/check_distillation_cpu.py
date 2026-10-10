"""Extract actual pure state/loss code; test CPU TP1 math, not framework integration."""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repository", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("--ref", help="Read a committed snapshot without switching branches")
args = parser.parse_args()
root = args.repository.resolve()
report_path = args.report
source_repository = str(root)
source_commit = None
snapshot = None
if args.ref:
    source_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", args.ref + "^{commit}"], text=True
    ).strip()
    snapshot = tempfile.TemporaryDirectory(prefix="aq-distillation-snapshot-")
    for rel in [
        "verl/experimental/teacher_loop/teacher_controller.py",
        "tests/experimental/teacher_loop/test_teacher_sleep_on_cpu.py",
        "verl/trainer/distillation/megatron/full_vocab_kl.py",
    ]:
        dest = Path(snapshot.name) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.check_output(["git", "-C", str(root), "show", source_commit + ":" + rel]))
    root = Path(snapshot.name)


def extract(path, names, env):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names), (path, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)


state = {"__name__": "aq_state_test", "Optional": Optional, "pytest": pytest}
extract(root / "verl/experimental/teacher_loop/teacher_controller.py", {"TeacherSleepState"}, state)
extract(root / "tests/experimental/teacher_loop/test_teacher_sleep_on_cpu.py", {"TestTeacherSleepState"}, state)
instance = state["TestTeacherSleepState"]()
passed = []
for name in dir(instance):
    if name.startswith("test_"):
        getattr(instance, name)()
        passed.append(name)
parallel = types.ModuleType("megatron.core.parallel_state")
# Populate the synthetic module used by the extracted TP1 implementation.
setattr(parallel, "get_tensor_model_parallel_group", lambda: dist.group.WORLD)
for n in ["megatron", "megatron.core"]:
    sys.modules[n] = types.ModuleType(n)
sys.modules["megatron.core.parallel_state"] = parallel
with tempfile.TemporaryDirectory() as tmp:
    dist.init_process_group("gloo", init_method="file://" + tmp + "/init", rank=0, world_size=1)
    try:
        env = {"torch": torch, "vocab_parallel_log_softmax": lambda x: torch.log_softmax(x.float(), dim=-1)}
        extract(
            root / "verl/trainer/distillation/megatron/full_vocab_kl.py",
            {"_VocabParallelFullVocabKL", "_condition_on_teacher_vocab"},
            env,
        )
        torch.manual_seed(42)
        for reverse in [False, True]:
            for teacher_vocab in [5, 8]:
                for chunk in [1, 4]:
                    x = torch.randn(2, 3, 8, requires_grad=True)
                    ref = x.detach().clone().requires_grad_()
                    hidden = torch.randn(2, 3, 4)
                    head = torch.randn(teacher_vocab, 4)
                    y = env["_VocabParallelFullVocabKL"].apply(x, hidden, head, chunk, reverse)
                    logq = torch.log_softmax(hidden @ head.T, dim=-1)
                    logp = torch.log_softmax(ref[..., :teacher_vocab] if reverse else ref, dim=-1)
                    expected = (
                        (logp.exp() * (logp - logq)).sum(-1)
                        if reverse
                        else (logq.exp() * (logq - logp[..., :teacher_vocab])).sum(-1)
                    )
                    torch.testing.assert_close(y, expected, atol=2e-6, rtol=2e-5)
                    # Other consumers mutate the original logits; saved input must remain valid.
                    with torch.no_grad():
                        x.add_(0.5)
                    y.sum().backward()
                    expected.sum().backward()
                    torch.testing.assert_close(x.grad, ref.grad, atol=2e-6, rtol=2e-5)
                    passed.append(f"full_vocab_reverse={reverse}_vocab={teacher_vocab}_chunk={chunk}")
    finally:
        dist.destroy_process_group()
report = {
    "status": "passed",
    "checks": len(passed),
    "cases": passed,
    "source": source_repository,
    "source_commit": source_commit,
    "scope": "extracted teacher LRU/pinning state + actual autograd KL forward/backward with CPU Gloo TP1; \
        torch log_softmax substitutes TP normalizer; no NPU/framework end-to-end acceptance",
}
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))

if snapshot is not None:
    snapshot.cleanup()
