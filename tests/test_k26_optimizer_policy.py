"""Check launcher policy forwarding and the pinned Core CPU-offload dispatch."""

import ast
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import unittest
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
POLICY = [
    "OPTIMIZER_CPU_OFFLOAD",
    "OPTIMIZER_OFFLOAD_FRACTION",
    "OVERLAP_CPU_OPTIMIZER_D2H_H2D",
    "USE_PRECISION_AWARE_OPTIMIZER",
    "SEQUENCE_PARALLEL",
]


class OptimizerPolicyTest(unittest.TestCase):
    def arguments(self, overrides=None):
        source = (ROOT / "recipes/kimi-k2.6/kimi_k26_rl.sh").read_text()
        block = source[source.index("ACTOR=(") : source.index("# Resolve endpoint layers")]
        names = set(re.findall(r"\$\{([A-Z_]+)", block))
        harness = "\n".join(f"{name}=1" for name in sorted(names - set(POLICY))) + "\n"
        harness += (
            "\n".join(line for line in source.splitlines() if any(line.startswith(name + "=") for name in POLICY))
            + "\n"
        )
        harness += block + '\nprintf "%s\\n" "${ACTOR[@]}"\n'
        env = {"PATH": os.environ["PATH"], **(overrides or {})}
        result = subprocess.run(["bash", "-eu", "-c", harness], env=env, text=True, capture_output=True, check=True)
        return dict(line.lstrip("+").split("=", 1) for line in result.stdout.splitlines())

    def test_safe_defaults_and_explicit_overrides_reach_core_config(self):
        prefix = "actor_rollout_ref.actor.optim.override_optimizer_config."
        values = self.arguments()
        expected = {
            "optimizer_cpu_offload": "True",
            "optimizer_offload_fraction": "1",
            "overlap_cpu_optimizer_d2h_h2d": "False",
            "use_precision_aware_optimizer": "False",
        }
        for key, value in expected.items():
            self.assertEqual(values[prefix + key], value)
        values = self.arguments({"OPTIMIZER_CPU_OFFLOAD": "False", "OPTIMIZER_OFFLOAD_FRACTION": "0.5"})
        self.assertEqual(values[prefix + "optimizer_cpu_offload"], "False")
        self.assertEqual(values[prefix + "optimizer_offload_fraction"], "0.5")

    def test_native_core_selects_hybrid_cpu_optimizer_policy(self):
        path = ROOT.parent / "Megatron-LM-aq/megatron/core/optimizer/__init__.py"
        if not path.is_file():
            self.skipTest("Provide pinned K26 Core source")
        tree = ast.parse(path.read_text())
        branch = next(
            n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "config.optimizer_cpu_offload"
        )
        calls = []

        def hybrid(groups, **kwargs):
            calls.append(kwargs)
            return object()

        config = SimpleNamespace(
            optimizer="adam",
            optimizer_cpu_offload=True,
            decoupled_weight_decay=True,
            use_torch_optimizer_for_cpu_offload=False,
            lr=0.001,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            optimizer_offload_fraction=1,
            overlap_cpu_optimizer_d2h_h2d=False,
            pin_cpu_grads=True,
            pin_cpu_params=True,
        )
        ns = dict(
            config=config,
            torch=torch,
            warnings=warnings,
            Adam=object(),
            CPUAdam=torch.optim.AdamW,
            HybridDeviceOptimizer=hybrid,
            param_groups=[{"params": []}],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(
                compile(ast.Module(body=branch.body, type_ignores=[]), "<actual Core CPU-offload dispatch>", "exec"), ns
            )
        self.assertIs(calls[0]["cpu_optimizer_cls"], torch.optim.AdamW)
        self.assertEqual(calls[0]["offload_fraction"], 1)
        self.assertFalse(calls[0]["overlap_cpu_optimizer_d2h_h2d"])
        self.assertTrue(calls[0]["param_update_in_fp32"])
