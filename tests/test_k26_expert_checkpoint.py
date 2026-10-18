"""Verify the legacy expert checkpoint adapter without importing NPU modules."""

import ast
import os
import subprocess
import tempfile
import types
import unittest
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472"


class K26ExpertCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = Path(os.environ.get("K26_MEGATRON_CHECKOUT", str(ROOT.parent / "Megatron-LM-aq")))
        relative = "megatron/core/transformer/moe/experts.py"
        result = subprocess.run(["git", "-C", str(checkout), "show", f"{COMMIT}:{relative}"],
                                capture_output=True, text=True)
        if result.returncode:
            raise unittest.SkipTest("Provide K26 baseline via K26_MEGATRON_CHECKOUT")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / relative
            target.parent.mkdir(parents=True)
            target.write_text(result.stdout)
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            subprocess.run(["git", "-C", directory, "apply", str(ROOT / "patches/megatron-lm/k26/ascend-expert-checkpoint.patch")], check=True)
            tree = ast.parse(target.read_text())
            cls.function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name == "expert_dist_ckpt_decorator")

    def setUp(self):
        self.attention_group = object()
        self.expert_group = object()
        self.state = types.SimpleNamespace(
            _MPU_TENSOR_MODEL_PARALLEL_RANK=3,
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=16,
            _TENSOR_MODEL_PARALLEL_GROUP=self.attention_group,
            get_expert_tensor_parallel_rank=lambda: 1,
            get_expert_tensor_parallel_world_size=lambda: 2,
            get_expert_tensor_parallel_group=lambda: self.expert_group,
        )
        namespace = {"parallel_state": self.state, "wraps": wraps}
        exec(compile(ast.Module(body=[self.function], type_ignores=[]), "<actual compatibility decorator>", "exec"), namespace)
        self.decorate = namespace["expert_dist_ckpt_decorator"]

    def snapshot(self):
        return (self.state._MPU_TENSOR_MODEL_PARALLEL_RANK,
                self.state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE,
                self.state._TENSOR_MODEL_PARALLEL_GROUP)

    def assert_restored(self):
        self.assertEqual(self.snapshot(), (3, 16, self.attention_group))

    def test_expert_state_arguments_result_and_restoration(self):
        @self.decorate
        def checkpoint(value, *, suffix):
            self.assertEqual(self.snapshot(), (1, 2, self.expert_group))
            return value + suffix
        self.assertEqual(checkpoint("model", suffix=".weights"), "model.weights")
        self.assertEqual(checkpoint.__name__, "checkpoint")
        self.assert_restored()

    def test_failed_checkpoint_restores_all_state(self):
        failure = RuntimeError("checkpoint failed")
        @self.decorate
        def checkpoint():
            raise failure
        with self.assertRaises(RuntimeError) as caught:
            checkpoint()
        self.assertIs(caught.exception, failure)
        self.assert_restored()

    def test_nested_checkpoints_restore_the_enclosing_state(self):
        @self.decorate
        def inner():
            self.assertEqual(self.snapshot(), (0, 2, self.expert_group))
        @self.decorate
        def outer():
            self.state.get_expert_tensor_parallel_rank = lambda: 0
            inner()
            self.assertEqual(self.snapshot(), (1, 2, self.expert_group))
        outer()
        self.assert_restored()

    def test_failed_group_lookup_restores_partially_changed_state(self):
        def fail():
            raise RuntimeError("group unavailable")
        self.state.get_expert_tensor_parallel_group = fail
        with self.assertRaisesRegex(RuntimeError, "group unavailable"):
            self.decorate(lambda: None)()
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
