"""CPU tests for DAPO group selection and the actual main_ppo collection block."""
import ast
from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/verl/verl/trainer/ppo"
if not SOURCE.exists():
    SOURCE = ROOT / "verl/trainer/ppo"
spec = importlib.util.spec_from_file_location("tested_group_filter", SOURCE / "group_filter.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GroupFilter = module.GroupFilter


class Batch:
    """Small DataProto-compatible tensor fixture; no distributed workers required."""
    def __init__(self, uids, values, final=None):
        self.batch = {
            "token_level_scores": torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
            "token_level_rewards": torch.tensor(values if final is None else final, dtype=torch.float32).reshape(-1, 1),
            "attention_mask": torch.ones(len(uids), 3, dtype=torch.int64),
            "row_id": torch.arange(len(uids)),
        }
        self.non_tensor_batch = {"uid": np.asarray(uids), "acc": np.asarray(values)}
        self.meta_info = {"temperature": 1.0, "global_token_num": [3] * len(uids)}

    def __getitem__(self, rows):
        result = object.__new__(Batch)
        result.batch = {key: value[rows] for key, value in self.batch.items()}
        result.non_tensor_batch = {key: value[rows] for key, value in self.non_tensor_batch.items()}
        result.meta_info = dict(self.meta_info)
        return result

    @staticmethod
    def concat(parts):
        result = object.__new__(Batch)
        result.batch = {key: torch.cat([p.batch[key] for p in parts]) for key in parts[0].batch}
        result.non_tensor_batch = {
            key: np.concatenate([p.non_tensor_batch[key] for p in parts]) for key in parts[0].non_tensor_batch
        }
        assert all(not p.meta_info for p in parts)
        result.meta_info = {}
        return result


class GroupFilterTest(unittest.TestCase):
    def setUp(self):
        stub = types.ModuleType("verl")
        stub.DataProto = Batch
        self.modules = patch.dict(sys.modules, {"verl": stub})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def collector(self, target=2, n=2, metric="seq_reward", limit=10):
        return GroupFilter({"metric": metric, "max_num_gen_batches": limit}, target, n)

    def test_replenishment_keeps_complete_groups_and_aligned_rows(self):
        c = self.collector()
        self.assertIsNone(c.add(Batch(["a", "b", "a", "b"], [0, 2, 1, 2])))
        result = c.add(Batch(["c", "c", "d", "d"], [2, 4, 0, 0]))
        self.assertEqual(result.non_tensor_batch["uid"].tolist(), ["a", "a", "c", "c"])
        self.assertEqual(result.batch["token_level_scores"].flatten().tolist(), [0, 1, 2, 4])
        self.assertEqual(c.metrics()["filter_groups/num_gen_batches"], 2)
        self.assertEqual(c.dropped, 2)

    def test_all_constant_then_valid(self):
        c = self.collector(target=1)
        self.assertIsNone(c.add(Batch(["a", "a"], [1, 1])))
        self.assertEqual(c.parts, [])
        self.assertIsNotNone(c.add(Batch(["b", "b"], [0, 1])))

    def test_generation_limit_is_inclusive(self):
        c = self.collector(target=1, limit=2)
        self.assertIsNone(c.add(Batch(["a", "a"], [0, 0])))
        with self.assertRaisesRegex(ValueError, "max_num_gen_batches=2"):
            c.add(Batch(["b", "b"], [1, 1]))
        c = self.collector(target=1, limit=1)
        self.assertIsNotNone(c.add(Batch(["a", "a"], [0, 1])))

    def test_unlimited_nonpositive_limits(self):
        for limit in (0, -1):
            c = self.collector(target=1, limit=limit)
            for index in range(12):
                self.assertIsNone(c.add(Batch([str(index)] * 2, [0, 0])))
            self.assertIsNotNone(c.add(Batch(["last"] * 2, [0, 1])))

    def test_overflow_truncates_whole_interleaved_groups(self):
        c = self.collector(target=1)
        result = c.add(Batch(["a", "b", "a", "b"], [0, 2, 1, 3]))
        self.assertEqual(result.non_tensor_batch["uid"].tolist(), ["a", "a"])

    def test_singletons_follow_upstream_semantics(self):
        c = self.collector(n=1)
        result = c.add(Batch(["a", "b"], [0, 0]))
        self.assertEqual(len(result.non_tensor_batch["uid"]), 2)

    def test_final_reward_uses_penalized_rewards(self):
        c = self.collector(target=1, metric="seq_final_reward")
        result = c.add(Batch(["a", "a", "b", "b"], [1, 1, 0, 1], [0, 1, 2, 2]))
        self.assertEqual(result.non_tensor_batch["uid"].tolist(), ["a", "a"])

    def test_custom_metric_and_missing_metric(self):
        c = self.collector(target=1, metric="acc")
        batch = Batch(["a", "a"], [1, 1])
        batch.non_tensor_batch["acc"] = np.asarray([0, 1])
        self.assertIsNotNone(c.add(batch))
        with self.assertRaisesRegex(ValueError, "absent"):
            self.collector(metric="missing").add(batch)

    def test_nonfinite_and_nonscalar_metrics_are_rejected(self):
        for values in ([float("nan"), 1], [0, float("inf")], [[0], [1]]):
            batch = Batch(["a", "a"], [0, 1])
            batch.non_tensor_batch["acc"] = np.asarray(values)
            with self.assertRaisesRegex(ValueError, "finite scalar"):
                self.collector(metric="acc").add(batch)

    def test_incomplete_group_and_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.collector().add(Batch(["a"], [1]))
        with self.assertRaisesRegex(ValueError, "metric must be set"):
            GroupFilter({}, 2, 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.collector(n=0)

    def test_reset_drops_previous_step_state(self):
        c = self.collector(target=1)
        c.add(Batch(["a", "a"], [0, 1]))
        c.timing["gen"] = 7
        c.reset()
        self.assertEqual((c.groups, c.attempts, c.parts, c.timing), (0, 0, [], {}))

    def test_actual_trainer_collection_block_skips_update_until_full(self):
        # Execute the unmodified AST block from fit(), including its real `continue`.
        tree = ast.parse((SOURCE / "ray_trainer.py").read_text())
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "group_filter is not None"
                     and any(isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "filtered"
                             for n in node.body))
        wrapper = ast.parse('''
def run(self, batches, group_filter):
    metrics = {}
    timing_raw = {}
    for batch in batches:
        reward_extra_infos_dict = {"acc": batch.non_tensor_batch["acc"]}
        PLACEHOLDER
        return batch, metrics, reward_extra_infos_dict
''')
        wrapper.body[0].body[2].body[1] = block
        ast.fix_missing_locations(wrapper)
        namespace = {"marked_timer": lambda *args: nullcontext()}
        exec(compile(wrapper, "actual_trainer_collection_block", "exec"), namespace)
        trainer = types.SimpleNamespace(config=types.SimpleNamespace(trainer=types.SimpleNamespace(balance_batch=True)))
        resumed = []
        trainer.global_steps = 7
        trainer.checkpoint_manager = types.SimpleNamespace(update_weights=resumed.append)
        trainer._balance_batch = lambda batch, metrics: metrics.update({"balanced": True})
        result, metrics, extra = namespace["run"](trainer, [
            Batch(["a", "a", "b", "b"], [0, 1, 0, 0]),
            Batch(["c", "c"], [0, 1]),
        ], self.collector())
        self.assertEqual(result.non_tensor_batch["uid"].tolist(), ["a", "a", "c", "c"])
        self.assertEqual(result.meta_info["global_token_num"], [3, 3, 3, 3])
        self.assertEqual(result.meta_info["temperature"], 1)
        self.assertEqual(extra["acc"], [0, 1, 0, 1])
        self.assertEqual(metrics["filter_groups/num_gen_batches"], 2)
        self.assertTrue(metrics["balanced"])
        self.assertEqual(resumed, [7])


if __name__ == "__main__":
    unittest.main()
