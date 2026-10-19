"""Exercise patched vocabulary embedding with CPU tensors and collective stand-ins."""

import ast
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "mindspeed/core/tensor_parallel/vocab_parallel/vocab_parallel.py"


class VocabEmbeddingGroupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / RELATIVE
            target.parent.mkdir(parents=True)
            target.write_text((ROOT / "third_party/MindSpeed" / RELATIVE).read_text())
            subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "apply",
                    str(ROOT / "patches/mindspeed/k26/ascend-vocab-embedding-group.patch"),
                ],
                check=True,
            )
            function = next(node for node in ast.parse(target.read_text()).body if isinstance(node, ast.FunctionDef))
            namespace = {"F": F, "Callable": object}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "<patched embedding>", "exec"), namespace)
            cls.forward = staticmethod(namespace["vocab_parallel_embedding_forward_impl"])

    def test_explicit_group_masking_and_gradients(self):
        group = SimpleNamespace(size=lambda: 8)
        for deterministic in (False, True):
            weight = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
            module = SimpleNamespace(
                tp_group=group,
                vocab_start_index=2,
                vocab_end_index=4,
                weight=weight,
                deterministic_mode=deterministic,
                reduce_scatter_embeddings=False,
            )
            calls = []

            def reduce(tensor, **kwargs):
                calls.append(kwargs)
                return tensor

            output = self.forward(module, torch.tensor([[0, 2, 3, 5]]), reduce, None)
            torch.testing.assert_close(
                output, torch.stack([torch.zeros(3), weight[0], weight[1], torch.zeros(3)]).unsqueeze(0)
            )
            output.sum().backward()
            torch.testing.assert_close(weight.grad, torch.ones_like(weight))
            self.assertIs(calls[0]["group"], group)

    def test_sequence_scatter_uses_explicit_group(self):
        group = SimpleNamespace(size=lambda: 1)
        module = SimpleNamespace(
            tp_group=group,
            weight=torch.arange(6.0).reshape(2, 3),
            deterministic_mode=False,
            reduce_scatter_embeddings=True,
        )
        calls = []

        def scatter(tensor, **kwargs):
            calls.append((tensor.shape, kwargs))
            return tensor

        output = self.forward(module, torch.tensor([[0, 1]]), None, scatter)
        self.assertEqual(output.shape, (2, 1, 3))
        self.assertIs(calls[0][1]["group"], group)

    def test_legacy_module_and_collective(self):
        module = SimpleNamespace(
            tensor_model_parallel_size=1,
            weight=torch.arange(6.0).reshape(2, 3),
            deterministic_mode=False,
            reduce_scatter_embeddings=False,
        )
        output = self.forward(module, torch.tensor([[1]]), lambda tensor: tensor, None)
        torch.testing.assert_close(output, module.weight[1].reshape(1, 1, 3))
