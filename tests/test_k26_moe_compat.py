"""CPU checks for the K26 routing contract and lazy MindSpeed exports."""
from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
from typing import Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472"


def apply_patch(directory, patch_file):
    subprocess.run(["git", "-C", str(directory), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(directory), "apply", str(patch_file)], check=True)


@unittest.skipIf(torch is None, "CPU PyTorch is required for routing checks")
class RoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = Path(os.environ.get("K26_MEGATRON_CHECKOUT", str(ROOT.parent / "Megatron-LM-aq")))
        relative = "megatron/core/transformer/moe/moe_utils.py"
        result = subprocess.run(["git", "-C", str(checkout), "show", f"{COMMIT}:{relative}"],
                                capture_output=True, text=True)
        if result.returncode:
            raise unittest.SkipTest("Set K26_MEGATRON_CHECKOUT to the pinned Core checkout")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / relative
            target.parent.mkdir(parents=True)
            target.write_text(result.stdout)
            apply_patch(directory, ROOT / "patches/megatron-lm/k26/ascend-moe-routing.patch")
            names = {"get_capacity", "group_limited_topk", "topk_routing_with_score_function",
                     "apply_router_token_dropping", "topk_softmax_with_capacity"}
            functions = [node for node in ast.parse(target.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name in names]
            namespace = dict(torch=torch, math=math, Optional=Optional, Tuple=Tuple)
            exec(compile("from __future__ import annotations\n" + "\n".join(ast.unparse(n) for n in functions),
                         "<patched Core routing>", "exec"), namespace)
            cls.route = staticmethod(namespace["topk_softmax_with_capacity"])

    def test_probabilities_selection_counts_and_gradients(self):
        for score in ("softmax", "sigmoid"):
            for pre in (False, True):
                for topk in (1, 2):
                    for deterministic in (False, True):
                        with self.subTest(score=score, pre=pre, topk=topk, deterministic=deterministic):
                            logits = torch.tensor([[.2, -.4, 1., .7], [.6, 1.3, -.1, -.5]], requires_grad=True)
                            bias = torch.tensor([.8, -.4, .1, -.2]) if score == "sigmoid" else None
                            raw = (logits.sigmoid() if score == "sigmoid" else
                                   logits.softmax(-1) if pre else logits)
                            indices = (raw + bias if bias is not None else raw).topk(topk, dim=1).indices
                            selected = raw.gather(1, indices)
                            if score == "softmax" and not pre:
                                selected = selected.softmax(-1)
                            elif score == "sigmoid" and topk > 1:
                                selected = selected / selected.sum(-1, keepdim=True)
                            expected = torch.zeros_like(logits).scatter(1, indices, selected * 1.7)
                            expected_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
                            probs, routing, counts = self.route(logits, topk, score_function=score,
                                use_pre_softmax=pre, expert_bias=bias, scaling_factor=1.7,
                                deterministic_mode=deterministic)
                            torch.testing.assert_close(probs, expected)
                            torch.testing.assert_close(routing, expected_map)
                            torch.testing.assert_close(counts, expected_map.sum(0))
                            torch.testing.assert_close(torch.autograd.grad(probs.square().sum(), logits, retain_graph=True)[0],
                                                       torch.autograd.grad(expected.square().sum(), logits)[0])

    def test_capacity_preserves_pre_drop_counts_and_padding_contract(self):
        logits = torch.tensor([[5., 1., -2., -3.], [4., .5, -1., -2.],
                               [3., .1, -.5, -1.], [2., .2, -.2, -.3]])
        for policy in ("probs", "position"):
            for pad in (False, True):
                probs, routing, counts = self.route(logits, 1, capacity_factor=1.,
                    drop_policy=policy, pad_to_capacity=pad, use_pre_softmax=True)
                torch.testing.assert_close(counts, torch.tensor([4, 0, 0, 0]))
                self.assertEqual(routing[:, 0].sum().item(), 1)
                self.assertEqual(routing.sum().item(), 4 if pad else 1)
                self.assertTrue(torch.all(probs[~routing] == 0))
        probs, routing, counts = self.route(logits, 1, capacity_factor=100.)
        self.assertEqual(routing.sum().item(), 4)
        torch.testing.assert_close(counts, torch.tensor([4, 0, 0, 0]))
        with self.assertRaises(ValueError):
            self.route(logits, 1, capacity_factor=100., drop_policy="invalid")

    def test_group_limited_routing(self):
        logits = torch.tensor([[3., 2., -1., -2., .2, .1, -.4, -.5]])
        _, routing, counts = self.route(logits, 2, num_groups=2, group_topk=1)
        self.assertEqual(routing[0, :4].sum().item(), 2)
        self.assertEqual(routing[0, 4:].sum().item(), 0)
        torch.testing.assert_close(counts, routing.sum(0))


class LazyExportsTest(unittest.TestCase):
    def test_exports_load_only_on_request_and_keep_aliases(self):
        paths = ["mindspeed/core/transformer/moe/moe_feature/__init__.py",
                 "mindspeed/core/transformer/moe/moe_feature/fb_overlap/__init__.py"]
        checkout = ROOT / "third_party/MindSpeed"
        with tempfile.TemporaryDirectory() as directory:
            originals = {}
            for relative in paths:
                source = checkout / relative
                if not source.is_file():
                    self.skipTest("Prepare MindSpeed sources first")
                originals[relative] = ast.parse(source.read_text())
                target = Path(directory) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text())
            apply_patch(directory, ROOT / "patches/mindspeed/k26/ascend-lazy-moe-imports.patch")
            for relative in paths:
                namespace = {"__name__": "fixture.package"}
                with patch("importlib.import_module") as importer:
                    importer.return_value = types.SimpleNamespace(MLP="mlp", BaseMoELayer="base", linear_backward_wgrad_detach="backward")
                    exec(compile((Path(directory) / relative).read_text(), relative, "exec"), namespace)
                    importer.assert_not_called()
                    old_exports = {a.asname or a.name for n in originals[relative].body
                                   if isinstance(n, ast.ImportFrom) for a in n.names}
                    self.assertEqual(set(namespace["__all__"]), old_exports)
                    symbol = "MegatronBaseMoeLayer" if "moe_feature/__init__" in relative else "linear_backward_wgrad_detach"
                    value = namespace["__getattr__"](symbol)
                    self.assertEqual(value, "base" if symbol == "MegatronBaseMoeLayer" else "backward")
                    self.assertEqual(namespace[symbol], value)
                    with self.assertRaises(AttributeError):
                        namespace["__getattr__"]("unknown")

class GmmCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = ['mindspeed/features_manager/moe/gmm.py',
                 'mindspeed/core/transformer/moe/moe_feature/gmm/experts.py']
        cls.sources = {}
        with tempfile.TemporaryDirectory() as directory:
            for relative in paths:
                source = ROOT / 'third_party/MindSpeed' / relative
                if not source.is_file():
                    raise unittest.SkipTest('Prepare MindSpeed sources first')
                target = Path(directory) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text())
            apply_patch(directory, ROOT / 'patches/mindspeed/k26/ascend-gmm-api.patch')
            cls.sources = {relative: (Path(directory) / relative).read_text() for relative in paths}

    def test_constructor_forwards_explicit_groups_and_preserves_old_call(self):
        tree = ast.parse(self.sources['mindspeed/core/transformer/moe/moe_feature/gmm/experts.py'])
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        constructor = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        calls = []
        class Base:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))
        fixture = ast.ClassDef(name='Fixture', bases=[ast.Name(id='Base', ctx=ast.Load())],
                               keywords=[], body=[constructor], decorator_list=[])
        ns = {'Base': Base}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[])), '<actual GMM constructor>', 'exec'), ns)
        config = types.SimpleNamespace(gated_linear_unit=False)
        group = object()
        ns['Fixture'](4, config, pg_collection=group)
        self.assertEqual(calls[-1], ((4, config), {'pg_collection': group}))
        ns['Fixture'](4, config)
        self.assertEqual(calls[-1], ((4, config), {}))

    def test_modern_dispatch_does_not_register_legacy_patch(self):
        tree = ast.parse(self.sources['mindspeed/features_manager/moe/gmm.py'])
        cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == 'register_patches')
        imports = [n.module for n in ast.walk(method) if isinstance(n, ast.ImportFrom)]
        self.assertNotIn('mindspeed.core.transformer.moe.moe_feature.adaptor', imports)
        self.assertIn('mindspeed.core.transformer.moe.moe_feature.gmm.adapter', imports)
        class RemoveImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return ast.Pass()
        method = RemoveImports().visit(method)
        for legacy_available, dispatcher in [(False, 'alltoall'), (True, 'alltoall_seq'), (False, 'alltoall_seq')]:
            calls = []
            experts, activation, preprocess = object(), object(), object()
            ns = {'MindSpeedGmmExperts': experts, 'find_spec': lambda name: object() if legacy_available else None,
                  'mindspeed_groupedmlp_weighted_bias_swiglu_impl': activation, 'ascend_gmm_preprocess': preprocess}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual GMM registration>', 'exec'), ns)
            args = types.SimpleNamespace(moe_grouped_gemm=True, use_ascend_mc2=False, moe_token_dispatcher_type=dispatcher)
            manager = types.SimpleNamespace(register_patch=lambda *args: calls.append(args))
            if not legacy_available and dispatcher == 'alltoall_seq':
                with self.assertRaisesRegex(RuntimeError, 'alltoall_seq'):
                    ns['register_patches'](None, manager, args)
            else:
                ns['register_patches'](None, manager, args)
                legacy = [c for c in calls if 'legacy_a2a' in c[0]]
                self.assertEqual(len(legacy), int(legacy_available))
                self.assertIs(calls[0][1], experts)


if __name__ == "__main__":
    unittest.main()
