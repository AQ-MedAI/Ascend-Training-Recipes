"""CPU execution of the patched apply adapter and pinned Core packed RoPE."""

import ast
import logging
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "mindspeed/core/fusions/fused_rope.py"


class ImportsForCpu(ast.NodeTransformer):
    """Replace only backend imports; execute the adapter and Core math unchanged."""

    def visit_Import(self, node):
        return ast.Assign(targets=[ast.Name(id="ru", ctx=ast.Store())], value=ast.Name(id="rope_utils", ctx=ast.Load()))

    def visit_ImportFrom(self, node):
        return ast.Assign(
            targets=[ast.Name(id="parallel_state", ctx=ast.Store())],
            value=ast.Name(id="default_parallel_state", ctx=ast.Load()),
        )


class ApplyRopeCpGroupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core = ROOT.parent / "Megatron-LM-aq/megatron/core/models/common/embeddings/rope_utils.py"
        if not core.is_file():
            raise unittest.SkipTest("Provide pinned K26 Megatron-LM-aq source")
        names = {
            "_rotate_half",
            "_apply_rotary_pos_emb_bshd",
            "_get_thd_freqs_on_this_cp_rank",
            "_apply_rotary_pos_emb_thd",
        }
        nodes = [n for n in ast.parse(core.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        for n in nodes:
            n.decorator_list = []
        cls.core_ns = {"torch": torch, "Tensor": torch.Tensor}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual Core RoPE>", "exec"), cls.core_ns)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / RELATIVE
            path.parent.mkdir(parents=True)
            path.write_text((ROOT / "third_party/MindSpeed" / RELATIVE).read_text())
            subprocess.run(
                ["git", "-C", directory, "apply", str(ROOT / "patches/mindspeed/k26/ascend-rope-apply-cp-group.patch")],
                check=True,
            )
            function = next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "apply_rotary_pos_emb"
            )
            function = ImportsForCpu().visit(function)
            cls.adapter_code = compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "<patched apply adapter>",
                "exec",
            )

    def adapter(self, group):
        ns = dict(
            Tensor=torch.Tensor,
            TransformerConfig=object,
            Optional=__import__("typing").Optional,
            logging=logging,
            __name__=__name__,
            HAVE_APPLY_ROPE_FUSION=True,
            rope_utils=SimpleNamespace(**self.core_ns),
            default_parallel_state=SimpleNamespace(get_context_parallel_group=lambda: group),
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Legacy packed fused kernel must not swallow the explicit CP group")

        ns["fused_apply_rotary_pos_emb_thd"] = forbidden
        exec(self.adapter_code, ns)
        return ns["apply_rotary_pos_emb"]

    def test_packed_group_ranks_scaling_and_fused_fallback(self):
        freqs = torch.arange(32, dtype=torch.float32).reshape(16, 1, 1, 2) / 20
        for rank, positions in [(0, [0, 1, 6, 7, 8, 9, 14, 15]), (1, [2, 3, 4, 5, 10, 11, 12, 13])]:
            group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
            t = torch.randn(8, 1, 2, requires_grad=True)
            f = freqs[positions, 0]
            rotated = torch.stack((-t[..., 1], t[..., 0]), dim=-1)
            expected = (t * f.cos() + rotated * f.sin()) * 1.25
            for fused in (False, True):
                config = SimpleNamespace(
                    apply_rope_fusion=fused, rotary_interleaved=False, multi_latent_attention=False
                )
                actual = self.adapter(group)(t, freqs, config, torch.tensor([0, 8, 16]), 1.25, cp_group=group)
                torch.testing.assert_close(actual, expected)
                grad = torch.autograd.grad(actual.sum(), t)[0]
                expected_grad = torch.autograd.grad(expected.sum(), t, retain_graph=True)[0]
                torch.testing.assert_close(grad, expected_grad)

    def test_default_group_and_nonpacked_call(self):
        group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
        apply = self.adapter(group)
        config = SimpleNamespace(apply_rope_fusion=False, rotary_interleaved=False, multi_latent_attention=False)
        t = torch.randn(4, 1, 2)
        freqs = torch.zeros(4, 1, 1, 2)
        torch.testing.assert_close(apply(t, freqs, config, torch.tensor([0, 4])), t)
        expanded = t.unsqueeze(1)
        torch.testing.assert_close(apply(expanded, freqs, config, cp_group=group), expanded)
