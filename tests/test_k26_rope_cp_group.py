from pathlib import Path
import ast, subprocess, tempfile
from types import SimpleNamespace
import torch

r = Path(__file__).resolve().parents[1]
p = r / "patches/mindspeed/k26/ascend-rope-cp-group.patch"
import unittest


class RopeCpGroupTest(unittest.TestCase):
    def test_patch_and_cp_slices(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in [
                "mindspeed/core/context_parallel/rotary_pos_embedding_utils.py",
                "mindspeed/features_manager/context_parallel/context_parallel_feature.py",
            ]:
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((r / "third_party/MindSpeed" / rel).read_bytes())
            subprocess.run(["git", "-C", d, "apply", "--check", str(p)], check=True)
            subprocess.run(["git", "-C", d, "apply", str(p)], check=True)
            source = (root / "mindspeed/core/context_parallel/rotary_pos_embedding_utils.py").read_text()
            functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
            args = SimpleNamespace(
                tp_y=1, context_parallel_algo="ulysses_cp_algo", attention_mask_type="causal", reset_position_ids=False
            )
            ns = dict(
                torch=torch,
                get_args=lambda: args,
                parallel_state=SimpleNamespace(
                    get_context_parallel_world_size=lambda: 2, get_context_parallel_rank=lambda: 0
                ),
            )
            exec(compile(ast.Module(body=functions, type_ignores=[]), "<patched RoPE>", "exec"), ns)
            x = torch.arange(16).reshape(8, 2)
            group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
            torch.testing.assert_close(ns["get_pos_emb_on_this_cp_rank"](x, 0, group), x[4:])
            torch.testing.assert_close(ns["get_pos_emb_on_this_cp_rank"](x, 0), x[:4])
            args.context_parallel_algo = "megatron_cp_algo"

            # CPU stand-in only for the index tensor's device transfer.
            class CpuIndex:
                def __init__(self, values):
                    self.values = values

                def cuda(self, non_blocking=False):
                    return torch.tensor(self.values)

            ns["torch"] = SimpleNamespace(tensor=lambda values, **kwargs: CpuIndex(values))
            torch.testing.assert_close(ns["get_pos_emb_on_this_cp_rank"](x, 0, group), x[2:6])
            args.context_parallel_algo = "hybrid_cp_algo"
            try:
                ns["get_pos_emb_on_this_cp_rank"](x, 0, group)
            except ValueError:
                pass
            else:
                raise AssertionError("Specialized mismatched group accepted")
