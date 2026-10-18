"""Check MLA process-group forwarding without importing NPU modules."""
from __future__ import annotations
import ast
from functools import wraps
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
MLA = 'mindspeed/core/transformer/multi_head_latent_attention/adaptor.py'
DOT = 'mindspeed/core/transformer/dot_product_attention.py'


class MlaGroupsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = {}
        with tempfile.TemporaryDirectory() as directory:
            for relative in (MLA, DOT):
                source = ROOT / 'third_party/MindSpeed' / relative
                if not source.is_file():
                    raise unittest.SkipTest('Prepare MindSpeed sources first')
                target = Path(directory) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text())
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/mindspeed/k26/ascend-mla-process-groups.patch')], check=True)
            cls.trees = {relative: ast.parse((Path(directory) / relative).read_text()) for relative in (MLA, DOT)}

    def function(self, path, name, namespace):
        node = next(n for n in self.trees[path].body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile('from __future__ import annotations\n' + ast.unparse(node), '<actual patched initializer>', 'exec'), namespace)
        return namespace[name]

    def config(self, rope='yarn'):
        return SimpleNamespace(v_head_dim=128, num_attention_heads=64, qk_head_dim=128,
            qk_pos_emb_head_dim=64, recompute_granularity=None, recompute_modules=[],
            rotary_scaling_factor=32, mscale=1, mscale_all_dim=2, rope_type=rope,
            qk_pos_emb_head_dim_unused=0, rotary_percent=1., rotary_base=10000,
            original_max_position_embeddings=4096, max_position_embeddings=32768,
            beta_fast=32, beta_slow=1, apply_rope_fusion=False, hidden_size=7168,
            output_layer_init_method=None, add_bias_linear=False, cache_mla_latents=False)

    def test_mla_forwards_explicit_groups_and_uses_original_yarn_context(self):
        for rope in ('rope', 'yarn'):
            calls = []
            group = SimpleNamespace(tp=object(), cp=object())
            class Attention:
                def __init__(self, **kwargs):
                    calls.append(('attention', kwargs))
                    self.config = kwargs['config']
                    self.layer_number = kwargs['layer_number']
                    self.attn_mask_type = kwargs['attn_mask_type']
                    self.attention_type = kwargs['attention_type']
                    self.pg_collection = kwargs['pg_collection']
            def build(module, *args, **kwargs):
                calls.append((module, kwargs))
                return object()
            def rotary(*args, **kwargs):
                calls.append(('rotary', kwargs))
                return object()
            ns = dict(Attention=Attention, build_module=build, RotaryEmbedding=rotary,
                      YarnRotaryEmbedding=rotary, math=math, _yarn_get_mscale=lambda factor, dim: dim)
            function = self.function(MLA, 'multi_latent_attention_init_impl', ns)
            instance = SimpleNamespace()
            function(instance, self.config(rope), SimpleNamespace(core_attention='core', linear_proj='projection'),
                     1, 'causal', 'self', cp_comm_type='ring', pg_collection=group)
            by_name = dict(calls)
            self.assertIs(by_name['attention']['pg_collection'], group)
            self.assertIs(by_name['core']['pg_collection'], group)
            self.assertIs(by_name['projection']['tp_group'], group.tp)
            self.assertIs(by_name['rotary']['cp_group'], group.cp)
            self.assertFalse(instance.cache_mla_latents)
            self.assertAlmostEqual(instance.softmax_scale, 4 / math.sqrt(192))
            if rope == 'yarn':
                self.assertEqual(by_name['rotary']['original_max_position_embeddings'], 4096)

    def test_mla_dot_wrapper_preserves_explicit_group(self):
        calls = []
        wrapper = self.function(MLA, 'dot_product_attention_init_wrapper', dict(wraps=wraps, divide=lambda x, y: x // y))
        def inner(instance, config, *args, **kwargs):
            instance.config = config
            calls.append(kwargs)
        group = object()
        wrapped = wrapper(inner)
        wrapped(SimpleNamespace(), self.config(), 1, 'causal', 'self', pg_collection=group)
        self.assertIs(calls[-1]['pg_collection'], group)
        wrapped(SimpleNamespace(), self.config(), 1, 'causal', 'self')
        self.assertNotIn('pg_collection', calls[-1])

    def test_dot_initializer_uses_explicit_tp_size(self):
        class Base:
            def __init__(self, config):
                self.config = config
        class Dot(Base):
            pass
        args = SimpleNamespace(alibi_fusion_attn_type=None, multi_head_latent_attention=False)
        def global_tp_size():
            raise AssertionError('Explicit TP must not fall back to the global group')
        ns = dict(DotProductAttention=Dot, parallel_state=SimpleNamespace(get_tensor_model_parallel_world_size=global_tp_size),
                  divide=lambda x, y: x // y, get_args=lambda: args, math=math,
                  FusedScaleMaskSoftmax=lambda **kwargs: SimpleNamespace(), attention_mask_func=object(),
                  torch=SimpleNamespace(nn=SimpleNamespace(Dropout=lambda probability: object())))
        function = self.function(DOT, 'dot_product_attention_init', ns)
        config = SimpleNamespace(context_parallel_size=1, window_size=None, kv_channels=128,
            num_attention_heads=64, num_query_groups=64, apply_query_key_layer_scaling=False,
            fp16=False, bf16=True, masked_softmax_fusion=False, attention_softmax_in_fp32=False,
            attention_dropout=0.)
        group = SimpleNamespace(tp=SimpleNamespace(size=lambda: 8))
        instance = Dot.__new__(Dot)
        function(instance, config, 1, 'causal', 'self', pg_collection=group)
        self.assertIs(instance.pg_collection, group)
        self.assertEqual(instance.num_attention_heads_per_partition, 8)
        self.assertEqual(instance.hidden_size_per_partition, 1024)


if __name__ == '__main__':
    unittest.main()
