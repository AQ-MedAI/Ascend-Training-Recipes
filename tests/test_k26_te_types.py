"""Execute Core attention initialization with incomplete optional TE symbols."""
from __future__ import annotations
import ast
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE_COMMIT = '96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472'
ATTENTION = 'megatron/core/transformer/attention.py'
MLA = 'megatron/core/transformer/multi_latent_attention.py'


class OptionalTeTypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = ROOT.parent / 'Megatron-LM-aq'
        if not checkout.is_dir():
            raise unittest.SkipTest('Provide the K26 Megatron-LM-aq checkout')
        files = [ATTENTION, MLA, 'megatron/core/transformer/moe/shared_experts.py', 'megatron/core/fp8_utils.py']
        cls.original = {}
        with tempfile.TemporaryDirectory() as directory:
            for rel in files:
                text = subprocess.check_output(['git', '-C', str(checkout), 'show', CORE_COMMIT + ':' + rel], text=True)
                cls.original[rel] = ast.parse(text)
                target = Path(directory) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/megatron-lm/k26/ascend-optional-te-types.patch')], check=True)
            cls.patched = {rel: ast.parse((Path(directory) / rel).read_text()) for rel in files}
        relative = 'mindspeed/core/transformer/multi_head_latent_attention/adaptor.py'
        source = ROOT / 'third_party/MindSpeed' / relative
        if not source.is_file():
            raise unittest.SkipTest('Prepare MindSpeed sources first')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / relative
            target.parent.mkdir(parents=True)
            target.write_text(source.read_text())
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', '--include=' + relative, str(ROOT / 'patches/mindspeed/k26/ascend-mla-process-groups.patch')], check=True)
            cls.adapter = ast.parse(target.read_text())

    def constructor(self, tree, name):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')

    def build_chain(self, tel_type, patched=True):
        nodes = self.patched if patched else self.original
        calls = []
        class Base:
            def __init__(self, config):
                self.config = config
        class Column:
            pass
        def build(module, *args, **kwargs):
            calls.append((module, kwargs))
            return module(*args, **kwargs) if module is core_attention else SimpleNamespace()
        def core_attention(*, config, layer_number, attn_mask_type, attention_type,
                           cp_comm_type=None, softmax_scale=None, pg_collection=None):
            calls.append(('core', pg_collection))
            return SimpleNamespace()
        def rotary(*args, **kwargs):
            calls.append(('rotary', kwargs))
            return SimpleNamespace()
        ns = dict(Base=Base, build_module=build, HAVE_TE=True, TELinear=tel_type,
            TEColumnParallelLinear=Column, ColumnParallelLinear=Column, Linear=None,
            divide=lambda x, y: x // y, get_pg_size=lambda group: group.size(),
            math=math, _yarn_get_mscale=lambda factor, dim: 1.,
            RotaryEmbedding=rotary, YarnRotaryEmbedding=rotary,
            AttnMaskType=SimpleNamespace(padding='padding'),
            is_te_min_version=lambda version: True, set_save_original_input=lambda module: calls.append(('save', module)))
        attention_init = self.constructor(nodes[ATTENTION], 'Attention')
        core_cls = ast.ClassDef(name='Attention', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[attention_init], decorator_list=[])
        exec(compile('from __future__ import annotations\n' + ast.unparse(ast.fix_missing_locations(core_cls)), '<actual Core Attention constructor>', 'exec'), ns)
        # Execute the complete patched MindSpeed replacement initializer,
        # rather than replacing Attention with a stub as earlier tests did.
        adapter = next(n for n in self.adapter.body if isinstance(n, ast.FunctionDef) and n.name == 'multi_latent_attention_init_impl')
        exec(compile('from __future__ import annotations\n' + ast.unparse(adapter), '<actual MindSpeed MLA initializer>', 'exec'), ns)
        ns['MultiLatentAttention'] = type('MultiLatentAttention', (ns['Attention'],), {'__init__': ns['multi_latent_attention_init_impl']})
        mla_init = self.constructor(nodes[MLA], 'MLASelfAttention')
        mla_cls = ast.ClassDef(name='MLASelfAttention', bases=[ast.Name(id='MultiLatentAttention', ctx=ast.Load())], keywords=[], body=[mla_init], decorator_list=[])
        exec(compile('from __future__ import annotations\n' + ast.unparse(ast.fix_missing_locations(mla_cls)), '<actual Core MLASelfAttention constructor>', 'exec'), ns)
        config = SimpleNamespace(batch_invariant_mode=False, kv_channels=8, num_query_groups=8,
            num_attention_heads=8, softmax_scale=None, recompute_granularity='full', recompute_modules=[],
            fine_grained_activation_offloading=False, offload_modules=[], hidden_size=16,
            output_layer_init_method=None, init_method=None, add_bias_linear=False, fp8=False, fp4=False,
            v_head_dim=4, qk_head_dim=4, qk_pos_emb_head_dim=4, rotary_scaling_factor=1., mscale_all_dim=1.,
            mscale=1., cache_mla_latents=False, rope_type='rope', rotary_percent=1., rotary_base=10000.,
            q_lora_rank=8, kv_lora_rank=8, layernorm_epsilon=1e-6)
        submodules = SimpleNamespace(core_attention=core_attention, linear_proj=object(), linear_q_down_proj=Column,
            linear_q_up_proj=object(), linear_kv_down_proj=Column, linear_kv_up_proj=object(),
            q_layernorm=object(), kv_layernorm=object())
        group = SimpleNamespace(tp=SimpleNamespace(size=lambda: 2), cp=SimpleNamespace(size=lambda: 1))
        instance = ns['MLASelfAttention'](config, submodules, 1, 'causal', pg_collection=group)
        return instance, group, calls

    def test_unpatched_complete_constructor_chain_reproduces_reported_failure(self):
        with self.assertRaisesRegex(TypeError, 'isinstance'):
            self.build_chain(None, patched=False)

    def test_complete_core_mindspeed_mla_constructor_chain_accepts_missing_te_type(self):
        for symbol in [None, lambda: None, object(), 'unavailable']:
            with self.subTest(symbol=type(symbol).__name__):
                instance, group, calls = self.build_chain(symbol)
                self.assertIs(instance.pg_collection, group)
                self.assertTrue(hasattr(instance, 'linear_q_down_proj'))
                self.assertTrue(hasattr(instance, 'linear_kv_up_proj'))
                self.assertTrue(hasattr(instance, 'q_layernorm'))
                self.assertFalse(any(name == 'save' for name, _ in calls))

    def test_both_projection_guards_preserve_complete_backend_behavior(self):
        for file, cls_name in [(ATTENTION, 'Attention'), (MLA, 'MultiLatentAttention')]:
            constructor = self.constructor(self.patched[file], cls_name)
            branch = next(n for n in constructor.body if isinstance(n, ast.If) and 'isinstance(self.linear_proj, TELinear)' in ast.unparse(n.test))
            for symbol in [None, lambda: None, object(), type('RealLinear', (), {})]:
                saved = []
                projection = symbol() if isinstance(symbol, type) else object()
                ns = dict(HAVE_TE=True, TELinear=symbol,
                    self=SimpleNamespace(linear_proj=projection, config=SimpleNamespace(fp8=True, fp8_recipe='current', fp4=False)),
                    is_te_min_version=lambda version: True, set_save_original_input=saved.append)
                exec(compile(ast.Module(body=[branch], type_ignores=[]), '<actual Core projection optimization>', 'exec'), ns)
                self.assertEqual(len(saved), int(isinstance(symbol, type)))

    def test_shared_expert_guard_and_linear_classification_reject_non_types(self):
        relative = 'megatron/core/transformer/moe/shared_experts.py'
        constructor = self.constructor(self.patched[relative], 'SharedExpertMLP')
        condition = next(n for n in ast.walk(constructor) if isinstance(n, ast.If) and ast.unparse(n.test).startswith('HAVE_TE and'))
        with_types = type('RealLinear', (), {})
        for symbol in [None, lambda: None, object(), with_types]:
            ns = dict(HAVE_TE=True, TELinear=symbol, self=SimpleNamespace(linear_fc1=object()), set_save_original_input=lambda value: None)
            exec(compile(ast.Module(body=[condition], type_ignores=[]), '<actual shared expert optional type guard>', 'exec'), ns)
        tree = self.patched['megatron/core/fp8_utils.py']
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'is_column_parallel_linear', 'is_row_parallel_linear'}]
        class Column:
            pass
        class Row:
            pass
        ns = dict(HAVE_TE=True, TEColumnParallelLinear=None, TELayerNormColumnParallelLinear=lambda: None,
                  TERowParallelLinear=object(), ColumnParallelLinear=Column, RowParallelLinear=Row)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<actual Core linear classifiers>', 'exec'), ns)
        self.assertTrue(ns['is_column_parallel_linear'](Column()))
        self.assertTrue(ns['is_row_parallel_linear'](Row()))
        self.assertFalse(ns['is_column_parallel_linear'](object()))


if __name__ == '__main__':
    unittest.main()
