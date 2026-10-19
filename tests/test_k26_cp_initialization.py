"""Check Core-version argument binding and CP constructor group forwarding."""
from __future__ import annotations
import ast
from copy import deepcopy
from functools import wraps
from inspect import signature
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
UTILS = 'mindspeed/core/context_parallel/model_parallel_utils.py'
DOT = 'mindspeed/core/context_parallel/dot_product_attention.py'
ADAPTOR = 'mindspeed/core/context_parallel/adaptor.py'


class CpInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core = ROOT.parent / 'Megatron-LM-aq/megatron/core/parallel_state.py'
        if not core.is_file():
            raise unittest.SkipTest('Provide K26 Megatron-LM-aq source')
        cls.core = ast.parse(core.read_text())
        cls.original = ast.parse((ROOT / 'third_party/MindSpeed' / UTILS).read_text())
        with tempfile.TemporaryDirectory() as directory:
            for relative in [UTILS, DOT, ADAPTOR]:
                target = Path(directory) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((ROOT / 'third_party/MindSpeed' / relative).read_text())
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/mindspeed/k26/ascend-cp-initialization.patch')], check=True)
            cls.trees = {rel: ast.parse((Path(directory) / rel).read_text()) for rel in [UTILS, DOT, ADAPTOR]}

    def wrappers(self, patched):
        core = deepcopy(next(n for n in self.core.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_model_parallel'))
        # Keep the actual Core signature and its SHARP validation. Do not create
        # process groups: capture the received arguments instead.
        sharp = next(n for n in core.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'use_sharp')
        core.body = [sharp, ast.parse('calls.append(locals().copy())').body[0]]
        core.decorator_list = []
        wrapper_tree = self.trees[UTILS] if patched else self.original
        wrapper = next(n for n in wrapper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_model_parallel_cp_wrapper')
        calls, helpers = [], []
        ns = dict(wraps=wraps, signature=signature, calls=calls)
        for name in ['send_recv_overlap', 'hybrid_cp', 'double_ring']:
            ns['initialize_context_parallel_group_for_' + name] = lambda *args, name=name: helpers.append((name, args))
        exec(compile('from __future__ import annotations\n' + ast.unparse(core) + '\n' + ast.unparse(wrapper), '<actual Core signature and CP wrapper>', 'exec'), ns)
        native = ns['initialize_model_parallel']
        return ns['initialize_model_parallel_cp_wrapper'](native), native, calls, helpers

    def test_old_wrapper_reproduces_sharp_assertion(self):
        fn, native, calls, helpers = self.wrappers(False)
        with self.assertRaisesRegex(AssertionError, 'sharp_enabled_group'):
            fn(tensor_model_parallel_size=2, pipeline_model_parallel_size=2,
               context_parallel_size=2, expert_model_parallel_size=4,
               expert_tensor_parallel_size=1, use_sharp=False)

    def test_requested_layout_and_native_defaults_are_preserved(self):
        fn, native, calls, helpers = self.wrappers(True)
        fn(tensor_model_parallel_size=2, pipeline_model_parallel_size=2,
           context_parallel_size=2, expert_model_parallel_size=4,
           expert_tensor_parallel_size=1, use_sharp=False)
        received = calls[0]
        self.assertFalse(received['use_sharp'])
        self.assertIsNone(received['sharp_enabled_group'])
        self.assertEqual(received['context_parallel_size'], 2)
        self.assertEqual(received['expert_model_parallel_size'], 4)
        self.assertEqual(received['expert_tensor_parallel_size'], 1)
        self.assertTrue(received['create_gloo_process_groups'])
        self.assertIsNone(received['get_embedding_ranks'])
        self.assertEqual(len(helpers), 3)
        self.assertTrue(all(args == (2, 2, 2, {}) for _, args in helpers))
        self.assertEqual(signature(fn), signature(native))

    def test_new_options_positional_calls_and_unknown_options(self):
        fn, native, calls, helpers = self.wrappers(True)
        fn(2, 2, None, use_sharp=False, context_parallel_size=2,
           hybrid_context_parallel=True, high_priority_stream_groups=['tp'], create_all_gather_group=True,
           create_gloo_process_groups=False)
        self.assertTrue(calls[0]['hybrid_context_parallel'])
        self.assertEqual(calls[0]['high_priority_stream_groups'], ['tp'])
        self.assertTrue(calls[0]['create_all_gather_group'])
        self.assertFalse(calls[0]['create_gloo_process_groups'])
        with self.assertRaises(TypeError):
            fn(removed_or_unknown_option=True)
        with self.assertRaisesRegex(AssertionError, 'sharp_enabled_group'):
            fn(use_sharp=False, sharp_enabled_group='dp')

    def test_eight_rank_attention_and_expert_pipeline_groups_match(self):
        nodes = [n for n in self.core.body if (isinstance(n, ast.FunctionDef) and n.name == 'generate_masked_orthogonal_rank_groups')
                 or (isinstance(n, ast.ClassDef) and n.name == 'RankGenerator')]
        ns = {}
        exec(compile('from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in nodes), '<actual Core rank generator>', 'exec'), ns)
        RankGenerator = ns['RankGenerator']
        dense = RankGenerator(tp=2, pp=2, cp=2, ep=1, dp=1, order='tp-cp-ep-dp-pp')
        expert = RankGenerator(tp=1, pp=2, cp=1, ep=4, dp=1, order='tp-cp-ep-dp-pp')
        self.assertEqual(dense.world_size, 8)
        self.assertEqual(expert.world_size, 8)
        self.assertEqual(dense.get_ranks('pp'), expert.get_ranks('pp'))
        self.assertEqual(dense.get_ranks('pp'), [[0, 4], [1, 5], [2, 6], [3, 7]])

    def test_attention_wrappers_keep_explicit_cp_group(self):
        for name in ['attention_init_wrapper', 'multi_latent_attention_init_wrapper']:
            function = next(n for n in self.trees[ADAPTOR].body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = []
            def native(instance, *args, **kwargs):
                calls.append(kwargs)
                instance.core_attention = object()
            ns = dict(wraps=wraps, UlyssesContextAttention=lambda attention, group: (attention, group))
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual CP attention wrappers>', 'exec'), ns)
            config = SimpleNamespace(context_parallel_size=2, tp_2d=False, context_parallel_algo='ulysses_cp_algo')
            groups = SimpleNamespace(cp=object())
            instance = SimpleNamespace()
            ns[name](native)(instance, config, object(), 1, 'causal', 'self', pg_collection=groups)
            self.assertIs(calls[0]['pg_collection'], groups)
            self.assertIs(instance.core_attention[1], groups.cp)

    def test_cp_attention_constructor_forwards_explicit_tp_group(self):
        cls = next(n for n in self.trees[DOT].body if isinstance(n, ast.ClassDef) and n.name == 'CPDotProductAttentionImpl')
        constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        calls = []
        class Base:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.config = kwargs['config']
        fixture = ast.ClassDef(name='CPDotProductAttentionImpl', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[constructor], decorator_list=[])
        ns = dict(Base=Base, math=math, divide=lambda x, y: x // y,
            FusedScaleMaskSoftmax=lambda **kwargs: object(), attention_mask_func=object(),
            torch=SimpleNamespace(nn=SimpleNamespace(Dropout=lambda value: object())))
        exec(compile('from __future__ import annotations\n' + ast.unparse(ast.fix_missing_locations(fixture)), '<actual CP attention constructor>', 'exec'), ns)
        config = SimpleNamespace(context_parallel_size=2, window_size=None, kv_channels=128,
            num_attention_heads=8, num_query_groups=8, apply_query_key_layer_scaling=False,
            fp16=False, bf16=True, masked_softmax_fusion=False, attention_softmax_in_fp32=False,
            attention_dropout=0., alibi_fusion_attn_type=None)
        group = SimpleNamespace(tp=SimpleNamespace(size=lambda: 2))
        result = ns['CPDotProductAttentionImpl'](config, 1, 'causal', 'self', pg_collection=group)
        self.assertIs(calls[0]['pg_collection'], group)
        self.assertEqual(result.num_attention_heads_per_partition, 4)
        self.assertEqual(config.context_parallel_size, 2)


if __name__ == '__main__':
    unittest.main()
