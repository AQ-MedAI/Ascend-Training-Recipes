"""Cross-source checks for K26 Core callers and MindSpeed BF16 adapters."""
from __future__ import annotations
import ast
import dataclasses
import inspect
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = 'mindspeed/te/pytorch/attention/dot_product_attention/dot_product_attention.py'
NORM = 'mindspeed/te/pytorch/module/layernorm_column_parallel_linear.py'
GROUPS = ['mindspeed/te/pytorch/module/grouped_linear.py', 'mindspeed/te/pytorch/module/performance_grouped_linear.py']
FEATURE = 'mindspeed/features_manager/megatron_basic/transformer_engine_basic.py'


class TeContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = ROOT.parent / 'Megatron-LM-aq'
        if not cls.core.is_dir():
            raise unittest.SkipTest('Provide K26 Megatron-LM-aq source')
        cls.trees = {}
        with tempfile.TemporaryDirectory() as directory:
            for relative in [ATTENTION, NORM, *GROUPS, FEATURE]:
                source = ROOT / 'third_party/MindSpeed' / relative
                if not source.is_file():
                    raise unittest.SkipTest('Prepare MindSpeed sources first')
                target = Path(directory) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text())
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/mindspeed/k26/ascend-te-model-contracts.patch')], check=True)
            cls.trees = {relative: ast.parse((Path(directory) / relative).read_text())
                         for relative in [ATTENTION, NORM, *GROUPS, FEATURE]}

    def method(self, relative, cls_name, name='__init__'):
        cls = next(n for n in self.trees[relative].body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def assert_keywords(self, function, keys):
        names = {a.arg for a in function.args.args + function.args.kwonlyargs}
        # Explicit group arguments must not be swallowed by **kwargs.
        for key in {'tp_group', 'pg_collection', 'stride'} & keys:
            self.assertIn(key, names, (function.name, key))
        if function.args.kwarg is None:
            self.assertFalse(keys - names, (function.name, keys - names))

    def test_core_attention_and_mlp_callers_match_adapter_signatures(self):
        cases = 0
        for file in ['transformer/attention.py', 'transformer/multi_latent_attention.py', 'transformer/mlp.py', 'transformer/moe/experts.py']:
            tree = ast.parse((self.core / 'megatron/core' / file).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if isinstance(target, ast.Name) and target.id == 'build_module' and node.args:
                    target = node.args[0]
                if not isinstance(target, ast.Attribute):
                    continue
                keys = {k.arg for k in node.keywords if k.arg is not None}
                candidates = []
                if target.attr == 'core_attention':
                    candidates = [self.method(ATTENTION, 'MindSpeedTEDotProductAttention', '__init__' if 'config' in keys else 'forward')]
                elif target.attr == 'linear_fc1' and file == 'transformer/mlp.py':
                    candidates = [self.method(NORM, 'MindSpeedTELayerNormColumnParallelLinear')]
                elif target.attr in {'linear_fc1', 'linear_fc2'} and file == 'transformer/moe/experts.py' and 'pg_collection' in keys:
                    for relative, stem in zip(GROUPS, ['MindSpeedTE', 'MindSpeedTEPerformance']):
                        mode = 'Column' if target.attr == 'linear_fc1' else 'Row'
                        candidates.append(self.method(relative, stem + mode + 'ParallelGroupedLinear'))
                for candidate in candidates:
                    self.assert_keywords(candidate, keys)
                    cases += 1
        self.assertGreaterEqual(cases, 7)

    def attention_fixture(self, cp_size=1):
        calls = []
        class Backend:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.cp_group = kwargs.get('cp_group')
            def forward(self, *args, **kwargs):
                calls.append(kwargs)
                return 'attention result'
        group = SimpleNamespace(tp=SimpleNamespace(size=lambda: 8), cp=SimpleNamespace(size=lambda: cp_size))
        packed_cls = dataclasses.make_dataclass('Packed', [(name, object, dataclasses.field(default=None)) for name in
            ['qkv_format', 'cu_seqlens_q', 'cu_seqlens_kv', 'cu_seqlens_q_padded', 'cu_seqlens_kv_padded',
             'max_seqlen_q', 'max_seqlen_kv', 'local_cp_size', 'cp_group']])
        def forbidden(*args, **kwargs):
            raise AssertionError('Explicit groups must not use global parallel state')
        torch = SimpleNamespace(distributed=SimpleNamespace(get_process_group_ranks=lambda pg: [2, 4]),
            npu=SimpleNamespace(Stream=lambda **kwargs: object(), current_device=lambda: 0))
        ns = dict(DotProductAttention=Backend, dataclasses=dataclasses, PackedSeqParams=packed_cls, torch=torch,
            get_tensor_model_parallel_group=forbidden, get_context_parallel_group=forbidden,
            get_context_parallel_global_ranks=forbidden, get_full_args=lambda: SimpleNamespace(context_parallel_algo='megatron_cp_algo'),
            AttnMaskType=SimpleNamespace(causal='causal'), get_attention_mask=lambda config: None)
        cls = next(n for n in self.trees[ATTENTION].body if isinstance(n, ast.ClassDef) and n.name == 'MindSpeedTEDotProductAttention')
        exec(compile('from __future__ import annotations\n' + ast.unparse(cls), '<actual TE attention adapter>', 'exec'), ns)
        config = SimpleNamespace(context_parallel_size=cp_size, num_query_groups=64, multi_latent_attention=True,
            qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128, kv_channels=128,
            num_attention_heads=64, attention_dropout=0., sequence_parallel=True, tensor_model_parallel_size=99,
            attention_mask_type='causal')
        instance = ns['MindSpeedTEDotProductAttention'](config, 1, 'causal', 'self', pg_collection=group)
        return instance, group, calls, packed_cls

    def test_real_te_attention_constructor_uses_supplied_tp_and_cp(self):
        for cp_size in (1, 2):
            instance, group, calls, packed = self.attention_fixture(cp_size)
            self.assertIs(calls[0]['tp_group'], group.tp)
            self.assertEqual(calls[0]['tp_size'], 8)
            self.assertEqual(calls[0]['kv_channels'], (192, 128))
            if cp_size > 1:
                self.assertIs(calls[0]['cp_group'], group.cp)
                self.assertEqual(calls[0]['cp_global_ranks'], [2, 4])

    def test_packed_cp_metadata_is_validated_and_not_forwarded_as_kernel_keywords(self):
        instance, group, calls, Packed = self.attention_fixture()
        result = instance.forward(None, None, None, 'mask', 'causal',
            packed_seq_params=Packed(qkv_format='thd', local_cp_size=1, cp_group=group.cp))
        self.assertEqual(result, 'attention result')
        self.assertNotIn('local_cp_size', calls[-1])
        self.assertNotIn('cp_group', calls[-1])
        self.assertEqual(calls[-1]['qkv_format'], 'thd')
        with self.assertRaises(NotImplementedError):
            instance.forward(None, None, None, 'mask', 'causal', packed_seq_params=Packed(local_cp_size=2))
        with self.assertRaises(NotImplementedError):
            instance.forward(None, None, None, 'mask', 'causal', packed_seq_params=Packed(cp_group=object()))
        backend = next(n for n in self.trees[ATTENTION].body if isinstance(n, ast.ClassDef) and n.name == 'DotProductAttention')
        forward = next(n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        self.assert_keywords(forward, instance.kept_packed_seq_params)

    def test_bf16_keeps_native_block_forward_and_recompute(self):
        fn = self.method(FEATURE, 'TransformerEngineBasicFeature', 'register_patches')
        selection = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "getattr(args, 'fp8', None)")
        for enabled in (False, True):
            calls = []
            ns = dict(args=SimpleNamespace(fp8=enabled, swap_attention=False, recompute_method='uniform'),
                      patch_manager=SimpleNamespace(register_patch=lambda *args: calls.append(args)),
                      transformer_block_forward=object(), transformer_block_checkpointed_forward=object())
            exec(compile(ast.Module(body=[selection], type_ignores=[]), '<actual block patch selection>', 'exec'), ns)
            self.assertEqual(len(calls), 2 if enabled else 0)

    def test_grouped_experts_partition_using_expert_tp_not_attention_tp(self):
        for relative, stem in zip(GROUPS, ['MindSpeedTE', 'MindSpeedTEPerformance']):
            original = self.method(relative, stem + 'GroupedLinear')
            # Execute actual initialization through group selection and shape
            # partitioning, stopping before the first NPU tensor allocation.
            prefix = []
            for node in original.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr in {'total_weight', 'weight'} for t in node.targets):
                    break
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == '__init__':
                    continue
                prefix.append(node)
            fn = ast.FunctionDef(name='initialize', args=original.args, body=prefix, decorator_list=[])
            ns = dict(_get_partition_dim=lambda mode: 0 if mode == 'column' else 1)
            exec(compile('from __future__ import annotations\n' + ast.unparse(ast.fix_missing_locations(fn)), '<actual grouped partition setup>', 'exec'), ns)
            groups = SimpleNamespace(tp=SimpleNamespace(size=lambda: 8), expt_tp=SimpleNamespace(size=lambda: 2))
            for mode in ['column', 'row']:
                instance = SimpleNamespace()
                ns['initialize'](instance, 4, 16, 32, parallel_mode=mode,
                    config=SimpleNamespace(expert_model_parallel_size=8), init_method=None,
                    bias=False, skip_bias_add=False, is_expert=True, pg_collection=groups)
                self.assertEqual((instance.input_size, instance.output_size), (16, 16) if mode == 'column' else (8, 32))
                with self.assertRaisesRegex(ValueError, 'Conflicting'):
                    ns['initialize'](SimpleNamespace(), 4, 16, 32, parallel_mode=mode,
                        config=SimpleNamespace(expert_model_parallel_size=8), init_method=None,
                        bias=False, skip_bias_add=False, is_expert=True, pg_collection=groups, tp_group=object())

    def test_fused_norm_stride_reaches_both_weight_initializers(self):
        constructor = self.method(NORM, 'MindSpeedTELayerNormColumnParallelLinear')
        calls = [node for node in ast.walk(constructor) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id in {'_initialize_affine_weight_cpu', '_initialize_affine_weight_gpu'}]
        self.assertEqual(len(calls), 2)
        for call in calls:
            stride = next(k.value for k in call.keywords if k.arg == 'stride')
            self.assertIsInstance(stride, ast.Name)
            self.assertEqual(stride.id, 'stride')

    def test_bf16_mla_low_rank_projection_uses_supported_native_linear(self):
        function = next(n for n in self.trees[FEATURE].body if isinstance(n, ast.FunctionDef) and n.name == 'k26_bf16_mla_linear')
        function.body = [n for n in function.body if not isinstance(n, ast.ImportFrom)]
        column = object()
        ns = dict(ColumnParallelLinear=column)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual BF16 spec selection>', 'exec'), ns)
        self.assertIs(ns['k26_bf16_mla_linear'](None), column)
        # Verify the actual pinned Core constructor, including explicit TP.
        layers = ast.parse((self.core / 'megatron/core/tensor_parallel/layers.py').read_text())
        cls = next(n for n in layers.body if isinstance(n, ast.ClassDef) and n.name == 'ColumnParallelLinear')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        mla = ast.parse((self.core / 'megatron/core/transformer/multi_latent_attention.py').read_text())
        for call in ast.walk(mla):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'build_module' and call.args:
                target = call.args[0]
                if isinstance(target, ast.Attribute) and target.attr in {'linear_q_down_proj', 'linear_kv_down_proj'}:
                    self.assert_keywords(init, {k.arg for k in call.keywords if k.arg is not None} | {'gather_output'})

    def test_startup_checker_rejects_old_adapters_before_ray(self):
        source = ROOT / 'tools/check_k26_runtime_contracts.py'
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef))
        fn.body = [node for node in fn.body if not isinstance(node, ast.ImportFrom)]
        def attention(self, pg_collection=None):
            pass
        def norm(self, tp_group=None, stride=1):
            pass
        def forward(self, rotary_pos_cos_sin=None, padding_mask=None):
            pass
        def checkpoint(self, use_inner_quantization_context=False, padding_mask=None):
            pass
        te = SimpleNamespace(TEDotProductAttention=SimpleNamespace(__init__=attention),
            TELayerNormColumnParallelLinear=SimpleNamespace(__init__=norm),
            TEColumnParallelGroupedLinear=SimpleNamespace(__init__=attention),
            TERowParallelGroupedLinear=SimpleNamespace(__init__=attention))
        block = SimpleNamespace(forward=forward, _checkpointed_forward=checkpoint)
        column = object()
        def initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, expert_model_parallel_size=1, expert_tensor_parallel_size=None,
            use_sharp=False, sharp_enabled_group=None, create_gloo_process_groups=True):
            pass
        def adam(params, adam_w_mode=True):
            pass
        ns = dict(inspect=inspect, te=te, TransformerBlock=block, ColumnParallelLinear=column,
                  optimizer_module=SimpleNamespace(USING_PYTORCH_OPTIMIZER=False, Adam=adam),
                  mpu=SimpleNamespace(initialize_model_parallel=initialize_model_parallel),
                  initialize_model_parallel_cp_wrapper=lambda fn: fn,
                  TESpecProvider=lambda: SimpleNamespace(linear=lambda: column))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual startup checker>', 'exec'), ns)
        self.assertEqual(ns['check_runtime_contracts'](), 9)
        def old(self, **kwargs):
            pass
        def old_adam(params):
            pass
        ns['optimizer_module'].Adam = old_adam
        with self.assertRaisesRegex(RuntimeError, 'Adam optimizer: missing adam_w_mode'):
            ns['check_runtime_contracts']()
        ns['optimizer_module'].Adam = adam
        te.TEDotProductAttention.__init__ = old
        with self.assertRaisesRegex(RuntimeError, 'TEDotProductAttention.*pg_collection'):
            ns['check_runtime_contracts']()

    def test_grouped_wrappers_forward_both_group_interfaces(self):
        for relative, stem in zip(GROUPS, ['MindSpeedTE', 'MindSpeedTEPerformance']):
            base_name = stem + 'GroupedLinear'
            for mode in ['Column', 'Row']:
                cls_name = stem + mode + 'ParallelGroupedLinear'
                cls = next(n for n in self.trees[relative].body if isinstance(n, ast.ClassDef) and n.name == cls_name)
                constructor = self.method(relative, cls_name)
                calls = []
                class Base:
                    def __init__(self, **kwargs):
                        calls.append(kwargs)
                cls = ast.ClassDef(name=cls_name, bases=[ast.Name(id=base_name, ctx=ast.Load())], keywords=[],
                                   body=[constructor], decorator_list=[])
                ns = {base_name: Base, 'condition_init_method': lambda config, method: method}
                exec(compile('from __future__ import annotations\n' + ast.unparse(ast.fix_missing_locations(cls)), '<actual grouped wrapper>', 'exec'), ns)
                group, tp = object(), object()
                ns[cls_name](2, 8, 16, config=object(), init_method=None, bias=False, skip_bias_add=True,
                             is_expert=True, pg_collection=group, tp_group=tp)
                self.assertIs(calls[-1]['pg_collection'], group)
                self.assertIs(calls[-1]['tp_group'], tp)


if __name__ == '__main__':
    unittest.main()
