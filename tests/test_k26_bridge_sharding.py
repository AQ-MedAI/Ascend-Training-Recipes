"""Check MindSpeed layer detection and actual Bridge TP splitting on CPU."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BASE = 'c15303e05b3d48505e4b1b0047faa0c321368856'


class BridgeShardingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = ROOT.parent / 'Megatron-Bridge-aq'
        if not checkout.is_dir():
            raise unittest.SkipTest('Provide pinned Megatron-Bridge-aq checkout')
        relative = 'src/megatron/bridge/models/conversion/param_mapping.py'
        original = subprocess.check_output(['git', '-C', str(checkout), 'show', BASE + ':' + relative], text=True)
        cls.original = ast.parse(original)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / relative
            target.parent.mkdir(parents=True)
            target.write_text(original)
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/megatron-bridge/ascend-automapping-sharding.patch')], check=True)
            cls.tree = ast.parse(target.read_text())

    def mapping_classes(self, patched=True):
        # Execute actual AutoMapping and ColumnParallelMapping methods; simulate
        # only distributed scatter, so CPU tests need no initialized NPU ranks.
        state = SimpleNamespace(rank=0, size=8, splits=None)
        class BaseMapping:
            def __class_getitem__(cls, item):
                return cls
            def __init__(self, megatron_param, hf_param):
                self.megatron_param = megatron_param
                self.hf_param = hf_param
                self.tp_rank = state.rank
                self.tp_size = state.size
                self.is_expert = False
                self.is_adapter = False
            def broadcast_from_pp_rank(self, value, cache_key=None):
                return value
            def broadcast_obj_from_pp_rank(self, value, cache_key=None):
                return value
            def maybe_dequantize(self, value):
                return value
            def gather_from_tp_ranks(self, value):
                return state.splits
            def _normalize_expert_param_name(self, name):
                return name
            def scatter_to_tp_ranks(self, splits, shape, dtype, device):
                if splits is not None:
                    state.splits = splits
                result = state.splits[self.tp_rank]
                if tuple(result.shape) != tuple(shape):
                    raise ValueError(f'TP shard shape {result.shape} differs from target {shape}')
                return result
        classes = [n for n in (self.tree if patched else self.original).body if isinstance(n, ast.ClassDef)
                   and n.name in {'AutoMapping', 'ColumnParallelMapping', 'RowParallelMapping', 'ReplicatedMapping'}]
        # Postpone annotations: the real class bodies can be loaded without GPU imports.
        torch_namespace = torch if torch is not None else SimpleNamespace(Tensor=object)
        namespace = dict(MegatronParamMapping=BaseMapping, torch=torch_namespace, json=json,
            get_module_and_param_from_name=lambda module, name: (module, getattr(module, name.split('.')[-1])),
            logger=SimpleNamespace(warning=lambda *args: None))
        code = 'from __future__ import annotations\n' + '\n\n'.join(ast.unparse(n) for n in classes)
        exec(compile(code, '<actual Bridge mapping classes>', 'exec'), namespace)
        return namespace, state

    def test_original_fused_projection_is_misclassified_as_replicated(self):
        ns, state = self.mapping_classes(patched=False)
        module = type('MindSpeedTELayerNormColumnParallelLinear', (), {})()
        mapping = ns['AutoMapping']('decoder.layers.0.self_attention.linear_kv_up_proj.weight', 'hf.weight')
        self.assertEqual(mapping._detect_parallelism_type(module), 'replicated')

    def test_fused_projection_is_sharded_but_norm_parameters_are_replicated(self):
        ns, state = self.mapping_classes()
        for class_name in ['TELayerNormColumnParallelLinear', 'MindSpeedTELayerNormColumnParallelLinear']:
            module = type(class_name, (), {})()
            for parameter, expected in [('weight', 'column'), ('bias', 'column'),
                                        ('layer_norm_weight', 'replicated'), ('layer_norm_bias', 'replicated')]:
                mapping = ns['AutoMapping']('decoder.layer.linear.' + parameter, 'hf.weight')
                self.assertEqual(mapping._detect_parallelism_type(module), expected)

    def test_mindspeed_grouped_parallel_layer_names(self):
        ns, state = self.mapping_classes()
        for prefix in ['MindSpeedTE', 'MindSpeedTEPerformance']:
            for direction, expected in [('Column', 'column'), ('Row', 'row')]:
                module = type(prefix + direction + 'ParallelGroupedLinear', (), {})()
                mapping = ns['AutoMapping']('decoder.experts.weight0', 'hf.weight')
                self.assertEqual(mapping._detect_parallelism_type(module), expected)

    @unittest.skipIf(torch is None, 'CPU PyTorch is needed for TP tensor splitting')
    def test_reported_kv_projection_shape_splits_into_eight_correct_shards(self):
        ns, state = self.mapping_classes()
        full = torch.arange(16384 * 512, dtype=torch.float32).reshape(16384, 512)
        module = type('MindSpeedTELayerNormColumnParallelLinear', (), {})()
        module.weight = torch.empty(2048, 512)
        shards = []
        for rank in range(8):
            state.rank = rank
            mapping = ns['AutoMapping']('decoder.layers.0.self_attention.linear_kv_up_proj.weight',
                'language_model.model.layers.0.self_attn.kv_b_proj.weight')
            shard = mapping.hf_to_megatron(full if rank == 0 else None, module)
            self.assertEqual(shard.shape, module.weight.shape)
            torch.testing.assert_close(shard, full[rank * 2048:(rank + 1) * 2048])
            shards.append(shard)
        torch.testing.assert_close(torch.cat(shards, dim=0), full)
        state.rank = 0
        exported = ns['AutoMapping']('decoder.layers.0.self_attention.linear_kv_up_proj.weight',
            'language_model.model.layers.0.self_attn.kv_b_proj.weight').megatron_to_hf(shards[0], module)
        torch.testing.assert_close(exported['language_model.model.layers.0.self_attn.kv_b_proj.weight'], full)


if __name__ == '__main__':
    unittest.main()
