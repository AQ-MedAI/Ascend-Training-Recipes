"""Exercise non-PEFT export scans with an unavailable TE operation fuser."""
from __future__ import annotations
import ast
import builtins
import itertools
from collections import defaultdict
import math
from pathlib import Path
import subprocess
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BASE = 'c15303e05b3d48505e4b1b0047faa0c321368856'
PEFT = 'src/megatron/bridge/models/conversion/peft_bridge.py'
LORA = 'src/megatron/bridge/peft/lora_layers.py'


class OptionalPeftScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = ROOT.parent / 'Megatron-Bridge-aq'
        if not checkout.is_dir():
            raise unittest.SkipTest('Provide pinned Megatron-Bridge-aq checkout')
        with tempfile.TemporaryDirectory() as directory:
            for relative in [PEFT, LORA]:
                target = Path(directory) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(subprocess.check_output(['git', '-C', str(checkout), 'show', BASE + ':' + relative], text=True))
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            for name in ['ascend-optional-peft.patch', 'ascend-optional-peft-scan.patch']:
                subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/megatron-bridge' / name)], check=True)
            cls.peft = ast.parse((Path(directory) / PEFT).read_text())
            cls.lora = ast.parse((Path(directory) / LORA).read_text())

    def test_non_peft_adapter_scan_and_task_build_never_import_lora(self):
        cls = next(n for n in self.peft.body if isinstance(n, ast.ClassDef) and n.name == 'MegatronPeftBridge')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                   {'_megatron_global_adapters_info_all_pp_ranks', 'build_adapter_conversion_tasks'}]
        self.assertEqual(len(methods), 2)
        collective_calls = []
        def gather(output, values, group):
            collective_calls.append(values)
            output[0] = values
        group = SimpleNamespace(size=lambda: 1)
        ns = dict(torch=SimpleNamespace(distributed=SimpleNamespace(all_gather_object=gather)),
            parallel_state=SimpleNamespace(get_pipeline_model_parallel_group=lambda: group),
            get_pg_rank=lambda pg: 0, unwrap_model=lambda models: models,
            persistent_buffers=lambda model: [], extract_sort_key=lambda value: value,
            defaultdict=defaultdict, itertools=itertools)
        exec(compile('from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in methods),
                     '<actual adapter scanning methods>', 'exec'), ns)
        scan = ns['_megatron_global_adapters_info_all_pp_ranks']
        model = SimpleNamespace(config=object(), named_parameters=lambda: [('decoder.layer.weight', object())])
        bridge = SimpleNamespace(_unwrap_name=lambda name: name, _is_adapter_param_name=lambda name: '.adapter.' in name,
                                 mapping_registry=lambda: object())
        bridge._megatron_global_adapters_info_all_pp_ranks = MethodType(scan, bridge)
        fake_model_bridge = SimpleNamespace(_megatron_local_name_to_global=lambda models, config, name, stage: name,
                                           WeightConversionTask=object)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith('megatron.bridge.peft.'):
                raise AssertionError('Non-PEFT export imported ' + name)
            if name == 'megatron.bridge.models.conversion.model_bridge':
                return fake_model_bridge
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import):
            result = ns['build_adapter_conversion_tasks'](bridge, model)
            self.assertEqual(dict(result), {})
            self.assertEqual(collective_calls, [[]])
            self.assertEqual(bridge._cached_param_objects_adapter, [])
            self.assertEqual(scan(bridge, [model]), [])
            self.assertEqual(len(collective_calls), 1)

    @unittest.skipIf(torch is None, 'CPU PyTorch is required for class definitions')
    def test_lora_class_definitions_do_not_evaluate_missing_te_ops_annotations(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, to_wrap, adapter):
                super().__init__()
                self.to_wrap = to_wrap
                self.adapter = adapter
        te = SimpleNamespace(Linear=torch.nn.Module)
        nodes = [n for n in self.lora.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        ns = dict(torch=torch, nn=torch.nn, te=te, math=math, AdapterWrapper=Wrapper,
                  safe_import=lambda name: (None, False), apply_random_logits=lambda value: value)
        exec(compile('from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in nodes),
                     '<actual LoRA class definitions>', 'exec'), ns)
        self.assertIn('te.ops.Sequential', ns['TEFusedLoRALinear']._make_fused_branches.__annotations__['return'])
        with self.assertRaisesRegex(RuntimeError, 'requires Transformer Engine ops'):
            ns['TEFusedLoRALinear'](torch.nn.Linear(2, 2), torch.nn.Identity())
        # Complete ops availability permits wrapper construction without executing
        # any fused kernels. A TE implementation is still needed for its forward.
        te.ops = SimpleNamespace(Sequential=type('Sequential', (), {}))
        wrapper = ns['TEFusedLoRALinear'](torch.nn.Linear(2, 2), torch.nn.Identity())
        self.assertIsNone(wrapper._fused_branches)


if __name__ == '__main__':
    unittest.main()
