"""Exercise both HF bridge-save branches with swap optimizer disabled."""
import __future__
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from _v4_opd_test_sources import source_root

ROOT = source_root()
SOURCE = ROOT / 'verl/utils/checkpoint/megatron_checkpoint_manager.py'


class CheckpointInspectScopeTest(unittest.TestCase):
    def test_hf_bridge_save_without_swap_optimizer(self):
        tree = ast.parse(SOURCE.read_text())
        manager = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and n.name == 'MegatronCheckpointManager')
        method = next(n for n in manager.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'save_checkpoint')
        namespace = dict(
            inspect=inspect, logger=None, log_with_rank=Mock(),
            local_mkdir_safe=lambda path: path,
            get_dist_checkpoint_path=lambda path: path + '/dist',
            get_hf_model_checkpoint_path=lambda path: path + '/huggingface',
            save_dist_checkpointing=Mock(return_value=None),
            torch=SimpleNamespace(distributed=SimpleNamespace(barrier=Mock())),
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        for distributed in [False, True]:
            with self.subTest(distributed=distributed):
                calls = []
                def save_weights(models, weights_path, memory_efficient=True):
                    calls.append((models, weights_path, memory_efficient))

                obj = SimpleNamespace(
                    _uses_swap_optimizer=lambda: False,
                    checkpoint_config=SimpleNamespace(async_save=False,
                        mbridge_config={'memory_efficient': False, 'unsupported': 'ignored'}),
                    ensure_checkpoint_capacity=Mock(),
                    _build_sharded_state_dict_metadata=lambda: {},
                    generate_state_dict=lambda *a, **kw: {'model': {}},
                    _maybe_filter_peft_state_dict=lambda sd: sd,
                    use_dist_checkpointing=distributed,
                    use_hf_checkpoint=not distributed,
                    should_save_model=not distributed,
                    should_save_hf_model=distributed,
                    should_save_optimizer=False, should_save_extra=False,
                    peft_cls=None, vanilla_bridge=True,
                    bridge=SimpleNamespace(save_weights=save_weights),
                    model=['model'], rank=1, register_checkpoint=Mock(),
                )
                namespace['save_checkpoint'](obj, '/checkpoint/step_1', global_step=1)
                self.assertEqual(calls, [(['model'], '/checkpoint/step_1/huggingface', False)])
                self.assertEqual(obj.previous_global_step, 1)
                obj.register_checkpoint.assert_called_once_with('/checkpoint/step_1', None)


if __name__ == '__main__':
    unittest.main()
