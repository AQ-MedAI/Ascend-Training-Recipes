"""Verify optional dist-train implementations stay outside config-only imports."""
import ast
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = 'mindspeed/core/multi_modal/dist_train/__init__.py'


class DistTrainImportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        source = ROOT / 'third_party/MindSpeed' / RELATIVE
        if not source.is_file():
            self.skipTest('Prepare MindSpeed sources first')
        target = root / RELATIVE
        target.parent.mkdir(parents=True)
        target.write_text(source.read_text())
        subprocess.run(['git', '-C', str(root), 'init', '-q'], check=True)
        subprocess.run(['git', '-C', str(root), 'apply', str(ROOT / 'patches/mindspeed/k26/ascend-lazy-dist-train.patch')], check=True)
        self.package = root / 'dist_fixture'
        target.parent.rename(self.package)
        self.package.joinpath('dist_train_config.py').write_text(source.with_name('dist_train_config.py').read_text())
        for name in ['inner_data_parallel', 'dist_parallel_state', 'dist_schedules', 'dist_communication']:
            self.package.joinpath(name + '.py').write_text("raise ImportError('legacy implementation requested')\n")
        self.path_patch = patch.object(sys, 'path', [str(root)] + sys.path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.remove_modules)

    def remove_modules(self):
        for name in list(sys.modules):
            if name == 'dist_fixture' or name.startswith('dist_fixture.'):
                del sys.modules[name]

    def test_real_config_import_does_not_load_implementations(self):
        # The configuration file only needs torch.distributed at import time.
        fake_torch = types.ModuleType('torch')
        fake_torch.__path__ = []
        fake_distributed = types.ModuleType('torch.distributed')
        fake_torch.distributed = fake_distributed
        with patch.dict(sys.modules, {'torch': fake_torch, 'torch.distributed': fake_distributed}):
            config = importlib.import_module('dist_fixture.dist_train_config')
            self.assertEqual(config.get_all_config(), {})
        self.assertFalse(any('dist_fixture.' + name in sys.modules for name in
                             ['inner_data_parallel', 'dist_parallel_state', 'dist_schedules', 'dist_communication']))

    def test_text_argument_validation_does_not_load_multimodal_implementations(self):
        source = ROOT / 'third_party/MindSpeed/mindspeed/features_manager/dist_train/dist_train_feature.py'
        tree = ast.parse(source.read_text().replace('mindspeed.core.multi_modal.dist_train', 'dist_fixture'))
        feature = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        validate = next(node for node in feature.body if isinstance(node, ast.FunctionDef) and node.name == 'validate_args')
        namespace = {}
        exec(compile(ast.Module(body=[validate], type_ignores=[]), '<actual DistTrainFeature validator>', 'exec'), namespace)
        fake_torch = types.ModuleType('torch')
        fake_torch.__path__ = []
        fake_torch.distributed = types.ModuleType('torch.distributed')
        args = types.SimpleNamespace(dist_train=False, sequence_parallel=True, tensor_model_parallel_size=8)
        with patch.dict(sys.modules, {'torch': fake_torch, 'torch.distributed': fake_torch.distributed}):
            namespace['validate_args'](None, args)
            self.assertTrue(args.sequence_parallel)
            self.assertTrue('dist_fixture.dist_train_config' in sys.modules)
            self.assertNotIn('dist_fixture.dist_parallel_state', sys.modules)

    def test_requested_missing_implementation_keeps_original_error(self):
        package = importlib.import_module('dist_fixture')
        with self.assertRaisesRegex(ImportError, 'legacy implementation requested'):
            _ = package.dist_parallel_state
        self.assertNotIn('dist_parallel_state', vars(package))
        with self.assertRaises(AttributeError):
            _ = package.unknown_export

    def test_successful_module_export_is_cached_and_discoverable(self):
        self.package.joinpath('dist_parallel_state.py').write_text('marker = object()\n')
        package = importlib.import_module('dist_fixture')
        from dist_fixture import dist_parallel_state
        self.assertIs(package.dist_parallel_state, dist_parallel_state)
        self.assertIs(vars(package)['dist_parallel_state'], dist_parallel_state)
        self.assertIn('dist_parallel_state', dir(package))
        self.assertEqual(set(package.__all__), {'inner_data_parallel', 'dist_parallel_state', 'dist_schedules', 'dist_communication'})


if __name__ == '__main__':
    unittest.main()
