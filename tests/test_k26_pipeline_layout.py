"""Validate tiny-model pipeline layouts before loading NPU frameworks."""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'recipes/kimi-k2.6'
spec = importlib.util.spec_from_file_location('k26_pipeline_layout', RECIPE / 'pipeline_layout.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class PipelineLayoutTest(unittest.TestCase):
    def test_cut_model_uses_even_pipeline_split(self):
        self.assertEqual(layout.resolve_layout(2, 1), (None, None))
        self.assertEqual(layout.resolve_layout(2, 2), (None, None))
        with self.assertRaises(ValueError):
            layout.resolve_layout(2, 4)
        self.assertEqual(layout.resolve_layout(4, 2), (None, None))
        self.assertEqual(layout.resolve_layout(4, 1), (None, None))
        self.assertEqual(layout.resolve_layout(4, 2, '2', '2'), (2, 2))
        with self.assertRaises(ValueError):
            layout.resolve_layout(4, 2, '3', '2')

    def test_full_profile_and_explicit_partial_splits(self):
        self.assertEqual(layout.resolve_layout(61, 10), (3, 2))
        self.assertEqual(layout.resolve_layout(4, 2, '1', None), (1, None))
        self.assertEqual(layout.resolve_layout(4, 2, 'none', 'none'), (None, None))
        for args in [(4, 3), (4, 8), (4, 0), (4, 1, '2', '2'), (4, 2, '0', None)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                layout.resolve_layout(*args)

    def test_old_error_is_reproduced_by_native_core_uneven_validator(self):
        source = ROOT.parent / 'Megatron-LM-aq/megatron/core/transformer/transformer_config.py'
        if not source.is_file():
            self.skipTest('Provide K26 Megatron-LM-aq source')
        cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'TransformerConfig')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
        branch = next(n for n in ast.walk(method) if isinstance(n, ast.If) and
            ast.unparse(n.test) == 'self.num_layers_in_first_pipeline_stage is not None or self.num_layers_in_last_pipeline_stage is not None')
        code = compile(ast.Module(body=branch.body, type_ignores=[]), '<actual Core uneven PP validation>', 'exec')
        config = SimpleNamespace(num_layers=4, pipeline_model_parallel_size=2,
            num_layers_in_first_pipeline_stage=3, num_layers_in_last_pipeline_stage=2,
            virtual_pipeline_model_parallel_size=None)
        with self.assertRaisesRegex(ValueError, '-1 middle layers remaining'):
            exec(code, {'self': config})
        config.num_layers_in_first_pipeline_stage = 2
        exec(code, {'self': config})
        config.num_layers, config.pipeline_model_parallel_size = 61, 10
        config.num_layers_in_first_pipeline_stage, config.num_layers_in_last_pipeline_stage = 3, 2
        exec(code, {'self': config})

    def test_shell_integration_emits_no_uneven_overrides_for_cut_model(self):
        script = (RECIPE / 'kimi_k26_rl.sh').read_text()
        block = script[script.index('# Resolve endpoint layers'):script.index('\nROLLOUT=(')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({'text_config': {'num_hidden_layers': 4}}))
            setup = 'set -euo pipefail\nACTOR=(existing)\n'
            # Pass values as positional shell arguments, not interpolated code.
            setup += 'MODEL_PATH="$1"\nK26_SCRIPT_DIR="$2"\nPYTHON_BIN="$3"\nACTOR_PP=2\n'
            setup += block + '\nprintf "%s\\n" "${ACTOR[@]}"\n'
            result = subprocess.run(['bash', '-c', setup, 'test', str(path), str(RECIPE), sys.executable], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'existing')
            invalid = setup.replace('ACTOR_PP=2\n', 'ACTOR_PP=2\nFIRST_PIPELINE_LAYERS=3\nLAST_PIPELINE_LAYERS=2\n')
            result = subprocess.run(['bash', '-c', invalid, 'test', str(path), str(RECIPE), sys.executable], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid PP layout', result.stderr)

    def test_smoke_actor_topology_is_overridable_and_token_budget_covers_sequence(self):
        script = (RECIPE / 'kimi_k26_rl_8npu_cutlayer.sh').read_text()
        lines = [line for line in script.splitlines() if line.startswith('export ACTOR_') or line.startswith('export LOG_PROB_MAX_TOKEN')]
        harness = 'set -euo pipefail\nMAX_PROMPT_LENGTH=8192\nMAX_RESPONSE_LENGTH=1\n'
        harness += 'ACTOR_TP=2\nACTOR_PP=2\nACTOR_CP=2\nACTOR_EP=4\nACTOR_ETP=1\n'
        harness += '\n'.join(lines) + '\nprintf "%s " "$ACTOR_TP" "$ACTOR_PP" "$ACTOR_CP" "$ACTOR_EP" "$ACTOR_ETP" "$ACTOR_MAX_TOKEN_LEN_PER_GPU" "$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"\n'
        result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '2 2 2 4 1 8193 8193')


if __name__ == '__main__':
    unittest.main()
