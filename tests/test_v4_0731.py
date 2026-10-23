"""Official 0731 index/config contracts and backbone-only RL/MOPD compatibility."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / 'third_party/verl'
CROP = ROOT / 'recipes/deepseek-v4/prepare_dsv4_flash_bf16_cut_model.py'
CONTRACT = Path(__file__).parent / 'fixtures/v4_0731_contract.json'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = load(VERL / 'verl/utils/deepseek_v4.py', 'v4_0731_compat')
crop = load(CROP, 'v4_0731_crop')


class V40731Tests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text())
        self.config = copy.deepcopy(self.contract['config'])
        self.disabled = dict(enable=False, enable_train=False, enable_rollout=False)

    def test_official_contract_has_three_prediction_blocks_not_legacy_projection(self):
        self.assertTrue(self.contract['backbone_names_match_original'])
        self.assertEqual(self.contract['prediction_head_indices'], [0, 1, 2])
        keys = self.contract['new_prediction_keys']
        self.assertIn('mtp.0.main_proj.weight', keys)
        self.assertIn('mtp.2.markov_head.markov_w1.weight', keys)
        self.assertIn('mtp.2.confidence_head.proj.weight', keys)
        self.assertEqual(self.config['num_nextn_predict_layers'], 1)
        self.assertEqual(len(self.config['compress_ratios']), 46)

    def test_normalize_preserves_backbone_geometry_and_source(self):
        source = copy.deepcopy(self.config)
        expected = {key: self.config[key] for key in (
            'hidden_size', 'num_attention_heads', 'n_routed_experts', 'hc_mult',
            'num_hidden_layers', 'vocab_size', 'num_hash_layers', 'rope_scaling')}
        self.assertTrue(compat.normalize_v4_dspark_config(self.config, self.disabled))
        self.assertEqual({key: self.config[key] for key in expected}, expected)
        self.assertEqual(self.config['compress_ratios'], source['compress_ratios'][:43])
        self.assertEqual(self.config['num_nextn_predict_layers'], 0)
        self.assertEqual(self.config['mtp_num_layers'], 0)
        self.assertFalse(any(k.startswith('dspark_') for k in self.config))
        self.assertIn('dspark_block_size', source)

    def test_transformers_like_config_and_idempotency(self):
        obj = NS(**self.config)
        compat.normalize_v4_dspark_config(obj, NS(**self.disabled))
        first = copy.deepcopy(vars(obj))
        compat.normalize_v4_dspark_config(obj, NS(**self.disabled))
        self.assertEqual(vars(obj), first)
        self.assertEqual(obj._verl_prediction_head_type, 'dspark')

    def test_each_unsupported_head_switch_fails_before_mutation(self):
        for flag in self.disabled:
            settings = dict(self.disabled, **{flag: True})
            source = copy.deepcopy(self.config)
            with self.assertRaisesRegex(ValueError, 'DSpark prediction heads'):
                compat.normalize_v4_dspark_config(self.config, settings)
            self.assertEqual(self.config, source)

    def test_reenabling_after_normalization_is_rejected(self):
        compat.normalize_v4_dspark_config(self.config, self.disabled)
        with self.assertRaisesRegex(ValueError, 'DSpark'):
            compat.normalize_v4_dspark_config(self.config, {'enable': True})

    def test_legacy_v4_and_other_models_are_unchanged(self):
        legacy = {key: value for key, value in self.config.items() if not key.startswith('dspark_')}
        for config in (legacy, dict(self.config, model_type='deepseek_v3')):
            original = copy.deepcopy(config)
            self.assertFalse(compat.normalize_v4_dspark_config(config, {'enable': True}))
            self.assertEqual(config, original)
            self.assertEqual(compat.v4_backbone_rollout_overrides(config), {})

    def test_rollout_overrides_normalize_disk_config_without_losing_quantization(self):
        compat.normalize_v4_dspark_config(self.config, self.disabled)
        overrides = {'quantization_config': {'quant_method': 'fp8'}}
        overrides.update(compat.v4_backbone_rollout_overrides(self.config))
        self.assertEqual(overrides['num_nextn_predict_layers'], 0)
        self.assertEqual(len(overrides['compress_ratios']), 43)
        self.assertEqual(overrides['quantization_config']['quant_method'], 'fp8')

    def test_export_config_roundtrip_advertises_backbone_only(self):
        compat.normalize_v4_dspark_config(self.config, self.disabled)
        restored = json.loads(json.dumps(self.config))
        self.assertEqual(restored['num_nextn_predict_layers'], 0)
        self.assertFalse(any(k.startswith('dspark_') for k in restored))
        with self.assertRaisesRegex(ValueError, 'DSpark'):
            compat.normalize_v4_dspark_config(restored, {'enable_train': True})

    def test_invalid_ratios_fail_before_changing_model(self):
        self.config['compress_ratios'] = [0]
        original = copy.deepcopy(self.config)
        with self.assertRaisesRegex(ValueError, 'every backbone layer'):
            compat.normalize_v4_dspark_config(self.config, self.disabled)
        self.assertEqual(self.config, original)

    def test_actual_megatron_check_disables_dspark_but_retains_legacy_mtp(self):
        tree = ast.parse((VERL / 'verl/utils/megatron_utils.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_mtp_config')
        namespace = dict(HFModelConfig=object, McoreEngineConfig=object)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'check_mtp_config', 'exec'), namespace)
        with patch.dict(sys.modules, {'verl.utils.deepseek_v4': compat}):
            model = NS(hf_config=NS(**self.config), mtp=NS(**self.disabled))
            namespace['check_mtp_config'](model, NS(override_transformer_config={}))
            self.assertEqual(model.hf_config.num_nextn_predict_layers, 0)
            legacy = NS(model_type='deepseek_v4', num_nextn_predict_layers=1)
            engine = NS(override_transformer_config={})
            model = NS(hf_config=legacy, mtp=NS(enable=True, mtp_loss_scaling_factor=0.17))
            namespace['check_mtp_config'](model, engine)
            self.assertEqual(legacy.num_nextn_predict_layers, 1)
            self.assertEqual(engine.override_transformer_config['mtp_loss_scaling_factor'], 0.17)

    def fixture(self, source):
        self.config['quantization_config'] = {'quant_method': 'fp8'}
        weights = {'embed.weight': torch.ones(8, 4), 'head.weight': torch.ones(8, 4)}
        weights.update({f'layers.{i}.attn.wq_a.weight': torch.full((4, 4), float(i)) for i in range(5)})
        for i in range(3):
            weights[f'mtp.{i}.attn.wq_a.weight'] = torch.full((4, 4), 10. + i)
        weights['mtp.0.main_proj.weight'] = torch.ones(4, 16)
        weights['mtp.0.main_norm.weight'] = torch.ones(16)
        weights['mtp.2.markov_head.markov_w1.weight'] = torch.ones(4, 4)
        weights['mtp.2.confidence_head.proj.weight'] = torch.ones(1, 4)
        crop.dump_json(source / 'config.json', self.config)
        save_file(weights, source / 'weights.safetensors')
        crop.dump_json(source / 'model.safetensors.index.json',
                       {'weight_map': {k: 'weights.safetensors' for k in weights}})
        return weights

    def test_real_safetensors_crop_preserves_backbone_and_omits_all_prediction_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source';source.mkdir()
            output = Path(tmp) / 'crop'
            original = self.fixture(source)
            original_config = (source / 'config.json').read_bytes()
            crop.materialize_bf16_cut(source, output, 4, False, keep_mtp=False)
            tensors = load_file(output / 'weights.safetensors')
            self.assertFalse(any(k.startswith('mtp.') for k in tensors))
            self.assertNotIn('layers.4.attn.wq_a.weight', tensors)
            for key, tensor in tensors.items():
                self.assertTrue(torch.equal(tensor, original[key].bfloat16()))
            config = crop.load_json(output / 'config.json')
            self.assertEqual(config['compress_ratios'], self.config['compress_ratios'][:4])
            self.assertEqual(config['num_nextn_predict_layers'], 0)
            self.assertFalse(any(k.startswith('dspark_') for k in config))
            self.assertEqual((source / 'config.json').read_bytes(), original_config)
            self.assertTrue(crop.is_complete_output(output, source.resolve(), 4))
            # Previously generated 0731 crops retained all 46 compression entries.
            config['compress_ratios'] = self.config['compress_ratios']
            crop.dump_json(output / 'config.json', config)
            self.assertFalse(crop.is_complete_output(output, source.resolve(), 4))

    def test_keep_mtp_rejected_before_replacing_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source';source.mkdir()
            output = Path(tmp) / 'crop';output.mkdir()
            sentinel = output / 'keep.txt';sentinel.write_text('preserve')
            self.fixture(source)
            with self.assertRaisesRegex(ValueError, 'DSpark heads'):
                crop.materialize_bf16_cut(source, output, 4, True, keep_mtp=True)
            self.assertEqual(sentinel.read_text(), 'preserve')

    def test_crop_detects_dspark_tensor_names_without_special_config(self):
        legacy = {key: value for key, value in self.config.items() if not key.startswith('dspark_')}
        with self.assertRaisesRegex(ValueError, 'DSpark heads'):
            crop.validate_mtp_weights(legacy, {'mtp.0.main_proj.weight': 'x.safetensors'})


if __name__ == '__main__':
    unittest.main()
