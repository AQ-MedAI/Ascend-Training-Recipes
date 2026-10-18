"""CPU checks of Kimi text-only provider inputs and checkpoint mappings."""
from __future__ import annotations
import ast
from copy import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = 'c15303e05b3d48505e4b1b0047faa0c321368856'


class KimiTextBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = ROOT.parent / 'Megatron-Bridge-aq'
        if not checkout.is_dir():
            raise unittest.SkipTest('Provide the pinned Megatron-Bridge-aq checkout')
        cls.registrations = []
        class Register:
            @staticmethod
            def register_bridge(**kwargs):
                cls.registrations.append(kwargs)
                return lambda value: value
        class Mapping:
            def __init__(self, megatron_param, hf_param=None, gate=None, up=None):
                self.megatron_param = megatron_param
                self.hf_param = hf_param if hf_param is not None else dict(gate=gate, up=up)
        class Registry:
            def __init__(self, *mappings):
                self.mappings = mappings
            def megatron_to_hf_lookup(self, name):
                return next((m for m in self.mappings if m.megatron_param == name), None)
        class Parent:
            def provider_bridge(self, hf):
                return SimpleNamespace(received=hf.config, seq_length=4096)
        relative = 'src/megatron/bridge/models/kimi/kimi_k25_bridge.py'
        initializer = 'src/megatron/bridge/models/__init__.py'
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / initializer
            target.parent.mkdir(parents=True)
            target.write_text(subprocess.check_output(['git', '-C', str(checkout), 'show', BASE + ':' + initializer], text=True))
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/megatron-bridge/ascend-kimi-text-bridge.patch')], check=True)
            text = (Path(directory) / relative).read_text()
            cls.assert_registration_import = 'from megatron.bridge.models.kimi.kimi_k25_bridge import KimiK25TextBridge' in target.read_text()
        # Exercise the actual baseline mapping declarations and patched bridge
        # with lightweight collaborators; no simulated NPU computation.
        common = subprocess.check_output(['git', '-C', str(checkout), 'show', BASE + ':src/megatron/bridge/models/deepseek/common.py'], text=True)
        common_fn = next(n for n in ast.parse(common).body if isinstance(n, ast.FunctionDef))
        ns = dict(copy=copy, json=json, Path=Path, SimpleNamespace=SimpleNamespace,
                  MegatronModelBridge=Register, DeepSeekV3Bridge=Parent,
                  GPTModel=object, MLAModelProvider=object, AutoMapping=Mapping,
                  GatedMLPMapping=Mapping, MegatronMappingRegistry=Registry)
        nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=[common_fn] + nodes, type_ignores=[]), '<patched Kimi bridge>', 'exec'), ns)
        cls.ns = ns

    def config(self, directory):
        text = SimpleNamespace(model_type='deepseek_v3', num_hidden_layers=4, hidden_size=7168,
            num_attention_heads=64, n_routed_experts=384, moe_intermediate_size=2048,
            first_k_dense_replace=1, n_shared_experts=1, kv_lora_rank=512,
            qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
            max_position_embeddings=32768)
        return SimpleNamespace(text_config=text, _name_or_path=directory, dtype='bfloat16')

    def write_index(self, directory, root='language_model.model.', head='language_model.lm_head.weight'):
        names = [root + tail for tail in ['embed_tokens.weight', 'norm.weight', 'layers.0.input_layernorm.weight']]
        names += [head, 'vision_tower.blocks.0.weight', 'mm_projector.proj.0.weight']
        (Path(directory) / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'fake.safetensors' for k in names}}))
        return set(names)

    def test_registration(self):
        self.assertTrue(self.assert_registration_import)
        self.assertEqual(self.registrations[0]['source'], 'KimiK25ForConditionalGeneration')

    def test_provider_uses_nested_config_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_index(directory)
            config = self.config(directory)
            bridge = self.ns['KimiK25TextBridge']()
            provider = bridge.provider_bridge(SimpleNamespace(config=config))
            self.assertEqual(provider.received.num_hidden_layers, 4)
            self.assertEqual(provider.received.dtype, 'bfloat16')
            self.assertEqual(provider.seq_length, 32768)
            self.assertIsNot(provider.received, config.text_config)
            self.assertFalse(hasattr(config.text_config, 'dtype'))

    def test_all_supported_layouts_rewrite_text_mappings_only(self):
        for prefix in ['language_model.model.', 'language_model.', 'model.language_model.model.', 'model.language_model.']:
            with tempfile.TemporaryDirectory() as directory:
                self.write_index(directory, prefix)
                bridge = self.ns['KimiK25TextBridge']()
                bridge.hf_config = self.config(directory)
                mappings = bridge.mapping_registry()
                by_name = {m.megatron_param: m.hf_param for m in mappings.mappings}
                self.assertEqual(by_name['embedding.word_embeddings.weight'], prefix + 'embed_tokens.weight')
                self.assertEqual(by_name['output_layer.weight'], 'language_model.lm_head.weight')
                gate_up = by_name['decoder.layers.*.mlp.experts.linear_fc1.weight*']
                self.assertEqual(gate_up['gate'], prefix + 'layers.*.mlp.experts.*.gate_proj.weight')
                self.assertEqual(gate_up['up'], prefix + 'layers.*.mlp.experts.*.up_proj.weight')
                self.assertEqual(by_name['decoder.layers.*.mlp.router.expert_bias'], prefix + 'layers.*.mlp.gate.e_score_correction_bias')
                for item in by_name.values():
                    names = [item] if isinstance(item, str) else list(item.values())
                    self.assertTrue(all('vision' not in n and 'projector' not in n for n in names))

    def test_ambiguous_or_unrecognized_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ns['_text_layout']({'model.embed_tokens.weight', 'lm_head.weight'})
        with tempfile.TemporaryDirectory() as directory:
            keys = self.write_index(directory)
            keys.add('lm_head.weight')
            with self.assertRaises(ValueError):
                self.ns['_text_layout'](keys)

    def test_missing_actor_tensor_or_mapping_cannot_be_silently_skipped(self):
        mapping = SimpleNamespace(megatron_param='actor.weight', hf_param='language_model.weight')
        registry = self.ns['KimiTextMappingRegistry'](set(), mapping)
        with self.assertRaisesRegex(ValueError, 'missing checkpoint tensors'):
            registry.megatron_to_hf_lookup('actor.weight')
        with self.assertRaisesRegex(ValueError, 'No Kimi text mapping'):
            registry.megatron_to_hf_lookup('unknown.weight')
        registry.checkpoint_keys.add('language_model.weight')
        self.assertIs(registry.megatron_to_hf_lookup('actor.weight'), mapping)

    def test_unsupported_configs_fail_explicitly(self):
        config = self.config('/missing')
        config.quantization_config = {'quant_method': 'fp8'}
        with self.assertRaisesRegex(ValueError, 'unquantized'):
            self.ns['_text_config'](config)
        config.quantization_config = None
        config.text_config.model_type = 'unrelated_model'
        with self.assertRaisesRegex(ValueError, 'DeepSeek-V3'):
            self.ns['_text_config'](config)
        with self.assertRaisesRegex(NotImplementedError, 'Standalone'):
            self.ns['KimiK25TextBridge'].megatron_to_hf_config(None)


if __name__ == '__main__':
    unittest.main()
