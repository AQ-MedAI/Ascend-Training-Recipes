"""Check two-layer checkpoint materialization without loading training backends."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("k26_cut_model", ROOT / "recipes/kimi-k2.6/prepare_kimi_k26_cut_model.py")
cut = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cut)


class CutModelTest(unittest.TestCase):
    def source(self, root):
        root.mkdir()
        config = {
            "text_config": {
                "num_hidden_layers": 4,
                "n_routed_experts": 8,
                "hidden_size": 16,
                "num_nextn_predict_layers": 1,
            }
        }
        (root / "config.json").write_text(json.dumps(config))
        tensors = {f"language_model.model.layers.{i}.mlp.experts.0.weight": torch.ones(2, 2) * i for i in range(4)}
        tensors.update(
            {
                "language_model.model.embed_tokens.weight": torch.ones(3, 2),
                "vision_tower.weight": torch.ones(2, 2),
                "model.mtp.weight": torch.ones(2, 2),
            }
        )
        save_file(tensors, root / "model.safetensors")
        return config

    def test_default_two_layers_preserves_dimensions_and_other_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "two"
            original = self.source(source)
            cut.prepare(source, output)
            config = json.loads((output / "config.json").read_text())["text_config"]
            self.assertEqual(config["num_hidden_layers"], 2)
            self.assertEqual(config["n_routed_experts"], 8)
            self.assertEqual(config["hidden_size"], 16)
            self.assertEqual(config["num_nextn_predict_layers"], 0)
            weights = cut.weight_map(output)
            self.assertIn("vision_tower.weight", weights)
            self.assertIn("language_model.model.embed_tokens.weight", weights)
            self.assertFalse(any(cut.MTP.search(k) for k in weights))
            self.assertEqual({cut.layer_id(k) for k in weights if cut.layer_id(k) is not None}, {0, 1})
            cut.prepare(None, output)
            self.assertEqual(json.loads((source / "config.json").read_text()), original)

    def test_existing_four_layer_output_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "four"
            self.source(source)
            cut.prepare(source, output, count=4)
            before = (output / "config.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "Expected 2 decoder layers"):
                cut.prepare(None, output)
            self.assertEqual((output / "config.json").read_bytes(), before)
