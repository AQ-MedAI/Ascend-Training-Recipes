"""Exercise the crop CLI implementation with tiny real safetensors checkpoints."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "v4_crop", ROOT / "recipes/deepseek-v4/prepare_dsv4_flash_bf16_cut_model.py"
)
crop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crop)


class CropMTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = (Path(self.tmp.name) / "source").resolve()
        self.source.mkdir()
        self.output = Path(self.tmp.name) / "crop"
        self.config = dict(
            num_hidden_layers=6,
            num_nextn_predict_layers=1,
            compress_ratios=[0, 0, 4, 128, 4, 128, 4],
            num_hash_layers=3,
            quantization_config={"quant_method": "fp8"},
        )
        crop.dump_json(self.source / "config.json", self.config)
        self.weights = {"embed.weight": torch.ones(8, 4), "head.weight": torch.ones(8, 4)}
        self.weights.update({f"layers.{i}.attn.wq_a.weight": torch.full((4, 4), float(i)) for i in range(6)})
        for suffix, shape in {
            "e_proj.weight": (4, 4),
            "h_proj.weight": (4, 4),
            "enorm.weight": (4,),
            "hnorm.weight": (4,),
            "norm.weight": (4,),
            "hc_head_fn": (2, 8),
            "hc_head_base": (2,),
            "hc_head_scale": (1,),
            "attn.wq_a.weight": (4, 4),
        }.items():
            self.weights["mtp.0." + suffix] = torch.arange(torch.tensor(shape).prod()).float().reshape(shape)
        # Packed MXFP4 with both nibbles=2 -> 1.0, scaled to 2.0.
        self.weights["mtp.0.ffn.experts.0.w1.weight"] = torch.full((2, 16), 0x22, dtype=torch.int8)
        self.weights["mtp.0.ffn.experts.0.w1.scale"] = torch.full((2, 1), 2.0)
        save_file(self.weights, self.source / "weights.safetensors")
        crop.dump_json(
            self.source / "model.safetensors.index.json",
            {"weight_map": {key: "weights.safetensors" for key in self.weights}},
        )

    def build(self, keep_mtp):
        crop.materialize_bf16_cut(self.source, self.output, 4, force=False, keep_mtp=keep_mtp)
        return load_file(self.output / "weights.safetensors")

    def test_default_omits_mtp_and_truncates_layer_settings(self):
        actual = self.build(False)
        self.assertFalse(any(key.startswith("mtp.") for key in actual))
        self.assertNotIn("layers.4.attn.wq_a.weight", actual)
        config = crop.load_json(self.output / "config.json")
        self.assertEqual(config["num_nextn_predict_layers"], 0)
        self.assertEqual(config["compress_ratios"], [0, 0, 4, 128])

    def test_keep_mtp_preserves_names_values_and_mtp_layer_settings(self):
        actual = self.build(True)
        self.assertNotIn("layers.4.attn.wq_a.weight", actual)
        self.assertNotIn("mtp.0.ffn.experts.0.w1.scale", actual)
        self.assertTrue(torch.equal(actual["mtp.0.e_proj.weight"], self.weights["mtp.0.e_proj.weight"].bfloat16()))
        self.assertTrue(
            torch.equal(actual["mtp.0.ffn.experts.0.w1.weight"], torch.full((2, 32), 2.0, dtype=torch.bfloat16))
        )
        config = crop.load_json(self.output / "config.json")
        self.assertEqual(config["num_nextn_predict_layers"], 1)
        self.assertEqual(config["mtp_num_layers"], 1)
        self.assertEqual(config["compress_ratios"], [0, 0, 4, 128, 4])
        self.assertEqual(crop.load_json(self.source / "config.json"), self.config)

    def test_cache_distinguishes_mtp_and_reuses_matching_output(self):
        self.build(False)
        self.assertFalse(crop.is_complete_output(self.output, self.source, 4, keep_mtp=True))
        self.build(True)
        self.assertTrue(crop.is_complete_output(self.output, self.source, 4, keep_mtp=True))
        before = (self.output / "weights.safetensors").stat().st_mtime_ns
        self.build(True)
        self.assertEqual(before, (self.output / "weights.safetensors").stat().st_mtime_ns)
        self.assertFalse(crop.is_complete_output(self.output, self.source, 4, keep_mtp=False))
        self.build(False)
        self.assertTrue(crop.is_complete_output(self.output, self.source, 4, keep_mtp=False))

    def test_incomplete_source_is_rejected_before_replacing_output(self):
        self.build(False)
        marker_before = (self.output / ".verl_bf16_cut_manifest.json").read_bytes()
        index = crop.load_json(self.source / "model.safetensors.index.json")
        del index["weight_map"]["mtp.0.h_proj.weight"]
        crop.dump_json(self.source / "model.safetensors.index.json", index)
        with self.assertRaisesRegex(ValueError, "h_proj"):
            self.build(True)
        self.assertEqual(marker_before, (self.output / ".verl_bf16_cut_manifest.json").read_bytes())

    def test_missing_source_mtp_and_invalid_crop_count_are_rejected(self):
        config = dict(self.config, num_nextn_predict_layers=0)
        crop.dump_json(self.source / "config.json", config)
        with self.assertRaisesRegex(ValueError, "source checkpoint with MTP"):
            self.build(True)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "num_layers"):
            crop.materialize_bf16_cut(self.source, self.output, 7, False)

    def test_source_crop_manifest_is_not_overwritten_through_a_symlink(self):
        original = b'{"source": "upstream-model"}\n'
        marker = self.source / ".verl_bf16_cut_manifest.json"
        marker.write_bytes(original)
        self.build(True)
        self.assertEqual(marker.read_bytes(), original)
        self.assertFalse((self.output / marker.name).is_symlink())

    def test_backbone_only_ratio_list_keeps_default_dense_mtp(self):
        config = dict(self.config, compress_ratios=[0, 0, 4, 128, 4, 128])
        crop.adjust_cut_config(config, 4, keep_mtp=True)
        self.assertEqual(config["compress_ratios"], [0, 0, 4, 128])
        self.assertEqual(config["num_nextn_predict_layers"], 1)


if __name__ == "__main__":
    unittest.main()
