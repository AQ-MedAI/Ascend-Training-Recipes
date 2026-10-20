"""Check K26 source composition and fail before mutation on invalid image inputs."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_k26_image_sources import image_sources, prepare_image


class K26ImageSourcesTest(unittest.TestCase):
    def test_complete_pinned_series_and_shared_inference_unchanged(self):
        sources = image_sources(ROOT)
        self.assertEqual(set(sources), {"Megatron-LM", "Megatron-Bridge", "MindSpeed", "verl"})
        expected = {"Megatron-LM": 6, "Megatron-Bridge": 6, "MindSpeed": 12, "verl": 3}
        for name, source in sources.items():
            with self.subTest(name=name):
                self.assertEqual(len(source["patches"]), expected[name])
                base = ROOT / ("dependencies" if name == "verl" else "docker/k26")
                for item in source["patches"]:
                    self.assertEqual(hashlib.sha256((base / item["path"]).read_bytes()).hexdigest(), item["sha256"])
        self.assertEqual(sources["Megatron-LM"]["commit"], "96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472")
        self.assertEqual(sources["Megatron-Bridge"]["commit"], "c15303e05b3d48505e4b1b0047faa0c321368856")
        shared = json.loads((ROOT / "dependencies/mindspeed-sources.json").read_text())
        self.assertEqual(
            sources["MindSpeed"]["reference_files"],
            shared["profiles"]["v41"]["sources"]["MindSpeed"]["reference_files"],
        )
        self.assertNotIn("vllm", sources)
        self.assertNotIn("vllm-ascend", sources)

    def inputs(self, base):
        (base / "docker/k26").mkdir(parents=True)
        (base / "dependencies").mkdir()
        source = {"commit": "a" * 40, "patches": [{"path": "missing.patch", "sha256": "0" * 64}]}
        (base / "docker/k26/sources.json").write_text(
            json.dumps({"sources": {name: source for name in ("Megatron-LM", "Megatron-Bridge", "MindSpeed")}})
        )
        (base / "dependencies/verl-sources.json").write_text(
            json.dumps({"profiles": {"k26": {"sources": {"verl": source}}}})
        )

    def test_existing_checkout_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.inputs(root)
            target = root / "sources/Megatron-LM"
            target.mkdir(parents=True)
            keep = target / "user.py"
            keep.write_text("user content")
            with patch("prepare_k26_image_sources.prepare") as prepare:
                with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                    prepare_image(root / "sources", root)
                prepare.assert_not_called()
            self.assertEqual(keep.read_text(), "user content")

    def test_missing_patch_fails_before_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.inputs(root)
            with patch("prepare_k26_image_sources.prepare") as prepare:
                with self.assertRaises(FileNotFoundError):
                    prepare_image(root / "sources", root)
                prepare.assert_not_called()
            self.assertFalse((root / "sources").exists())

    def test_activation_selects_k26_preflight_from_image_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "recipes/kimi-k2.6").mkdir(parents=True)
            shutil.copyfile(ROOT / "tools/activate.sh", root / "tools/activate.sh")
            (root / "recipes/kimi-k2.6/runtime_env.sh").write_text("export K26_PREFLIGHT_SELECTED=yes\n")
            environment = dict(os.environ, RECIPE_PROFILE="k26", K26_TEST_ROOT=str(root))
            output = subprocess.check_output(
                ["bash", "-c", 'source "$K26_TEST_ROOT/tools/activate.sh"; printf "%s" "$K26_PREFLIGHT_SELECTED"'],
                env=environment,
                text=True,
            )
            self.assertEqual(output, "yes")

    def test_hidden_rpc_patch_is_present(self):
        source = image_sources(ROOT)["verl"]
        names = [Path(item["path"]).name for item in source["patches"]]
        self.assertIn("k26-full-vocab-rpc.patch", names)
        content = (ROOT / "recipes/kimi-k2.6/multi_reward.py").read_text()
        self.assertIn('"pred": gsm8k_extract_solution', content)


if __name__ == "__main__":
    unittest.main()
