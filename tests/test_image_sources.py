"""Offline integration checks for the image's seven source repositories."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_image_sources import image_sources, prepare_image


class ImageSourcesTest(unittest.TestCase):
    def test_real_profiles_and_patch_checksums(self):
        frameworks, training = image_sources(ROOT)
        self.assertEqual(set(frameworks["sources"]) | set(training),
                         {"Megatron-LM", "vllm", "vllm-ascend", "MindSpeed", "MindSpeed-LLM", "mbridge", "verl"})
        from patched_sources import load_source
        registry = json.loads((ROOT / "dependencies/registry.json").read_text())["components"]
        for name, component in registry.items():
            self.assertEqual(load_source(ROOT, name, component, "v4"), load_source(ROOT, name, component, "v41"))

    def test_activation_uses_image_and_honors_overrides(self):
        import os
        environment = os.environ.copy()
        for name in ['MEGATRON_ROOT', 'VLLM_ROOT', 'VLLM_ASCEND_ROOT', 'MINDSPEED_ROOT',
                     'MINDSPEED_LLM_ROOT', 'MBRIDGE_ROOT', 'VERL_ROOT', 'DEPENDENCY_ROOT']:
            environment.pop(name, None)
        environment['PYTHON_BIN'] = sys.executable
        for profile in ['v4', 'v41']:
            result = subprocess.check_output(
                ['bash', '-c', 'source "$1/tools/activate.sh" "$2"; printf "%s\\n" "$VERL_ROOT" "$MEGATRON_ROOT"',
                 'check', str(ROOT), profile], env=environment, text=True)
            self.assertEqual(result.splitlines(), ['/opt/ascend-training/third_party/verl',
                                                  '/opt/ascend-training/third_party/Megatron-LM'])
        environment['VERL_ROOT'] = '/development/verl'
        result = subprocess.check_output(
            ['bash', '-c', 'source "$1/tools/activate.sh" v41; printf "%s" "$VERL_ROOT"', 'check', str(ROOT)],
            env=environment, text=True)
        self.assertEqual(result, '/development/verl')

    def test_all_seven_reconstructed_and_divergence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            upstream = base / "upstream"
            upstream.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(upstream), *args], text=True)
            git("init", "-q")
            (upstream / "model.txt").write_text("base\n")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
            commit = git("rev-parse", "HEAD").strip()
            (upstream / "model.txt").write_text("adapted\n")
            patch = git("diff")
            inputs = base / "inputs"
            (inputs / "dependencies").mkdir(parents=True)
            (inputs / "patches").mkdir()
            (inputs / "patches/adaptation.patch").write_text(patch)
            def files(content):
                return {"model.txt": {"sha256": hashlib.sha256(content.encode()).hexdigest(), "executable": False}}
            source = {"repository": str(upstream), "commit": commit,
                      "patches": [{"path": "../patches/adaptation.patch", "sha256": hashlib.sha256(patch.encode()).hexdigest()}],
                      "reference_files": files("base\n"), "files": files("adapted\n"), "gitlinks": {}}
            def write(name, data):
                (inputs / "dependencies" / name).write_text(json.dumps(data))
            write("framework-sources.json", {"sources": {name: source for name in ["Megatron-LM", "vllm", "vllm-ascend"]}})
            components = {}
            manifests = {}
            for name in ["MindSpeed", "MindSpeed-LLM", "mbridge", "verl"]:
                manifest = {"profiles": {p: {"sources": {name: copy.deepcopy(source)}} for p in ["v4", "v41"]}}
                manifests[name] = manifest
                write(name + ".json", manifest)
                components[name] = {"repository": str(upstream), "source_manifest": "dependencies/" + name + ".json",
                                    "profiles": {p: {"commit": commit} for p in ["v4", "v41"]}}
            write("registry.json", {"components": components})
            prepare_image(base / "sources", inputs)
            for name in ["Megatron-LM", "vllm", "vllm-ascend", *components]:
                self.assertEqual((base / "sources" / name / "model.txt").read_text(), "adapted\n")
            manifests["verl"]["profiles"]["v4"]["sources"]["verl"]["commit"] = "wrong"
            write("verl.json", manifests["verl"])
            with self.assertRaisesRegex(RuntimeError, "sources differ"):
                prepare_image(base / "blocked", inputs)
            self.assertFalse((base / "blocked").exists())


if __name__ == "__main__":
    unittest.main()
