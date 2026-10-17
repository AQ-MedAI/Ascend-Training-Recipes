"""Check shared source selection and model entrypoints without importing NPU packages."""

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceContractsTest(unittest.TestCase):
    def test_shared_profiles(self):
        from sys import path

        path.insert(0, str(ROOT / "tools"))
        from patched_sources import load_source

        components = json.loads((ROOT / "dependencies/registry.json").read_text())["components"]
        for name, component in components.items():
            with self.subTest(component=name):
                v4, _ = load_source(ROOT, name, component, "v4")
                v41, _ = load_source(ROOT, name, component, "v41")
                self.assertEqual(v4, v41)

    def test_model_provider_paths(self):
        implementation = ROOT / "third_party/verl/verl/workers/engine/mindspeed/transformer_impl.py"
        if not implementation.is_file():
            self.skipTest("Initialize shared recipe sources first")
        manifest = json.loads((ROOT / "dependencies/mindspeed-llm-sources.json").read_text())
        files = manifest["profiles"]["v41"]["sources"]["MindSpeed-LLM"]["files"]
        providers = [
            node.module
            for node in ast.walk(ast.parse(implementation.read_text()))
            if isinstance(node, ast.ImportFrom) and any(alias.name == "model_provider" for alias in node.names)
        ]
        self.assertEqual(len(providers), 2)
        for module in providers:
            self.assertIn(module.replace(".", "/") + ".py", files)


if __name__ == "__main__":
    unittest.main()
