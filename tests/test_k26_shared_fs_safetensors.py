"""Apply the Bridge patch and exercise shared-filesystem byte serialization."""

import ast
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "src/megatron/bridge/models/hf_pretrained/state.py"


class SharedFsSafetensorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT.parent / "Megatron-Bridge-aq" / RELATIVE
        if not source.is_file():
            raise unittest.SkipTest("Provide pinned K26 Bridge source")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / RELATIVE
            target.parent.mkdir(parents=True)
            target.write_bytes(source.read_bytes())
            subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "apply",
                    str(ROOT / "patches/megatron-bridge/ascend-shared-fs-safetensors.patch"),
                ],
                check=True,
            )
            cls.tree = ast.parse(target.read_text())
        helper = next(
            n for n in cls.tree.body if isinstance(n, ast.FunctionDef) and n.name == "_save_safetensors_sequential"
        )
        ns = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<patched Bridge writer>", "exec"), ns)
        cls.save = staticmethod(ns[helper.name])

    def test_roundtrip_preserves_tensors_and_metadata_without_native_file_writer(self):
        tensors = {
            "fp32": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "bf16": torch.ones(5, dtype=torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            with patch("safetensors.torch.save_file", side_effect=AssertionError("native writer used")):
                self.save(tensors, path, metadata={"format": "pt"})
            restored = load_file(path)
            for name, tensor in tensors.items():
                torch.testing.assert_close(restored[name], tensor)
            with safe_open(path, framework="pt", device="cpu") as reader:
                self.assertEqual(reader.metadata(), {"format": "pt"})

    def test_write_errors_propagate(self):
        with patch("builtins.open", side_effect=OSError(95, "Operation not supported")):
            with self.assertRaises(OSError) as caught:
                self.save({"weight": torch.ones(2)}, "/unused/model.safetensors")
        self.assertEqual(caught.exception.errno, 95)

    def test_all_single_sharded_and_distributed_save_sites_bind_the_helper(self):
        bindings = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Name)
            and node.value.id == "_save_safetensors_sequential"
        ]
        self.assertEqual(len(bindings), 2)
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "save_file"
        ]
        self.assertEqual(len(calls), 5)
        self.assertFalse(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "safetensors.torch"
                and any(alias.name == "save_file" for alias in node.names)
                for node in ast.walk(self.tree)
            )
        )
