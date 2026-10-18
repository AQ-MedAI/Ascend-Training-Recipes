"""Exercise the actual K26 optional-TE import and non-FP8 checkpoint branches on CPU."""

import ast
import contextlib
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472"


class K26OptionalTETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = Path(os.environ.get("K26_MEGATRON_CHECKOUT", str(ROOT.parent / "Megatron-LM-aq")))
        relative = "megatron/core/tensor_parallel/random.py"
        result = subprocess.run(["git", "-C", str(checkout), "show", f"{COMMIT}:{relative}"],
                                text=True, capture_output=True)
        if result.returncode:
            raise unittest.SkipTest("Provide the K26 baseline via K26_MEGATRON_CHECKOUT")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / relative
            target.parent.mkdir(parents=True)
            target.write_text(result.stdout)
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            subprocess.run(["git", "-C", directory, "apply", str(ROOT / "patches/megatron-lm/k26/ascend-optional-te.patch")],
                           check=True)
            cls.tree = ast.parse(target.read_text())

    def import_backend(self, complete):
        modules = {}
        for name in ("transformer_engine", "transformer_engine.pytorch", "transformer_engine.pytorch.distributed",
                     "transformer_engine.pytorch.fp8"):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        if complete:
            modules["transformer_engine.pytorch.distributed"].activation_recompute_forward = object()
            modules["transformer_engine.pytorch.fp8"].FP8GlobalStateManager = object()
            modules["transformer_engine.pytorch.fp8"].fp8_autocast = object()
        block = next(node for node in self.tree.body if isinstance(node, ast.Try)
                     and any(isinstance(item, ast.Import) and item.names[0].name == "transformer_engine"
                             for item in node.body))
        namespace = {}
        with patch.dict(sys.modules, modules):
            exec(compile(ast.Module(body=[block], type_ignores=[]), "<patched TE import>", "exec"), namespace)
        return namespace

    def constructor(self, namespace):
        original = next(node for node in self.tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "CheckpointWithoutOutput")
        initializer = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        node = ast.ClassDef(name=original.name, bases=[], keywords=[], body=[initializer], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<checkpoint init>", "exec"), namespace)
        return namespace[original.name]

    def test_incomplete_shim_allows_non_fp8_and_rejects_fp8(self):
        namespace = self.import_backend(False)
        self.assertFalse(namespace["HAVE_TE"])
        constructor = self.constructor(namespace)
        self.assertFalse(constructor().fp8)
        self.assertFalse(constructor(False).fp8)
        self.assertFalse(constructor(None).fp8)
        for mode in (True, "hybrid"):
            with self.assertRaisesRegex(RuntimeError, "complete Transformer Engine"):
                constructor(mode)

    def test_complete_backend_retains_fp8_selection(self):
        namespace = self.import_backend(True)
        self.assertTrue(namespace["HAVE_TE"])
        self.assertTrue(self.constructor(namespace)("hybrid").fp8)

    def test_non_fp8_forward_does_not_call_te(self):
        namespace = self.import_backend(False)
        checkpoint = self.constructor(namespace)(False)
        original = next(node for node in self.tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "CheckpointWithoutOutputFunction")
        forward = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        namespace.update(torch=types.SimpleNamespace(no_grad=contextlib.nullcontext),
                         contextlib=contextlib, detach_variable=lambda values: values)
        exec(compile(ast.Module(body=[forward], type_ignores=[]), "<actual checkpoint forward>", "exec"), namespace)
        context = types.SimpleNamespace(save_for_backward=lambda *args: None)
        self.assertEqual(namespace["forward"].__func__(context, lambda value: value * 3, checkpoint, 2), 6)
        self.assertFalse(context.fp8)


if __name__ == "__main__":
    unittest.main()
