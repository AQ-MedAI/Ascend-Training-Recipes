"""Exercise the actual K26 optional-TE import and non-FP8 checkpoint branches on CPU."""

import ast
import __future__
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
            for name in (relative, "megatron/core/fp8_utils.py", "megatron/core/fp4_utils.py"):
                original = subprocess.check_output(["git", "-C", str(checkout), "show", f"{COMMIT}:{name}"], text=True)
                target = Path(directory) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(original)
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            for patch_name in ("ascend-optional-te.patch", "ascend-fp8-utils.patch"):
                subprocess.run(["git", "-C", directory, "apply", str(ROOT / "patches/megatron-lm/k26" / patch_name)], check=True)
            cls.tree = ast.parse((Path(directory) / relative).read_text())
            cls.fp8_source = (Path(directory) / "megatron/core/fp8_utils.py").read_text()
            cls.fp4_source = (Path(directory) / "megatron/core/fp4_utils.py").read_text()

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


    def test_complete_fp8_utilities_backend_is_recognized(self):
        modules = {}
        for name in ("transformer_engine", "transformer_engine.pytorch", "transformer_engine.pytorch.fp8",
                     "transformer_engine.pytorch.tensor"):
            modules[name] = types.ModuleType(name)
            modules[name].__path__ = []
        manager = object()
        tensor_class = type("QuantizedTensor", (), {})
        modules["transformer_engine.pytorch.fp8"].FP8GlobalStateManager = manager
        modules["transformer_engine.pytorch.tensor"].QuantizedTensor = tensor_class
        block = next(node for node in ast.parse(self.fp8_source).body if isinstance(node, ast.Try)
                     and any(isinstance(item, ast.Import) and item.names[0].name == "transformer_engine"
                             for item in node.body))
        namespace = {"is_te_min_version": lambda version: True, "HAVE_TE": False}
        with patch.dict(sys.modules, modules):
            exec(compile(ast.Module(body=[block], type_ignores=[]), "<actual TE backend detection>", "exec"), namespace)
        self.assertTrue(namespace["HAVE_TE"])
        self.assertIs(namespace["FP8GlobalStateManager"], manager)
        self.assertIs(namespace["FP8_TENSOR_CLASS"], tensor_class)

    def test_full_utilities_import_with_mindspeed_style_shim(self):
        modules = {}
        def module(name, **attributes):
            value = types.ModuleType(name)
            value.__dict__.update(attributes)
            modules[name] = value
            return value
        for name in ("megatron", "megatron.core", "megatron.core.transformer", "megatron.core.extensions"):
            module(name).__path__ = []
        module("torch")
        module("megatron.core.enums", Fp4Recipe=object, Fp8Recipe=object)
        module("megatron.core.tensor_parallel", ColumnParallelLinear=object, RowParallelLinear=object,
               gather_from_sequence_parallel_region=lambda x, **kw: x,
               reduce_scatter_to_sequence_parallel_region=lambda x, **kw: x)
        module("megatron.core.transformer.transformer_config", TransformerConfig=object)
        def unexpected_version(*args, **kwargs):
            raise AssertionError("Incomplete TE must not enter version-gated quantization paths")
        module("megatron.core.utils", get_te_version=unexpected_version, is_te_min_version=unexpected_version)
        module("megatron.core.extensions.transformer_engine")
        module("transformer_engine").__path__ = []
        # Like the real MindSpeed dummy, this module has no __path__ or fp8 submodule.
        module("transformer_engine.pytorch")
        fp8 = module("megatron.core.fp8_utils")
        fp4 = module("megatron.core.fp4_utils")
        with patch.dict(sys.modules, modules):
            exec(compile(self.fp8_source, "<complete patched fp8_utils>", "exec",
                         flags=__future__.annotations.compiler_flag), fp8.__dict__)
            exec(compile(self.fp4_source, "<complete patched fp4_utils>", "exec",
                         flags=__future__.annotations.compiler_flag), fp4.__dict__)
        self.assertFalse(fp8.HAVE_TE)
        self.assertFalse(fp4.HAVE_TE)
        self.assertFalse(fp8.is_float8tensor(object()))
        self.assertFalse(fp4.is_nvfp4tensor(object()))
        config = types.SimpleNamespace(fp8=None, fp8_param=False, fp4=None, fp4_param=False)
        for utility, precision in ((fp8, "fp8"), (fp4, "fp4")):
            with getattr(utility, "get_" + precision + "_context")(config):
                pass
            self.assertIsNone(getattr(utility, "get_" + precision + "_recipe")(config))
            setattr(config, precision, "enabled")
            with self.assertRaisesRegex(RuntimeError, "complete Transformer Engine"):
                getattr(utility, "get_" + precision + "_context")(config)
            setattr(config, precision, None)
            setattr(config, precision + "_param", True)
            with self.assertRaisesRegex(RuntimeError, "complete Transformer Engine"):
                getattr(utility, "get_" + precision + "_context")(config, is_init=True)
            setattr(config, precision + "_param", False)
        value = types.SimpleNamespace(data="before")
        with self.assertRaisesRegex(RuntimeError, "Invalid Transformer Engine"):
            fp8.modify_underlying_storage(value, "after")
        self.assertIsNone(fp8.post_all_gather_processing([value]))


if __name__ == "__main__":
    unittest.main()
