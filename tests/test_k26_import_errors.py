"""CPU regression for partially initialized optional K26 engines."""

import __future__
import ast
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class K26ImportErrorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((ROOT / "dependencies/verl-sources.json").read_text())
        commit = manifest["profiles"]["k26"]["sources"]["verl"]["commit"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for filename in ("base.py", "__init__.py"):
                relative = "verl/workers/engine/" + filename
                source = subprocess.run(
                    ["git", "-C", str(ROOT / "third_party/verl"), "show", f"{commit}:{relative}"],
                    capture_output=True, text=True,
                )
                if source.returncode:
                    raise unittest.SkipTest("K26 upstream Git objects are needed for this regression")
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source.stdout)
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            subprocess.run(["git", "-C", directory, "apply", str(ROOT / "patches/verl/k26-engine-registration.patch")],
                           check=True)
            cls.base = ast.parse((target / "verl/workers/engine/base.py").read_text())
            cls.init = ast.parse((target / "verl/workers/engine/__init__.py").read_text())

    def registry(self, device):
        node = next(node for node in self.base.body if isinstance(node, ast.ClassDef) and node.name == "EngineRegistry")
        namespace = {"BaseEngine": object, "get_device_name": lambda: device}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<patched registry>", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        return namespace["EngineRegistry"]

    def test_first_failure_is_retained_and_never_retried(self):
        handlers = [node for node in self.init.body if isinstance(node, ast.Try)
                    and any(isinstance(item, ast.ImportFrom) and item.module in ("mindspeed", "megatron")
                            for item in node.body)]
        first = ImportError("original MindSpeed failure")
        second = ImportError("later Megatron failure")
        namespace = {"__all__": []}
        def fail(name, *args, **kwargs):
            raise first if name == "mindspeed" else second
        with patch("builtins.__import__", side_effect=fail):
            exec(compile(ast.Module(body=handlers, type_ignores=[]), "<import handlers>", "exec"), namespace)
        package = types.ModuleType("verl.workers.engine")
        package._mindspeed_import_error = namespace["_mindspeed_import_error"]
        package._megatron_import_error = namespace["_megatron_import_error"]
        registry = self.registry("npu")
        registry._engines = {"language_model": {"fsdp": {"npu": object}}}
        with patch.dict(sys.modules, {"verl.workers.engine": package}), patch("importlib.import_module") as imports:
            for _ in range(2):
                with self.assertRaises(ImportError) as caught:
                    registry.get_engine_cls("language_model", "megatron")
                self.assertIs(caught.exception, first)
            imports.assert_not_called()
            self.assertIs(registry.get_engine_cls("language_model", "fsdp"), object)
            with self.assertRaises(ImportError) as caught:
                self.registry("cuda").get_engine_cls("language_model", "megatron")
            self.assertIs(caught.exception, second)

    def test_successful_registration_is_usable(self):
        package = types.ModuleType("verl.workers.engine")
        package._mindspeed_import_error = package._megatron_import_error = None
        registry = self.registry("npu")
        registry._engines = {"language_model": {"megatron": {"npu": object}}}
        with patch.dict(sys.modules, {"verl.workers.engine": package}):
            self.assertIs(registry.get_engine_cls("language_model", "megatron"), object)


if __name__ == "__main__":
    unittest.main()
