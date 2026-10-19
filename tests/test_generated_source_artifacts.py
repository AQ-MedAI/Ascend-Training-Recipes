"""Keep source verification strict while allowing named editable build artifacts."""

import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "framework_sources_for_artifacts", ROOT / "tools/prepare_framework_sources.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeneratedArtifactsTest(unittest.TestCase):
    def repository(self, path):
        subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
        source = path / "setup.py"
        source.write_text("SOURCE = True\n")
        subprocess.run(["git", "-C", str(path), "add", "setup.py"], check=True)
        return {"files": {"setup.py": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "executable": False}}}

    def test_only_named_extension_is_allowed_after_editable_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.repository(root)
            extension = "ascendspeed_te_ops" + importlib.machinery.EXTENSION_SUFFIXES[0]
            (root / extension).write_bytes(b"generated extension fixture")
            with self.assertRaisesRegex(RuntimeError, "extra=.*ascendspeed_te_ops"):
                module.verify(root, manifest, tracked=True)
            module.verify(root, manifest, tracked=True, generated_files=[extension])
            (root / "unrecognized.so").write_bytes(b"not allowed")
            with self.assertRaisesRegex(RuntimeError, "extra=.*unrecognized.so"):
                module.verify(root, manifest, tracked=True, generated_files=[extension])

    def test_generated_allowlist_cannot_hide_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.repository(root)
            with self.assertRaisesRegex(ValueError, "must not mask"):
                module.verify(root, manifest, tracked=True, generated_files=["setup.py"])
            (root / "setup.py").write_text("SOURCE = False\n")
            with self.assertRaisesRegex(RuntimeError, "checksum or mode mismatch"):
                module.verify(root, manifest, tracked=True, generated_files=["ascendspeed_te_ops.so"])
