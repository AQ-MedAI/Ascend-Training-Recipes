"""Check that source verification rejects incomplete or modified checkouts."""

import hashlib
import json
import subprocess
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "framework_sources", Path(__file__).resolve().parents[1] / "prepare_framework_sources.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourceVerificationTest(unittest.TestCase):
    def test_inventory_content_mode_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.py"
            source.write_bytes(b"original\n")
            source.chmod(0o644)
            expected = {
                "files": {
                    "source.py": {
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "executable": False,
                    }
                }
            }
            module.verify(root, expected)
            source.write_bytes(b"changed\n")
            with self.assertRaises(RuntimeError):
                module.verify(root, expected)
            source.write_bytes(b"original\n")
            source.chmod(0o755)
            with self.assertRaises(RuntimeError):
                module.verify(root, expected)
            source.chmod(0o644)
            extra = root / "extra"
            extra.write_text("unexpected")
            with self.assertRaises(RuntimeError):
                module.verify(root, expected)
            extra.unlink()
            source.unlink()
            source.symlink_to("/missing-source-target")
            with self.assertRaises(RuntimeError):
                module.verify(root, expected)


class PatchSequenceTest(unittest.TestCase):
    def test_local_commit_reference_and_backport_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            upstream = base / "upstream"
            upstream.mkdir()

            def git(*args):
                return subprocess.check_output(["git", "-C", str(upstream), *args], text=True)

            git("init", "-q")
            (upstream / "model.py").write_text("base\n")
            git("add", ".")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "base",
            )
            commit = git("rev-parse", "HEAD").strip()
            (upstream / "model.py").write_text("v4\n")
            local_patch = git("diff", "--binary")
            (base / "local.patch").write_text(local_patch)
            git("add", ".")
            (upstream / "model.py").write_text("v4 and v41\n")
            backport = git("diff", "--binary")
            (base / "backport.patch").write_text(backport)

            def inventory(content):
                return {
                    "model.py": {
                        "sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "executable": False,
                    }
                }

            source = {
                "repository": str(upstream),
                "commit": commit,
                "patch": "local.patch",
                "patch_sha256": hashlib.sha256(local_patch.encode()).hexdigest(),
                "backport_patch": "backport.patch",
                "backport_patch_sha256": hashlib.sha256(backport.encode()).hexdigest(),
                "reference_files": inventory("v4\n"),
                "files": inventory("v4 and v41\n"),
            }
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"sources": {"framework": source}}))
            module.prepare(base / "output", manifest)
            self.assertEqual((base / "output/framework/model.py").read_text(), "v4 and v41\n")
            # A wrong reference receipt must fail even if the final inventory is correct.
            source["reference_files"] = inventory("wrong\n")
            manifest.write_text(json.dumps({"sources": {"framework": source}}))
            with self.assertRaisesRegex(RuntimeError, "checksum or mode"):
                module.prepare(base / "bad-reference", manifest)
            (base / "backport.patch").write_text(backport + "tampered")
            with self.assertRaisesRegex(RuntimeError, "Patch checksum"):
                module.prepare(base / "bad-patch", manifest)
            self.assertFalse((base / "bad-patch/framework").exists())


if __name__ == "__main__":
    unittest.main()
