"""Check that source verification rejects incomplete or modified checkouts."""

import hashlib
import json
import subprocess
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "framework_sources", Path(__file__).resolve().parents[3] / "tools/prepare_framework_sources.py"
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


class ManifestIntegrityTest(unittest.TestCase):
    def test_registered_patch_checksums(self):
        root = Path(__file__).resolve().parents[3]
        manifest_path = root / "dependencies/framework-sources.json"
        manifest = json.loads(manifest_path.read_text())
        for name, source in manifest["sources"].items():
            for key in ("patch", "backport_patch"):
                if key not in source:
                    continue
                with self.subTest(framework=name, patch=key):
                    patch = (manifest_path.parent / source[key]).resolve()
                    self.assertTrue(patch.is_relative_to(root.resolve() / "patches"))
                    self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), source[key + "_sha256"])


class PatchSequenceTest(unittest.TestCase):
    def test_local_commit_reference_and_backport_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            upstream = base / "upstream"
            upstream.mkdir()
            (base / "patches").mkdir()
            (base / "dependencies").mkdir()

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
            (base / "patches/local.patch").write_text(local_patch)
            git("add", ".")
            (upstream / "model.py").write_text("v4 and v41\n")
            backport = git("diff", "--binary")
            (base / "patches/backport.patch").write_text(backport)

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
                "patch": "../patches/local.patch",
                "patch_sha256": hashlib.sha256(local_patch.encode()).hexdigest(),
                "backport_patch": "../patches/backport.patch",
                "backport_patch_sha256": hashlib.sha256(backport.encode()).hexdigest(),
                "reference_files": inventory("v4\n"),
                "files": inventory("v4 and v41\n"),
            }
            manifest = base / "dependencies/framework-sources.json"
            manifest.write_text(json.dumps({"sources": {"framework": source}}))
            module.prepare(base / "output", manifest)
            self.assertEqual((base / "output/framework/model.py").read_text(), "v4 and v41\n")
            # A wrong reference receipt must fail even if the final inventory is correct.
            source["reference_files"] = inventory("wrong\n")
            manifest.write_text(json.dumps({"sources": {"framework": source}}))
            with self.assertRaisesRegex(RuntimeError, "checksum or mode"):
                module.prepare(base / "bad-reference", manifest)
            (base / "patches/backport.patch").write_text(backport + "tampered")
            with self.assertRaisesRegex(RuntimeError, "Patch checksum"):
                module.prepare(base / "bad-patch", manifest)
            self.assertFalse((base / "bad-patch/framework").exists())


class FetchRetryTest(unittest.TestCase):
    def test_transient_failure_retries_exact_commit(self):
        from unittest.mock import patch

        failure = subprocess.CalledProcessError(128, ["git", "fetch"])
        with patch.object(module.subprocess, "check_call", side_effect=[failure, None]) as call:
            with patch.object(module.time, "sleep") as sleep:
                module.fetch_commit(Path("/tmp/source"), "abc123")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[0], call.call_args_list[1])
        self.assertIn("http.version=HTTP/1.1", call.call_args.args[0])
        self.assertEqual(call.call_args.args[0][-1], "abc123")
        sleep.assert_called_once_with(5)

    def test_exhausted_retries_raise(self):
        from unittest.mock import patch

        failure = subprocess.CalledProcessError(128, ["git", "fetch"])
        with patch.object(module.subprocess, "check_call", side_effect=failure) as call:
            with patch.object(module.time, "sleep") as sleep:
                with self.assertRaises(subprocess.CalledProcessError):
                    module.fetch_commit(Path("/tmp/source"), "abc123")
        self.assertEqual(call.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 10])


if __name__ == "__main__":
    unittest.main()
