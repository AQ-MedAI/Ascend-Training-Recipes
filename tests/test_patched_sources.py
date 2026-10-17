"""Offline checks for managed upstream-submodule patches."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from patched_sources import ensure_source, is_prepared, load_source


class PatchedSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.git("init", "-q")
        (self.repo / ".gitignore").write_text("*.egg-info/\n")
        (self.repo / "model.py").write_text("base\n")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
        commit = self.git("rev-parse", "HEAD").strip()
        reference = self.inventory()
        (self.repo / "model.py").write_text("adapted\n")
        self.patch = self.root / "adaptation.patch"
        self.patch.write_text(self.git("diff", "--binary"))
        self.source = {
            "repository": "https://example.invalid/upstream.git",
            "commit": commit,
            "patch": "adaptation.patch",
            "patch_sha256": hashlib.sha256(self.patch.read_bytes()).hexdigest(),
            "reference_files": reference,
            "files": self.inventory(),
        }
        self.git("restore", "model.py")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def inventory(self):
        return {
            p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "executable": False}
            for p in self.repo.iterdir()
            if p.is_file()
        }

    def test_apply_repeat_and_ignored_build_artifact(self):
        ensure_source(self.repo, self.source, self.patch)
        self.assertTrue(is_prepared(self.repo, self.source))
        (self.repo / "package.egg-info").mkdir()
        (self.repo / "package.egg-info/PKG-INFO").write_text("generated")
        ensure_source(self.repo, self.source, self.patch)
        self.assertEqual((self.repo / "model.py").read_text(), "adapted\n")
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.source["commit"])

    def test_user_edit_is_preserved(self):
        ensure_source(self.repo, self.source, self.patch)
        (self.repo / "model.py").write_text("user changes\n")
        with self.assertRaisesRegex(RuntimeError, "Local changes"):
            ensure_source(self.repo, self.source, self.patch)
        self.assertEqual((self.repo / "model.py").read_text(), "user changes\n")

    def test_staged_changes_are_preserved(self):
        ensure_source(self.repo, self.source, self.patch)
        self.git("add", "model.py")
        with self.assertRaisesRegex(RuntimeError, "Staged changes"):
            ensure_source(self.repo, self.source, self.patch)
        self.assertEqual(self.git("show", ":model.py"), "adapted\n")

    def test_hash_checked_before_mutation(self):
        (self.root / "manifest.json").write_text(json.dumps({"sources": {"mbridge": self.source}}))
        component = {
            "source_manifest": "manifest.json",
            "repository": self.source["repository"],
            "profiles": {"v41": {"commit": self.source["commit"]}},
        }
        load_source(self.root, "mbridge", component, "v41")
        self.patch.write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "Patch checksum"):
            load_source(self.root, "mbridge", component, "v41")
        self.assertEqual((self.repo / "model.py").read_text(), "base\n")


if __name__ == "__main__":
    unittest.main()
