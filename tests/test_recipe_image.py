"""Offline checks for recipe snapshots and container entrypoint activation."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("record_recipe", ROOT / "tools/record_recipe.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecipeImageTest(unittest.TestCase):
    def test_snapshot_tracks_content_and_mode_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            recipe = base / "recipe"
            for relative in ("tools/activate.sh", "recipes/deepseek-v4/dsv4_flash_rl.sh",
                             "recipes/deepseek-v4.1/dsv41_flash_rl.sh", "workspace/rollout/dsv41/runtime/worker.py"):
                path = recipe / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original\n")
                path.chmod(0o644)
            receipt = base / "manifest/recipe.json"
            original = module.record_recipe(recipe, receipt, "abc123")
            self.assertEqual(json.loads(receipt.read_text())["revision"], "abc123")
            self.assertEqual(original["snapshot_sha256"], module.record_recipe(recipe, receipt)["snapshot_sha256"])
            script = recipe / "tools/activate.sh"
            script.chmod(0o755)
            self.assertNotEqual(original["snapshot_sha256"], module.record_recipe(recipe, receipt)["snapshot_sha256"])
            script.chmod(0o644)
            script.write_text("modified\n")
            self.assertNotEqual(original["snapshot_sha256"], module.record_recipe(recipe, receipt)["snapshot_sha256"])
            (recipe / "third_party").mkdir()
            with self.assertRaisesRegex(ValueError, "development checkout"):
                module.record_recipe(recipe, receipt)

    def test_entrypoint_activates_bundled_or_overridden_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            # CANN setup is machine-specific; exercise the real remainder of the entrypoint.
            entrypoint = (ROOT / "docker/v41/entrypoint.sh").read_text()
            entrypoint = "\n".join(line for line in entrypoint.splitlines()
                                   if not line.startswith("source /usr/local/Ascend/")) + "\n"
            script = base / "entrypoint.sh"
            script.write_text(entrypoint)
            recipe = base / "mounted recipe"
            (recipe / "tools").mkdir(parents=True)
            (recipe / "tools/activate.sh").write_text('export ACTIVATED_PROFILE="$1"\nexport RECIPE_ACTIVATED=yes\n')
            environment = os.environ.copy()
            environment.update(RECIPE_ROOT=str(recipe), RECIPE_PROFILE="v4")
            command = ["bash", str(script), sys.executable, "-c",
                       'import os,sys; print(os.environ["RECIPE_ACTIVATED"], os.environ["ACTIVATED_PROFILE"]); sys.exit(7)']
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(result.stdout.strip(), "yes v4")
            environment["RECIPE_ROOT"] = str(base / "missing")
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Missing recipe activation script", result.stderr)


if __name__ == "__main__":
    unittest.main()
