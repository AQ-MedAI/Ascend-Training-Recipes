"""Apply the opt-in trace patch and verify that it preserves serialization."""

from contextlib import redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


class CheckpointTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference = ROOT.parent / "verl_hpu_mopd_patch/code/verl/verl/utils/checkpoint/megatron_checkpoint_manager.py"
        if not reference.is_file():
            raise unittest.SkipTest("Provide reference K26 checkpoint manager")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = root / "verl/utils/checkpoint/megatron_checkpoint_manager.py"
            manager.parent.mkdir(parents=True)
            manager.write_bytes(reference.read_bytes())
            subprocess.run(
                ["git", "-C", directory, "apply", str(ROOT / "patches/verl/k26-checkpoint-trace.patch")], check=True
            )
            spec = importlib.util.spec_from_file_location("k26_trace_test", manager.parent / "serialization_trace.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_trace_preserves_payload_and_restores_save(self):
        original = torch.save
        manager = SimpleNamespace(
            checkpoint_config=SimpleNamespace(async_save=False), rank=0, use_dist_checkpointing=True
        )

        @self.module.trace_checkpoint_save
        def save(self):
            buffer = io.BytesIO()
            torch.save({"weight": torch.arange(8)}, buffer)
            buffer.seek(0)
            return torch.load(buffer, weights_only=True)

        output = io.StringIO()
        with patch.dict(os.environ, K26_CHECKPOINT_TRACE="1"), redirect_stderr(output):
            result = save(manager)
        torch.testing.assert_close(result["weight"], torch.arange(8))
        self.assertIs(torch.save, original)
        self.assertIn("device=cpu", output.getvalue())
        self.assertIn("torch.save completed", output.getvalue())

    def test_original_save_is_restored_after_error(self):
        original = torch.save

        @self.module.trace_checkpoint_save
        def fail(self):
            raise RuntimeError("save failed")

        manager = SimpleNamespace(checkpoint_config=SimpleNamespace(async_save=False), rank=0)
        with patch.dict(os.environ, K26_CHECKPOINT_TRACE="1"), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                fail(manager)
        self.assertIs(torch.save, original)
