"""Exercise the patched Core writer with native CPU DCP serialization and no fork."""

import ast
import importlib.util
import io
import inspect
import logging
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.distributed.checkpoint.filesystem import _StorageWriterTransforms, _write_item
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "megatron/core/dist_checkpointing/strategies/filesystem_async.py"


class InprocessCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT.parent / "Megatron-LM-aq" / RELATIVE
        if not source.is_file():
            raise unittest.SkipTest("Provide pinned K26 Core")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / RELATIVE
            path.parent.mkdir(parents=True)
            path.write_bytes(source.read_bytes())
            subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "apply",
                    str(ROOT / "patches/megatron-lm/k26/ascend-checkpoint-inprocess.patch"),
                ],
                check=True,
            )
            tree = ast.parse(path.read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FileSystemWriterAsync")
        methods = [
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name in {"write_preloaded_data_multiproc", "write_preloaded_data"}
        ]
        for method in methods:
            method.decorator_list = [ast.Name(id="staticmethod", ctx=ast.Load())]
        cls_tree = ast.ClassDef(name="FileSystemWriterAsync", bases=[], keywords=[], body=methods, decorator_list=[])
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls_tree],
            type_ignores=[],
        )

        def no_fork(*args, **kwargs):
            raise AssertionError("fork/process creation attempted")

        cls.ns = dict(
            torch=torch,
            os=os,
            queue=queue,
            inspect=inspect,
            logging=logging,
            time=time.time,
            _write_item=_write_item,
            _process_memory=lambda: 0,
            mp=SimpleNamespace(get_context=no_fork),
            __name__=__name__,
        )
        exec(compile(ast.fix_missing_locations(module), "<patched native Core writer>", "exec"), cls.ns)
        cls.writer = cls.ns["FileSystemWriterAsync"]

    def bucket(self, path, key, tensor):
        item = WriteItem(
            index=MetadataIndex(key),
            type=WriteItemType.TENSOR,
            tensor_data=TensorWriteData(
                chunk=ChunkStorageMetadata(offsets=torch.Size([0]), sizes=tensor.size()),
                properties=TensorProperties.create_from_tensor(tensor),
                size=tensor.size(),
            ),
        )
        return str(path), path.name, ([], [(item, tensor)])

    def test_cpu_tensor_roundtrip_multiple_buckets_without_fork(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ASCEND_CHECKPOINT_WRITER="inprocess"):
            root = Path(directory)
            values = [torch.arange(8, dtype=torch.float32), torch.arange(5, dtype=torch.bfloat16)]
            buckets = [self.bucket(root / f"{i}.distcp", f"tensor{i}", value) for i, value in enumerate(values)]
            results = queue.Queue()
            self.writer.write_preloaded_data_multiproc([_StorageWriterTransforms()], False, 0, buckets, results)
            written = results.get_nowait()
            self.assertEqual(set(written), {0, 1})
            for i, value in enumerate(values):
                self.assertEqual(len(written[i]), 1)
                torch.testing.assert_close(torch.load(root / f"{i}.distcp", weights_only=True), value)

    def test_write_error_is_returned_to_the_existing_result_protocol(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ASCEND_CHECKPOINT_WRITER="inprocess"):
            bucket = self.bucket(Path(directory) / "absent" / "data.distcp", "weight", torch.ones(2))
            results = queue.Queue()
            self.writer.write_preloaded_data_multiproc([_StorageWriterTransforms()], False, 0, [bucket], results)
            self.assertIsInstance(results.get_nowait(), FileNotFoundError)

    def test_legacy_mode_still_selects_the_multiprocessing_path(self):
        with patch.dict(os.environ, ASCEND_CHECKPOINT_WRITER="multiprocess"):
            with self.assertRaisesRegex(AssertionError, "fork/process creation attempted"):
                self.writer.write_preloaded_data_multiproc([], False, 0, [], queue.Queue())
