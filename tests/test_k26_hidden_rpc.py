"""CPU regression for hidden-state transport through untyped vLLM utility RPC."""

import __future__
import ast
import logging
from pathlib import Path
import subprocess
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

try:
    import msgspec
except ImportError:
    msgspec = None

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "verl/workers/rollout/vllm_rollout/full_vocab_hidden_export.py"


class HiddenRpcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            for patch in ("k26.patch", "k26-full-vocab-rpc.patch"):
                subprocess.run(
                    ["git", "-C", directory, "apply", f"--include={RELATIVE}", str(ROOT / "patches/verl" / patch)],
                    check=True,
                )
            nodes = [
                node
                for node in ast.parse((root / RELATIVE).read_text()).body
                if not isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            cls.ns = dict(torch=torch, logging=logging, MethodType=MethodType, vLLMColocateWorkerExtension=object)
            exec(
                compile(
                    ast.Module(body=nodes, type_ignores=[]),
                    RELATIVE,
                    "exec",
                    flags=__future__.annotations.compiler_flag,
                ),
                cls.ns,
            )

    def payload(self, tensor):
        return self.ns["_encode_hidden_payload"](tensor)

    def unwrap(self, value):
        return self.ns["unwrap_captured_hidden"](value)

    def test_exact_dtype_values_and_noncontiguous_input(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                tensor = torch.arange(35).reshape(5, 7).to(dtype).t()
                actual = self.unwrap([None, [self.payload(tensor), None]])
                torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device.type, "cpu")
                actual[0, 0] = -1
                self.assertEqual(tensor[0, 0], 0)

    @unittest.skipIf(msgspec is None, "msgspec is needed for the wire round-trip")
    def test_native_messagepack_roundtrip_without_custom_type_decoding(self):
        for width in (2, 64):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(width=width, dtype=dtype):
                    tensor = torch.arange(5 * width).reshape(5, width).to(dtype)
                    # No custom decoder or type metadata is needed at either RPC hop.
                    wire = msgspec.msgpack.encode([None, [self.payload(tensor)]])
                    actual = self.unwrap(msgspec.msgpack.decode(wire))
                    torch.testing.assert_close(actual, tensor, rtol=0, atol=0)

    def test_invalid_payload_is_rejected(self):
        payload = self.payload(torch.ones(2, 3, dtype=torch.bfloat16))
        for change in (
            {"dtype": "uint8"},
            {"shape": [2, -3]},
            {"shape": [True, 6]},
            {"shape": [6]},
            {"data": payload["data"][:-1]},
            {"data": "not bytes"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.unwrap({**payload, **change})

    def test_existing_tensor_and_missing_results(self):
        tensor = torch.ones(2, 3)
        self.assertIs(self.unwrap([None, (tensor,)]), tensor)
        self.assertIsNone(self.unwrap([None, [None]]))
        with self.assertRaises(ValueError):
            self.payload(torch.empty(0, 3))

    def test_worker_capture_fetch_is_consumed_once_and_skips_logits(self):
        original = Mock(side_effect=AssertionError("Prompt logits should not be materialized"))
        runner = SimpleNamespace(_get_prompt_logprobs_dict=original)
        extension = self.ns["FullVocabHiddenWorkerExtension"]()
        extension.model_runner = runner
        extension.start_hidden_capture()
        hidden = torch.arange(15).reshape(5, 3).to(torch.bfloat16)
        self.assertEqual(runner._get_prompt_logprobs_dict(hidden), {})
        result = self.unwrap([extension.fetch_captured_hidden()])
        torch.testing.assert_close(result, hidden, rtol=0, atol=0)
        self.assertIsNone(extension.fetch_captured_hidden())
        original.assert_not_called()


if __name__ == "__main__":
    unittest.main()
