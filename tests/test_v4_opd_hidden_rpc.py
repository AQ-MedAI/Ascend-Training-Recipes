"""CPU regression for hidden-state transport through untyped vLLM utility RPC."""

import __future__
import ast
import asyncio
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, AsyncMock

import torch

try:
    import msgspec
except ImportError:
    msgspec = None

from _v4_opd_test_sources import source_root

ROOT = source_root()
RELATIVE = "verl/workers/rollout/vllm_rollout/full_vocab_hidden_export.py"


class HiddenRpcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / RELATIVE
        nodes = [node for node in ast.parse(source.read_text()).body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))]
        cls.ns = dict(torch=torch, logging=logging, MethodType=MethodType, vLLMColocateWorkerExtension=object)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec',
                     flags=__future__.annotations.compiler_flag), cls.ns)

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

    def test_fetch_checks_nonfirst_dp_engine(self):
        source = ROOT / 'verl/workers/rollout/vllm_rollout/vllm_async_server.py'
        tree = ast.parse(source.read_text())
        node = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == '_export_full_vocab_hidden')
        export = Mock(side_effect=lambda **kwargs: kwargs)
        namespace = dict(asyncio=asyncio, unwrap_captured_hidden=self.unwrap, export_hidden_to_tq=export)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        hidden = torch.ones(2, 3, dtype=torch.bfloat16)
        payload = self.payload(hidden)
        core = SimpleNamespace(core_engines=['dp0', 'dp1'],
                               _call_utility_async=AsyncMock(side_effect=[[None], [payload]]))
        engine = SimpleNamespace(engine_core=core, collective_rpc=AsyncMock(return_value=[payload]))
        server = SimpleNamespace(engine=engine, _full_vocab_export_config={'prefix': 'test'})
        metadata = dict(teacher_name='healthbench', step=1, uid='sample', row_start=4, row_end=6)
        result = asyncio.run(namespace['_export_full_vocab_hidden'](server, 'request', metadata, 7))
        torch.testing.assert_close(result['hidden'], hidden)
        self.assertEqual(core._call_utility_async.await_count, 2)
        self.assertEqual([call.kwargs['engine'] for call in core._call_utility_async.await_args_list],
                         ['dp0', 'dp1'])
        engine.collective_rpc.assert_not_awaited()
        # The single-engine path remains compatible with public collective_rpc.
        core.core_engines = ['dp0']
        result = asyncio.run(namespace['_export_full_vocab_hidden'](server, 'request', metadata, 7))
        engine.collective_rpc.assert_awaited_once_with(method='fetch_captured_hidden')
        torch.testing.assert_close(result['hidden'], hidden)

    def test_status_distinguishes_missing_hook_from_non_capture_rank(self):
        runner = SimpleNamespace(_get_prompt_logprobs_dict=Mock())
        extension = self.ns['FullVocabHiddenWorkerExtension']()
        extension.model_runner = runner
        status = extension.fetch_captured_hidden()
        self.assertFalse(status['hook_installed'])
        self.ns['install_hidden_capture'](runner, capture_hidden=False)
        runner._get_prompt_logprobs_dict(torch.ones(5, 3))
        status = extension.fetch_captured_hidden()
        self.assertTrue(status['hook_installed'])
        self.assertFalse(status['capture_enabled'])
        self.assertEqual(status['hook_calls'], 1)
        # A new request must reset diagnostic counters as well as stale capture.
        self.ns['install_hidden_capture'](runner)
        self.assertEqual(extension.fetch_captured_hidden()['hook_calls'], 0)

    def test_worker_capture_fetch_is_consumed_once_and_skips_logits(self):
        original = Mock(side_effect=AssertionError("Prompt logits should not be materialized"))
        runner = SimpleNamespace(_get_prompt_logprobs_dict=original)
        extension = self.ns["FullVocabHiddenWorkerExtension"]()
        extension.model_runner = runner
        self.ns["install_hidden_capture"](runner, row_start=1, row_end=4)
        hidden = torch.arange(15).reshape(5, 3).to(torch.bfloat16)
        self.assertEqual(runner._get_prompt_logprobs_dict(hidden), {})
        result = self.unwrap([extension.fetch_captured_hidden()])
        torch.testing.assert_close(result, hidden[1:4], rtol=0, atol=0)
        status = extension.fetch_captured_hidden()
        self.assertEqual(status['format'], 'verl.full_vocab_capture_status.v1')
        self.assertTrue(status['capture_enabled'])
        self.assertEqual(status['hook_calls'], 1)
        self.assertEqual(status['last_hidden_shape'], [5, 3])
        original.assert_not_called()

    def test_chunked_capture_matches_complete_rows_and_handles_recompute(self):
        hidden = torch.arange(60).reshape(20, 3).to(torch.bfloat16)
        for row_range in ((None, None), (7, 18), (9, 16), (18, 19)):
            runner = SimpleNamespace(
                _get_prompt_logprobs_dict=Mock(),
                requests={'req': SimpleNamespace(prompt_token_ids=list(range(20)), num_computed_tokens=0)},
                input_batch=SimpleNamespace(req_id_to_index={'req': 0}),
                query_start_loc=SimpleNamespace(np=[0]),
            )
            extension = self.ns['FullVocabHiddenWorkerExtension']()
            extension.model_runner = runner
            self.ns['install_hidden_capture'](runner, request_id='req',
                                              row_start=row_range[0], row_end=row_range[1])
            # Repeated offsets simulate preemption/recomputation, not extra tokens.
            for start, end in ((0, 4), (4, 9), (0, 4), (4, 9), (9, 16), (16, 20)):
                runner.requests['req'].num_computed_tokens = start
                chunk = hidden[start:end].clone()
                runner._get_prompt_logprobs_dict(chunk, {'req': end - start})
                chunk.zero_()  # Runner reuses its device buffer after the hook.
            actual = self.unwrap(extension.fetch_captured_hidden())
            lo, hi = row_range
            torch.testing.assert_close(actual, hidden[0 if lo is None else lo:19 if hi is None else hi])
            self.assertIsNone(self.unwrap(extension.fetch_captured_hidden()))

    def test_chunk_capture_rejects_gaps_and_foreign_requests(self):
        runner = SimpleNamespace(
            _get_prompt_logprobs_dict=Mock(),
            requests={'req': SimpleNamespace(prompt_token_ids=list(range(10)), num_computed_tokens=5)},
            input_batch=SimpleNamespace(req_id_to_index={'req': 0}),
            query_start_loc=SimpleNamespace(np=[0]),
        )
        self.ns['install_hidden_capture'](runner, request_id='req')
        runner._get_prompt_logprobs_dict(torch.ones(5, 3), {'req': 5})
        extension = self.ns['FullVocabHiddenWorkerExtension']()
        extension.model_runner = runner
        with self.assertRaisesRegex(RuntimeError, 'missing 5 hidden rows'):
            extension.fetch_captured_hidden()
        with self.assertRaisesRegex(RuntimeError, 'Unexpected full-vocab request'):
            runner._get_prompt_logprobs_dict(torch.ones(5, 3), {'other': 5})

    def test_internal_request_id_and_hidden_offset(self):
        runner = SimpleNamespace(
            _get_prompt_logprobs_dict=Mock(),
            requests={'req-ab12cd34': SimpleNamespace(prompt_token_ids=[1, 2, 3], num_computed_tokens=0)},
            input_batch=SimpleNamespace(req_id_to_index={'req-ab12cd34': 0}),
            query_start_loc=SimpleNamespace(np=[2]),
        )
        self.ns['install_hidden_capture'](runner, request_id='req')
        hidden = torch.arange(15).reshape(5, 3).float()
        runner._get_prompt_logprobs_dict(hidden, {'req-ab12cd34': 3})
        extension = self.ns['FullVocabHiddenWorkerExtension']()
        extension.model_runner = runner
        torch.testing.assert_close(self.unwrap(extension.fetch_captured_hidden()), hidden[2:4])
        with self.assertRaisesRegex(RuntimeError, 'Unexpected full-vocab request'):
            runner._get_prompt_logprobs_dict(hidden, {'req-ffffffff': 3})

    def test_full_vocab_config_allows_chunked_prefill(self):
        source = ROOT / 'verl/workers/config/distillation.py'
        node = next(node for node in ast.walk(ast.parse(source.read_text()))
                    if isinstance(node, ast.FunctionDef) and node.name == '_validate_full_vocab_inference')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        config = SimpleNamespace(inference=SimpleNamespace(
            max_num_seqs=1, enable_chunked_prefill=True,
            max_num_batched_tokens=4096, max_model_len=65536))
        namespace[node.name](config)
        config.inference.max_num_seqs = 2
        with self.assertRaises(ValueError):
            namespace[node.name](config)


if __name__ == "__main__":
    unittest.main()
