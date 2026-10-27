"""CPU regression for different teacher/student prefixes in both KL directions."""
import __future__
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
# Import before patch.dict(sys.modules) so PyTorch registrations are not re-imported.
import torch.distributed.tensor  # noqa: F401

from _v4_opd_test_sources import source_root

ROOT = source_root()
SOURCE = ROOT / 'verl/trainer/distillation/megatron/full_vocab_kl.py'


class RubricKLAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = {'_student_artifact_row_range', 'compute_full_vocab_kl',
                 'compute_forward_kl_full_vocab', 'compute_reverse_kl_full_vocab'}
        nodes = [node for node in ast.parse(SOURCE.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.ns = {'torch': torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec',
                     flags=__future__.annotations.compiler_flag), cls.ns)

    def test_reported_lengths_and_legacy_validation(self):
        resolve = self.ns['_student_artifact_row_range']
        artifact = dict(sequence_length=2735, row_start=686, row_end=2734,
                        student_sequence_length=2625, student_row_start=576, student_row_end=2624)
        self.assertEqual(resolve(artifact, 2625, 0), (576, 2624))
        self.assertEqual(resolve(dict(sequence_length=5, row_start=2, row_end=4), 5, 0), (2, 4))
        self.assertEqual(resolve({}, 5, 0), (0, 4))
        for bad in [
            dict(sequence_length=2735, row_start=686, row_end=2734),
            {**artifact, 'student_row_end': 2623},
            {**artifact, 'student_row_start': 577},
            {**artifact, 'student_sequence_length': 2626},
            {**artifact, 'row_start': -1},
            {**artifact, 'row_end': 2733},
            {'student_row_start': 576},
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve(bad, 2625, 0)

    def test_both_kl_directions_match_response_losses_and_gradients(self):
        torch.manual_seed(7)
        # Sample 0: teacher prefix is longer; sample 1: teacher prefix is shorter.
        # Mix two teachers to exercise grouping as well as per-sample offsets.
        artifacts = [
            dict(teacher='t0', sequence_length=7, row_start=4, row_end=6,
                 student_sequence_length=5, student_row_start=2, student_row_end=4),
            dict(teacher='t1', sequence_length=3, row_start=1, row_end=2,
                 student_sequence_length=4, student_row_start=2, student_row_end=3),
        ]
        hiddens = [torch.randn(2, 3), torch.randn(1, 3)]
        heads = {'t0': torch.randn(4, 3), 't1': torch.randn(4, 3)}
        tq = ModuleType('verl.trainer.distillation.full_vocab_tq')
        tq.init_full_vocab_tq = lambda: None
        tq.multi_get_hidden = lambda **kw: hiddens
        data = {
            'input_ids': SimpleNamespace(is_nested=True, offsets=lambda: torch.tensor([0, 5, 9])),
            'teacher_full_vocab_artifact': artifacts,
        }
        def pack(nested, **kwargs):
            return (torch.cat(list(nested.unbind())).unsqueeze(0),)

        class KL:
            @staticmethod
            def apply(logits, hidden, head, chunk_tokens, reverse):
                log_p = logits.log_softmax(-1)
                log_q = (hidden @ head.t()).log_softmax(-1)
                return ((log_p.exp() * (log_p - log_q)).sum(-1) if reverse
                        else (log_q.exp() * (log_q - log_p)).sum(-1))

        namespace = self.ns
        namespace.update(
            TEACHER_HIDDEN_ARTIFACT_FIELD='teacher_full_vocab_artifact',
            tu=SimpleNamespace(get=lambda d, k, default: d.get(k, default),
                               unwrap_non_tensor_data=lambda x: x,
                               get_non_tensor_data=lambda **kw: None),
            _per_sample_temperature=lambda d, b, device: torch.ones(b, device=device),
            preprocess_thd_engine=pack, preprocess_bshd_engine=pack,
            _VocabParallelFullVocabKL=KL,
        )
        config = SimpleNamespace(distillation_loss=SimpleNamespace(full_vocab_chunk_tokens=2))
        for reverse in [False, True]:
            for fmt in ['thd', 'bshd']:
                with self.subTest(reverse=reverse, fmt=fmt), patch.dict(sys.modules, {tq.__name__: tq}):
                    logits = torch.randn(1, 9, 4, requires_grad=True)
                    fn = namespace['compute_reverse_kl_full_vocab' if reverse else 'compute_forward_kl_full_vocab']
                    out = fn(student_logits=logits, data=data, config=config,
                             data_format=fmt, teacher_lm_head_shards=heads)
                    expected = logits.new_zeros(1, 9)
                    expected[:, 2:4] = KL.apply(logits[:, 2:4], hiddens[0], heads['t0'], 2, reverse)
                    expected[:, 7:8] = KL.apply(logits[:, 7:8], hiddens[1], heads['t1'], 2, reverse)
                    torch.testing.assert_close(out['distillation_losses'], expected)
                    actual_grad = torch.autograd.grad(out['distillation_losses'].sum(), logits, retain_graph=True)[0]
                    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
                    torch.testing.assert_close(actual_grad, expected_grad)
                    self.assertEqual(torch.count_nonzero(actual_grad[:, [0, 1, 4, 5, 6, 8]]).item(), 0)
                    norm = torch.cat([hiddens[0].norm(dim=-1), hiddens[1].norm(dim=-1)])
                    torch.testing.assert_close(out['teacher_hidden_norm'][0, [2, 3, 7]], norm)


if __name__ == '__main__':
    unittest.main()
