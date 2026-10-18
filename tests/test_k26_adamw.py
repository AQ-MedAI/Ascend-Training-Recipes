"""Verify Core optimizer creation, AdamW updates, and state restoration on CPU."""
from __future__ import annotations
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, 'CPU PyTorch is required')
class AdamwCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        relative = 'mindspeed/core/optimizer/adamw.py'
        source = ROOT / 'third_party/MindSpeed' / relative
        if not source.is_file():
            raise unittest.SkipTest('Prepare MindSpeed sources first')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / relative
            target.parent.mkdir(parents=True)
            target.write_text(source.read_text())
            subprocess.run(['git', '-C', directory, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', directory, 'apply', str(ROOT / 'patches/mindspeed/k26/ascend-adamw-api.patch')], check=True)
            tree = ast.parse(target.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        ns = dict(torch=torch, Tensor=torch.Tensor, Optimizer=torch.optim.Optimizer,
                  TorchAdamW=torch.optim.AdamW, _to_step_int=lambda step: int(step.item()) if torch.is_tensor(step) else int(step))
        exec(compile('from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in nodes), '<actual patched MindSpeed AdamW>', 'exec'), ns)
        # Only NPU device placement for the cached step scalar is replaced;
        # the real AdamW.step and real torch._fused_adamw_ arithmetic execute.
        def step_tensor(optimizer, group):
            cache = optimizer._step_tensor_cache
            if id(group) not in cache:
                cache[id(group)] = torch.empty((), dtype=torch.int64)
            cache[id(group)].fill_(group['step'])
            return cache[id(group)]
        ns['_get_step_tensor'] = step_tensor
        cls.ns = ns
        core = ROOT.parent / 'Megatron-LM-aq/megatron/core/optimizer/__init__.py'
        if not core.is_file():
            raise unittest.SkipTest('Provide K26 Megatron-LM-aq source')
        tree = ast.parse(core.read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_megatron_optimizer_based_on_param_groups')
        cls.adam_branch = next(n for n in ast.walk(factory) if isinstance(n, ast.If) and ast.unparse(n.test) == "config.optimizer == 'adam'" and any(isinstance(x, ast.Name) and x.id == 'adam_cls' for x in ast.walk(n)))

    def build_from_core(self, parameter):
        ns = dict(torch=torch, Adam=self.ns['AdamW'], USING_PYTORCH_OPTIMIZER=False,
            param_groups=[parameter], config=SimpleNamespace(lr=.003, weight_decay=.1,
                adam_beta1=.9, adam_beta2=.95, adam_eps=1e-8, decoupled_weight_decay=True,
                use_precision_aware_optimizer=False), is_te_min_version=lambda version: False)
        exec(compile(ast.Module(body=self.adam_branch.body, type_ignores=[]), '<actual Core Adam factory branch>', 'exec'), ns)
        return ns['optimizer'], ns['init_state_fn']

    def test_core_factory_kwarg_and_preinitialized_moment_state(self):
        p = torch.nn.Parameter(torch.tensor([1., -2., .5]))
        opt, initialize = self.build_from_core(p)
        initialize(opt)
        self.assertEqual(set(opt.state[p]), {'exp_avg', 'exp_avg_sq'})
        p.grad = torch.tensor([.1, -.2, .3])
        opt.step()
        self.assertEqual(opt.param_groups[0]['step'], 1)
        self.assertFalse(torch.equal(p, torch.tensor([1., -2., .5])))

    def test_numerical_updates_and_resume_match_torch_adamw(self):
        p = torch.nn.Parameter(torch.tensor([1., -2., .5]))
        q = torch.nn.Parameter(p.detach().clone())
        opt, initialize = self.build_from_core(p)
        initialize(opt)
        reference = torch.optim.AdamW([q], lr=.003, weight_decay=.1, betas=(.9, .95), eps=1e-8, foreach=False)
        for step in range(5):
            gradient = torch.tensor([.1 + .01 * step, -.2, .3])
            p.grad = gradient.clone()
            q.grad = gradient.clone()
            opt.step()
            reference.step()
            torch.testing.assert_close(p, q)
            if step == 1:
                saved = deepcopy(opt.state_dict())
                self.assertEqual(saved['param_groups'][0]['step'], 2)
                restored = torch.nn.Parameter(p.detach().clone())
                opt, initialize = self.build_from_core(restored)
                opt.load_state_dict(saved)
                self.assertEqual(opt._step_tensor_cache, {})
                p = restored
        torch.testing.assert_close(opt.state[p]['exp_avg'], reference.state[q]['exp_avg'])
        torch.testing.assert_close(opt.state[p]['exp_avg_sq'], reference.state[q]['exp_avg_sq'])
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)

    def test_core_distributed_optimizer_extracts_group_step_metadata(self):
        path = ROOT.parent / 'Megatron-LM-aq/megatron/core/optimizer/distrib_optimizer.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'DistributedOptimizer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'state_dict')
        ns = dict(torch=torch, HAVE_APEX_OR_TE=True, USING_APEX_OPTIMIZER=True,
                  USING_TE_OPTIMIZER=False, HybridDeviceOptimizer=type('Hybrid', (), {}))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual Core distributed optimizer metadata>', 'exec'), ns)
        p = torch.nn.Parameter(torch.ones(2))
        optimizer, initialize = self.build_from_core(p)
        initialize(optimizer)
        for i in range(2):
            p.grad = torch.tensor([.1, .2])
            optimizer.step()
        metadata = ns['state_dict'](SimpleNamespace(optimizer=optimizer, grad_scaler=None))
        self.assertEqual(metadata['optimizer']['param_groups'][0]['step'], 2)
        self.assertNotIn('state', metadata['optimizer'])

    def test_mindspeed_chained_step_clips_before_adam_update(self):
        source = ROOT / 'third_party/MindSpeed/mindspeed/core/megatron_basic/count_zero_fix.py'
        function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        core_clip = ROOT.parent / 'Megatron-LM-aq/megatron/core/optimizer/clip_grads.py'
        core_function = next(n for n in ast.parse(core_clip.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'clip_grad_by_total_norm_fp32')
        core_keys = {a.arg for a in core_function.args.args + core_function.args.kwonlyargs}
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'clip_grad_by_total_norm_fp32':
                self.assertFalse({k.arg for k in node.keywords} - core_keys)
        p = torch.nn.Parameter(torch.ones(2))
        optimizer, initialize = self.build_from_core(p)
        initialize(optimizer)
        p.grad = torch.tensor([3., 4.])
        calls = []
        def clip(parameters, max_norm, total_norm, use_decoupled_grad):
            calls.append('clip')
            for parameter in parameters:
                parameter.grad.mul_(max_norm / (total_norm + 1e-6))
        def update():
            calls.append('step')
            optimizer.step()
            return True
        ns = dict(clip_grad_by_total_norm_fp32=clip)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual MindSpeed chained step>', 'exec'), ns)
        inner = SimpleNamespace(config=SimpleNamespace(clip_grad=1., use_precision_aware_optimizer=False,
            log_num_zeros_in_grad=False), get_parameters=lambda: [p])
        chain = SimpleNamespace(prepare_grads=lambda: False, get_grad_norm=lambda: 5.,
            chained_optimizers=[inner], step_with_ready_grads=update)
        self.assertEqual(ns['step'](chain), (True, 5., 0))
        self.assertEqual(calls, ['clip', 'step'])
        self.assertLessEqual(p.grad.norm().item(), 1.)

    def test_both_registered_implementations_accept_adamw_and_reject_coupled_mode(self):
        for name in ['AdamW', 'FusedTorchAdamW']:
            cls = self.ns[name]
            cls([torch.nn.Parameter(torch.ones(2))], adam_w_mode=True)
            with self.assertRaisesRegex(ValueError, 'decoupled'):
                cls([torch.nn.Parameter(torch.ones(2))], adam_w_mode=False)
            with self.assertRaises(TypeError):
                cls([torch.nn.Parameter(torch.ones(2))], unimplemented_option=True)


if __name__ == '__main__':
    unittest.main()
