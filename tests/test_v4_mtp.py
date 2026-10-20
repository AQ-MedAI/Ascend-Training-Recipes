"""CPU contracts for official V4 MTP names, TP projection, streams and CP shifts."""

import ast
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest
import torch
import torch.nn.functional as F

import os

ROOT = (
    Path(os.environ["V4_TEST_ROOT"])
    if "V4_TEST_ROOT" in os.environ
    else Path(__file__).resolve().parents[1] / "third_party"
)
BRIDGE = ROOT / "mbridge/mbridge/models/deepseek_v3.py"
MTP = ROOT / "MindSpeed-LLM/mindspeed_llm/core/transformer/multi_token_prediction.py"


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def make_bridge(kind="deepseek_v4", layers=43):
    tree = ast.parse(BRIDGE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    wanted = {"_DIRECT_MAPPING", "_MLP_MAPPING", "_ATTENTION_MAPPING", "_SHARED_STATE_DICT_MAPPING"}
    methods = {
        "_convert_mtp_param",
        "_weight_name_mapping_mlp",
        "_weight_name_mapping_attention",
        "_weight_name_mapping_mcore_to_hf",
        "_weight_to_hf_format",
        "_weight_to_mcore_format",
    }
    cls.decorator_list = []
    cls.body = [
        n
        for n in cls.body
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name in methods)
    ]

    class Base:
        def _weight_to_hf_format(self, name, tensor):
            return self._weight_name_mapping_mcore_to_hf(name), [tensor]

    base_tree = ast.parse((ROOT / "mbridge/mbridge/core/bridge.py").read_text())
    base_cls = next(n for n in base_tree.body if isinstance(n, ast.ClassDef))
    method = next(
        n for n in base_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_weight_name_mapping_attention"
    )
    method_ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "attention-map", "exec"), method_ns)
    Base._weight_name_mapping_attention = method_ns["_weight_name_mapping_attention"]
    ns = {"torch": torch, "re": re, "LLMBridge": Base}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(BRIDGE), "exec"), ns)
    bridge = ns[cls.name]()
    bridge.hf_config = NS(model_type=kind)
    bridge.config = NS(num_layers=layers, mtp_num_layers=1)
    return bridge


class MappingTests(unittest.TestCase):
    def test_v4_mtp_names(self):
        b = make_bridge()
        cases = {
            "layers.0.enorm.weight": ["enorm.weight"],
            "layers.0.hnorm.weight": ["hnorm.weight"],
            "layers.0.eh_proj.weight": ["e_proj.weight", "h_proj.weight"],
            "final_layernorms.0.weight": ["norm.weight"],
            "layers.0.hc_head.hc_base": ["hc_head_base"],
            "layers.0.hc_head.hc_fn.weight": ["hc_head_fn"],
            "layers.0.hc_head.hc_scale": ["hc_head_scale"],
            "layers.0.transformer_layer.attn_mhc.hc_base": ["hc_attn_base"],
            "layers.0.transformer_layer.mlp_mhc.hc_fn.weight": ["hc_ffn_fn"],
            "layers.0.transformer_layer.self_attention.linear_q.weight": ["attn.wq_a.weight"],
            "layers.0.transformer_layer.mlp.experts.local_experts.17.linear_fc1.weight": [
                "ffn.experts.17.w1.weight",
                "ffn.experts.17.w3.weight",
            ],
        }
        for source, names in cases.items():
            self.assertEqual(b._weight_name_mapping_mcore_to_hf("mtp." + source), ["mtp.0." + n for n in names])

    def test_shared_weights_keep_official_names_at_61_layers(self):
        b = make_bridge(layers=61)
        for name, expected in [
            ("embedding.word_embeddings.weight", "embed.weight"),
            ("output_layer.weight", "head.weight"),
        ]:
            self.assertEqual(b._weight_to_hf_format(name, torch.ones(4, 4))[0], [expected])

    def test_v3_legacy_mtp_namespace_is_preserved(self):
        b = make_bridge("deepseek_v3", 61)
        self.assertEqual(b._convert_mtp_param("mtp.layers.0.eh_proj.weight"), ["model.layers.61.eh_proj.weight"])

    def test_projection_roundtrip_and_tp_shards(self):
        b = make_bridge()
        e, h = torch.randn(16, 16), torch.randn(16, 16)
        name = "mtp.layers.0.eh_proj.weight"
        combined = b._weight_to_mcore_format(name, [e, h])
        for tp in (1, 2, 4, 8):
            rebuilt = torch.cat(list(combined.chunk(tp, dim=0)), dim=0)
            names, tensors = b._weight_to_hf_format(name, rebuilt)
            self.assertEqual(names, ["mtp.0.e_proj.weight", "mtp.0.h_proj.weight"])
            self.assertTrue(all(tensor.is_contiguous() for tensor in tensors))
            self.assertTrue(torch.equal(tensors[0], e))
            self.assertTrue(torch.equal(tensors[1], h))


class ConfigTests(unittest.TestCase):
    def configure(self, enabled=True, **overrides):
        import sys
        from types import ModuleType
        from unittest.mock import patch

        path = ROOT / "verl/verl/workers/engine/mindspeed/utils.py"
        captured = []
        namespace = {
            "MindSpeedEngineConfig": object,
            "add_mcore_arguments": lambda value: value,
            "set_global_config": captured.append,
        }
        function(path, "get_base_mcore_config_from_engine_config", namespace)
        apply = function(path, "apply_patch", namespace)
        hf = NS(model_type="deepseek_v4", num_nextn_predict_layers=1, compress_ratios=[0, 0, 4, 128, 0])
        model = NS(
            architectures=["DeepseekV4ForCausalLM"],
            hf_config=hf,
            mtp=NS(enable=enabled, enable_train=enabled, mtp_loss_scaling_factor=0.1),
        )
        engine = NS(
            tensor_model_parallel_size=4,
            expert_model_parallel_size=8,
            expert_tensor_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=2,
            sequence_parallel=True,
            use_distributed_optimizer=True,
            seed=42,
            strategy="mindspeed_llm",
            use_remove_padding=False,
            dynamic_context_parallel=False,
            llm_kwargs=dict(num_layers=4, compress_ratios=[0, 0, 4, 128], enable_mhc=True),
        )
        for key, value in overrides.items():
            if key == "llm_kwargs":
                engine.llm_kwargs.update(value)
            elif key == "count":
                hf.num_nextn_predict_layers = value
            elif key == "train":
                model.mtp.enable_train = value
            else:
                setattr(engine, key, value)
        adaptor = ModuleType("mindspeed_llm.tasks.megatron_adaptor_v2")
        adaptor.repatch = lambda value: None
        with patch.dict(sys.modules, {"mindspeed_llm.tasks.megatron_adaptor_v2": adaptor}):
            apply(model, engine, None)
        return captured[0]

    def test_engine_object_survives_conversion_and_mtp_geometry_is_extended(self):
        config = self.configure()
        self.assertEqual(config["mtp_num_layers"], 1)
        self.assertEqual(config["compress_ratios"], [0, 0, 4, 128, 0])
        self.assertEqual(config["mtp_loss_scaling_factor"], 0.1)
        self.assertEqual(self.configure(enabled=False)["mtp_num_layers"], 0)

    def test_incompatible_mtp_modes_fail_before_model_construction(self):
        for options in [
            dict(enabled=False, train=True),
            dict(count=0),
            dict(use_remove_padding=True),
            dict(dynamic_context_parallel=True),
            dict(llm_kwargs={"enable_mhc": False}),
            dict(llm_kwargs={"schedules_method": "dualpipev"}),
            dict(llm_kwargs={"moe_fb_overlap": True}),
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.configure(**options)


class RuntimeTests(unittest.TestCase):
    def test_projection_preserves_distinct_streams_and_gradients(self):
        class Projection(torch.nn.Linear):
            def forward(self, x):
                return super().forward(x), None

        class IdentityBlock(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states, None

        args = NS(position_embedding_type="deepseek4", enable_mhc=True, hc_mult=4)
        from contextlib import nullcontext

        ns = {
            "torch": torch,
            "Tensor": torch.Tensor,
            "InferenceParams": object,
            "PackedSeqParams": object,
            "get_args": lambda: args,
            "make_viewless_tensor": lambda inp, **kw: inp,
            "nullcontext": nullcontext,
            "all_gather_last_dim_from_tensor_parallel_region": lambda x: x,
        }
        forward = function(MTP, "mtp_layer_forward", ns)
        torch.manual_seed(17)
        e = torch.randn(3, 2, 8, dtype=torch.double, requires_grad=True)
        h = torch.randn(3, 2, 4, 8, dtype=torch.double, requires_grad=True)
        proj = Projection(16, 8, bias=False).double()
        layer = NS(
            config=NS(sequence_parallel=False, fp8=False),
            sequence_parallel=False,
            enorm=torch.nn.Identity(),
            hnorm=torch.nn.Identity(),
            eh_proj=proj,
            transformer_layer=IdentityBlock(),
        )
        actual = forward(layer, e, h, None, pre_process=True, post_process=True)
        reference = F.linear(e, proj.weight[:, :8]).unsqueeze(2) + F.linear(h, proj.weight[:, 8:])
        torch.testing.assert_close(actual, reference)
        probe = torch.randn_like(actual)
        leaves = (e, h, proj.weight)
        actual_grads = torch.autograd.grad((actual * probe).sum(), leaves, retain_graph=True)
        expected_grads = torch.autograd.grad((reference * probe).sum(), leaves)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b)

    def test_auxiliary_loss_backpropagates_with_zero_main_gradient(self):
        from contextlib import nullcontext
        import sys
        import types

        helper_path = ROOT / "MindSpeed-LLM/mindspeed_llm/tasks/models/transformer/deepseek4/mtp.py"
        spec = importlib.util.spec_from_file_location("v4_mtp_test_helper", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        module_name = "mindspeed_llm.tasks.models.transformer.deepseek4.mtp"
        old = sys.modules.get(module_name)
        sys.modules[module_name] = helper

        class Scale(torch.autograd.Function):
            @staticmethod
            def forward(ctx, main, loss):
                ctx.save_for_backward(loss)
                return main

            @staticmethod
            def backward(ctx, grad):
                return grad, torch.ones_like(ctx.saved_tensors[0])

        class Embedding(torch.nn.Embedding):
            @property
            def word_embeddings(self):
                return self

            def forward(self, input_ids, position_ids):
                return super().forward(input_ids).transpose(0, 1)

        class Head(torch.nn.Linear):
            def forward(self, x, **kw):
                return super().forward(x), None

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.head_mix = torch.nn.Parameter(torch.ones(2))

            def forward(self, hidden_states, decoder_input, **kw):
                return self.proj(hidden_states + decoder_input.unsqueeze(2))

            def hc_head(self, x, **kw):
                return (x * self.head_mix.reshape(1, 1, 2, 1)).sum(2)

        parallel = NS(
            get_context_parallel_world_size=lambda: 1,
            get_context_parallel_rank=lambda: 0,
            get_context_parallel_group=lambda: None,
            get_tensor_and_data_parallel_group=lambda **kw: None,
        )
        args = NS(
            position_embedding_type="deepseek4", enable_mhc=True, is_instruction_dataset=False, context_parallel_size=1
        )
        ns = {
            "torch": torch,
            "Tensor": torch.Tensor,
            "Optional": __import__("typing").Optional,
            "InferenceParams": object,
            "PackedSeqParams": object,
            "get_args": lambda: args,
            "get_mtp_batch_list": lambda: None,
            "parallel_state": parallel,
            "MTPLossAutoScaler": Scale,
            "MTPLossLoggingHelper": NS(save_loss_to_tracker=lambda *a, **kw: None),
            "make_viewless_tensor": lambda inp, **kw: inp,
        }
        forward = function(MTP, "mtp_block_forward", ns)
        try:
            for empty, detach in ((False, False), (False, True), (True, False)):
                torch.manual_seed(5)
                layer = Layer()
                block = NS(
                    layers=[layer],
                    final_layernorms=[torch.nn.Identity()],
                    config=NS(mtp_num_layers=1, calculate_per_token_loss=False),
                    mtp_loss_scaling_factor=0.1,
                    verl_mtp=True,
                    detach_encoder=detach,
                    training=True,
                )
                emb = Embedding(7, 4)
                head = Head(4, 7, bias=False)
                raw = torch.randn(4, 1, 2, 4, requires_grad=True)
                main = torch.randn(4, 1, 4, requires_grad=True)
                ids = torch.tensor([[1, 2, 3, 4]])
                mask = torch.zeros(1, 4) if empty else torch.ones(1, 4)

                def loss_fn(labels, logits):
                    return (
                        F.cross_entropy(logits.reshape(-1, 7), labels.T.reshape(-1), reduction="none").reshape(4, 1).T
                    )

                result = forward(
                    block,
                    ids,
                    ids,
                    raw,
                    None,
                    labels=ids,
                    loss_mask=mask,
                    embedding=emb,
                    output_layer=head,
                    compute_language_model_loss=loss_fn,
                    main_hidden_states=main,
                )
                # PPO has zero advantage: MTP must still train through its auxiliary loss.
                (result.sum() * 0).backward()
                self.assertTrue(torch.isfinite(layer.proj.weight.grad).all())
                if empty:
                    self.assertEqual(layer.proj.weight.grad.abs().sum(), 0)
                else:
                    self.assertGreater(layer.proj.weight.grad.abs().sum(), 0)
                    self.assertGreater(layer.head_mix.grad.abs().sum(), 0)
                    if detach:
                        self.assertIsNone(raw.grad)
                        self.assertIsNone(emb.weight.grad)
                    else:
                        self.assertGreater(raw.grad.abs().sum(), 0)
                        self.assertGreater(emb.weight.grad.abs().sum(), 0)
        finally:
            if old is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = old

    def test_cp_shift_crosses_both_mirrored_boundaries(self):
        helper = ROOT / "MindSpeed-LLM/mindspeed_llm/tasks/models/transformer/deepseek4/mtp.py"
        spec = importlib.util.spec_from_file_location("v4_mtp_helper", helper)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for cp in (1, 2, 4, 8):
            full = torch.arange(32).reshape(1, 32)
            target = torch.cat((full[:, 1:], full.new_zeros(1, 1)), dim=1)
            half = full.shape[1] // (2 * cp)

            def shard(x, rank):
                return torch.cat(
                    (x[:, rank * half : (rank + 1) * half], x[:, 32 - (rank + 1) * half : 32 - rank * half]), dim=1
                )

            parts = [shard(full, rank) for rank in range(cp)]
            for rank in range(cp):
                self.assertTrue(torch.equal(mod.shift_cp_parts(parts, rank), shard(target, rank)))


if __name__ == "__main__":
    unittest.main()
