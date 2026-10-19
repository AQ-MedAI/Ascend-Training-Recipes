#!/usr/bin/env python3
"""Check selected K26 BF16 model interfaces before starting Ray workers."""
from __future__ import annotations

import inspect


def check_runtime_contracts():
    # The caller must initialize MindSpeed before importing Core, as runtime_env.sh does.
    from megatron.core import optimizer as optimizer_module
    from megatron.core import parallel_state as mpu
    from megatron.core.extensions import transformer_engine as te
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_block import TransformerBlock
    from mindspeed.core.context_parallel.model_parallel_utils import initialize_model_parallel_cp_wrapper

    targets = [
        ("TEDotProductAttention", te.TEDotProductAttention.__init__, {"pg_collection"}),
        ("TELayerNormColumnParallelLinear", te.TELayerNormColumnParallelLinear.__init__, {"tp_group", "stride"}),
        ("TEColumnParallelGroupedLinear", te.TEColumnParallelGroupedLinear.__init__, {"pg_collection"}),
        ("TERowParallelGroupedLinear", te.TERowParallelGroupedLinear.__init__, {"pg_collection"}),
        ("TransformerBlock.forward", TransformerBlock.forward, {"rotary_pos_cos_sin", "padding_mask"}),
        (
            "TransformerBlock._checkpointed_forward",
            TransformerBlock._checkpointed_forward,
            {"use_inner_quantization_context", "padding_mask"},
        ),
    ]
    failures = []
    for label, function, required in targets:
        parameters = inspect.signature(function).parameters
        # These group arguments must be implemented explicitly, not swallowed
        # by **kwargs in an older adapter.
        missing = required - parameters.keys()
        if missing:
            failures.append(f"{label}: missing {sorted(missing)} ({inspect.getfile(function)})")
    if TESpecProvider().linear() is not ColumnParallelLinear:
        failures.append("TESpecProvider.linear: BF16 MLA still selects a TE Linear shim")
    if not optimizer_module.USING_PYTORCH_OPTIMIZER:
        constructor = optimizer_module.Adam
        if "adam_w_mode" not in inspect.signature(constructor).parameters:
            failures.append(f"Adam optimizer: missing adam_w_mode ({inspect.getfile(constructor)})")
    # Probe the CP wrapper's argument forwarding without creating process groups.
    core_signature = inspect.signature(mpu.initialize_model_parallel)
    received = {}

    class ProbeComplete(Exception):
        pass

    def probe(*args, **kwargs):
        bound = core_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        received.update(bound.arguments)
        raise ProbeComplete

    setattr(probe, "__signature__", core_signature)
    try:
        initialize_model_parallel_cp_wrapper(probe)(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
            use_sharp=False,
        )
    except ProbeComplete:
        pass
    expected = dict(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        use_sharp=False,
        sharp_enabled_group=None,
        create_gloo_process_groups=True,
    )
    if any(received.get(key) != value for key, value in expected.items()):
        failures.append(
            "CP initializer: positional forwarding changes Core arguments; apply ascend-cp-initialization.patch"
        )
    if failures:
        raise RuntimeError(
            "K26 BF16 model interface mismatch:\n"
            + "\n".join(failures)
            + "\nApply patches/mindspeed/k26/ascend-te-model-contracts.patch and restart workers."
        )
    return len(targets) + 3


if __name__ == "__main__":
    import mindspeed.megatron_adaptor  # noqa: F401

    print(f"K26 BF16 runtime interfaces: {check_runtime_contracts()} checks passed")
