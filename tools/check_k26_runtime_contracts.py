#!/usr/bin/env python3
"""Check selected K26 BF16 model interfaces before starting Ray workers."""
from __future__ import annotations

import inspect


def check_runtime_contracts():
    # The caller must initialize MindSpeed before importing Core, as runtime_env.sh does.
    from megatron.core.extensions import transformer_engine as te
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_block import TransformerBlock

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
    if failures:
        raise RuntimeError(
            "K26 BF16 model interface mismatch:\n"
            + "\n".join(failures)
            + "\nApply patches/mindspeed/k26/ascend-te-model-contracts.patch and restart workers."
        )
    return len(targets) + 1


if __name__ == "__main__":
    import mindspeed.megatron_adaptor  # noqa: F401

    print(f"K26 BF16 model interfaces: {check_runtime_contracts()} checks passed")
