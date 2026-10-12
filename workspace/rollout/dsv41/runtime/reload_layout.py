"""Adapt checkpoint axes to the native loader without changing weight storage."""
from contextlib import contextmanager
import re


def cpu_expert_weight(name):
    """Keep individual BF16 expert matrices on host for native EP/TP loading."""
    return re.fullmatch(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", name) is not None



def expert_reload_parameters(core):
    """Validate post-load storage against the native expert mesh, not dense TP.

    vLLM owns global-to-local expert mapping and expert TP slicing. Its local
    dimensions also include any kernel padding, so do not derive them from
    the checkpoint's global expert count or the attention TP group.
    """
    parameters = []
    for layer in core.layers:
        experts = layer.mlp.experts.routed_experts
        if type(experts.quant_method).__name__ != 'AscendUnquantizedFusedMoEMethod':
            raise ValueError('only unquantized MoE supported')
        if any(getattr(experts, name, None) is not None
               for name in ('w13_weight_list', 'w2_weight_list')):
            raise ValueError('derived MoE lists unsupported')
        count = experts.local_num_experts
        hidden = experts.hidden_size
        intermediate = experts.intermediate_size_per_partition
        for part, shape in (
            ('w13', (count, hidden, 2 * intermediate)),
            ('w2', (count, intermediate, hidden)),
        ):
            param = getattr(experts, part + '_weight')
            if getattr(param, 'is_transposed', False) or tuple(param.shape) != shape:
                raise ValueError(
                    f'unsupported expert layout {part}: actual={tuple(param.shape)}, '
                    f'expected local expert mesh={shape}'
                )
            parameters.append(param)
    return parameters


@contextmanager
def native_expert_reload(parameters):
    """Unquantized Ascend runtime weights are [E, input, output].

    Delegate expert ownership and TP slicing to the original vLLM loader.
    Only the incoming 2D checkpoint tensor is transposed; live parameters
    retain their shape, strides, identity and sleep-managed allocation.
    Generation must remain stopped for the duration of the reload.
    """
    saved = []
    try:
        for param in parameters:
            loader = getattr(param, 'weight_loader', None)
            if not callable(loader):
                raise ValueError('expert parameter has no native weight_loader')
            present = hasattr(param, 'is_transposed')
            flag = getattr(param, 'is_transposed', False)
            if flag:
                raise ValueError('expert loader already uses transposed checkpoint axes')
            saved.append((param, loader, present, flag, param.data_ptr(), tuple(param.shape), param.stride()))

            def load(param, loaded_weight, *args, _loader=loader, **kwargs):
                if loaded_weight.ndim != 2:
                    raise ValueError('expert reload requires individual 2D checkpoint weights')
                return _loader(param, loaded_weight.t(), *args, **kwargs)

            param.is_transposed = True
            param.weight_loader = load
        yield
        for param, _, _, _, ptr, shape, stride in saved:
            if param.data_ptr() != ptr or tuple(param.shape) != shape or param.stride() != stride:
                raise RuntimeError('native MoE loader changed sleep-managed storage/layout')
    finally:
        for param, loader, present, flag, *_ in saved:
            param.weight_loader = loader
            if present:
                param.is_transposed = flag
            else:
                delattr(param, 'is_transposed')
