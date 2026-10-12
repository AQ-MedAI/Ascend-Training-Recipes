"""Full BF16 tables on the existing eager Engram gate and V4 bucket receiver."""
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from safetensors import safe_open
from .components import component


@torch.no_grad()
def install(worker, source_dir):
    from transformers import AutoTokenizer
    from .engram_groups import rollout_group
    from vllm_ascend.models.deepseek_v41.engram_hash import PagedNgramHistory
    from .engram_config import hash_arguments
    from .worker import prepare_cropped
    model = worker.get_model().language_model.model
    if any(layer.engram is not None for layer in model.layers):
        raise ValueError('Full adapter requires native Engram disabled during base load')
    if not worker.vllm_config.model_config.enforce_eager:
        raise ValueError('Full Engram adapter requires enforce_eager=True')
    source = Path(source_dir)
    raw = json.loads((source / 'config.json').read_text())
    c = raw.get('text_config', raw)
    args = hash_arguments(c, worker.vllm_config.model_config.max_model_len)
    for name in ('image_token_id', 'image_pad_token_id'):
        value = c.get(name, raw.get(name, getattr(model.config, name, None)))
        if value is None and name == 'image_pad_token_id':
            value = args.image_token_id + 1
        if value is None:
            raise ValueError('Missing hash image marker ' + name)
        setattr(args, name, value)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    with torch.device('cpu'):
        history = PagedNgramHistory(args, tokenizer)
    io = component('full_engram_io')
    meta = io.small_table_metadata(source)
    if meta is not None:
        import hashlib
        path = Path(worker.vllm_config.model_config.model)/'config.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['model_config_sha256']:
            raise ValueError('Small-table fixture model identity mismatch')
    table_class = component('full_engram_table').FullEngramTable
    device = next(model.parameters()).device
    group = rollout_group(worker)
    specs = []
    required_bytes = 0
    world = torch.distributed.get_world_size(group)
    for layer in args.engram_layer_ids:
        shape, dtype = io.header(source, f'layers.{layer}.engram.embed.weight')
        if dtype != 'BF16':
            raise ValueError('Full Engram adapter requires BF16 tables')
        specs.append((layer, shape))
        required_bytes += ((shape[0] + world - 1) // world) * shape[1] * 2
        for suffix in ('wkv.weight', 'q_weight', 'k_weight'):
            gate_shape, _ = io.header(source, f'layers.{layer}.engram.{suffix}')
            required_bytes += gate_shape[0] * gate_shape[1] * 4
    free, _ = torch.npu.mem_get_info()
    if required_bytes >= free:
        raise MemoryError(f'Full Engram installation needs {required_bytes} bytes; {free} free. '
                          'Increase ROLLOUT_ENGRAM_SHARDS within the rollout TP*DP domain or lower vLLM memory utilization.')
    modules = {}
    for layer, shape in specs:
        module = nn.Module()
        module.embed = table_class(*shape, group=group, storage_device=device,
                                  communication_device=device, trainable=False, original_ids=io.original_ids(source, layer))
        io.load_table(module.embed, source, layer)
        for suffix in ('wkv.weight', 'q_weight', 'k_weight'):
            key = f'layers.{layer}.engram.{suffix}'
            with safe_open(io.source_file(source, key), framework='pt', device='cpu') as f:
                value = f.get_tensor(key)
            if suffix == 'wkv.weight':
                with torch.device('meta'):
                    module.wkv = nn.Linear(value.shape[1], value.shape[0], bias=False)
                module.wkv.weight = nn.Parameter(value.to(device, torch.bfloat16), requires_grad=False)
            else:
                setattr(module, suffix, nn.Parameter(value.to(device, torch.float32), requires_grad=False))
        modules[layer] = module
    model.engram_rotation.copy_(torch.eye(model.engram_rotation.shape[0], device=device))
    for layer, module in modules.items():
        model.layers[layer].engram = module
    model.engram_history = history
    model._engram_input_buffers = None
    model._cropped_engram_state = dict(mode='full', layers=list(args.engram_layer_ids),
        columns=(args.engram_max_ngram_size-1)*args.engram_n_heads,
        source=str(source), version='base')
    model.prepare_engram = MethodType(prepare_cropped, model)
    return dict(mode='full', table_shards=world, table_bytes=required_bytes)


@torch.no_grad()
def apply_rows(table, start, end, value):
    if not 0 <= start < end <= table.rows or tuple(value.shape) != (end-start, table.width):
        raise ValueError('Invalid full Engram row chunk')
    lo, hi = max(start, table.start), min(end, table.end)
    if lo < hi:
        table.weight[lo-table.start:hi-table.start].copy_(value[lo-start:hi-start])


def verify_original_ids(table, value):
    if table.original_ids is None or value.dtype != torch.int64 or not torch.equal(
            table.original_ids.detach().cpu(), value.detach().cpu()):
        raise ValueError('Stream Engram row-map identity mismatch')
