"""Full text and cropped Engram transfer inventory."""
import json
import re
from pathlib import Path
from .checkpoint_layout import logical_config, mappings

def targets(model, crop, source=None):
    raw=json.loads((Path(model)/'config.json').read_text())
    shapes={x['source']:x['shape'] for x in mappings(logical_config(raw))}
    from safetensors import safe_open
    # Not trained by the text actor, but still required by the vLLM loader.
    # Match the file-export path by transmitting the unchanged base values.
    from .components import component
    index=component('full_engram_io').index(model)
    for name,filename in sorted(index.items()):
        if not re.fullmatch(r'layers\.\d+\.ffn\.gate\.bias_vl',name):continue
        path=(Path(model)/filename).resolve();path.relative_to(Path(model).resolve())
        with safe_open(path,framework='pt',device='cpu') as f:
            shapes[name]=list(f.get_slice(name).get_shape())
    if not crop:
        from .components import component
        io = component('full_engram_io')
        source = Path(source or model)
        c = json.loads((source/'config.json').read_text())
        c = c.get('text_config', c)
        for layer in c.get('engram_layer_ids', []):
            table_shape, dtype = io.header(source, f'layers.{layer}.engram.embed.weight')
            if dtype != 'BF16':
                raise ValueError('Full Engram sync requires BF16 source')
            for start, end in io.chunks(table_shape[0]):
                shapes[io.chunk_name(layer, start, end)] = [end-start, table_shape[1]]
            ids = io.original_ids(source, layer)
            if ids is not None:
                shapes[f'engram/{layer}.embed.original_ids'] = list(ids.shape)
            for suffix in ('wkv.weight', 'q_weight', 'k_weight'):
                shape, _ = io.header(source, f'layers.{layer}.engram.{suffix}')
                shapes[f'engram/{layer}.{suffix}'] = shape
        return shapes
    meta=json.loads((Path(crop)/'manifest.json').read_text())
    with safe_open(Path(crop)/'engram.safetensors',framework='pt',device='cpu') as f:
        for spec in meta['layers']:
            for suffix in ('rows','wkv.weight','q_weight','k_weight'):
                key=f'{spec["layer"]}.{suffix}'
                shapes['engram/'+key]=list(f.get_slice(key).get_shape())
    return shapes
