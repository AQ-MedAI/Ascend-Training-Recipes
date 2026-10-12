"""Read a complete exported actor bundle without materializing the full model."""
import json
from pathlib import Path
from safetensors import safe_open
from .common import sha


def verify_bundle(root,require_actor=True):
    root=Path(root)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    if require_actor:
        meta=json.loads((root/'actor_export.json').read_text())
        if meta['schema']!=1 or meta['status']!='complete' or sha(root/'config.json')!=meta['config_sha256']:
            raise ValueError('incomplete/mismatched actor export')
        if set(index)!=set(meta['tensors']) or set(index.values())!=set(meta['files']):raise ValueError('bundle tensor inventory differs')
        for file,digest in meta['files'].items():
            path=(root/file).resolve();path.relative_to(root.resolve())
            if sha(path)!=digest:raise ValueError('bundle checksum mismatch '+file)
        crop=root/'engram-crop';em=json.loads((crop/'manifest.json').read_text())
        if em['payload_sha256']!=meta['engram_payload_sha256'] or sha(crop/'engram.safetensors')!=em['payload_sha256']:
            raise ValueError('exported Engram checksum mismatch')
    return index


def iter_weights(root):
    root=Path(root);index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];grouped={}
    for name,file in index.items():grouped.setdefault(file,[]).append(name)
    for file,names in grouped.items():
        path=(root/file).resolve();path.relative_to(root.resolve())
        with safe_open(path,framework='pt',device='cpu') as f:
            for name in names:yield name,f.get_tensor(name)
