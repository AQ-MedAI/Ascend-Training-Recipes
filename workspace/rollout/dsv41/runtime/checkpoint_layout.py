"""Logical checkpoint tensor inventory, shared by runtime and header audits."""
import importlib.util
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4] / "third_party"

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

def logical_config(raw):
    c=dict(raw.get('text_config',raw))
    aliases={'dim':'hidden_size','n_layers':'num_hidden_layers','n_heads':'num_attention_heads',
             'kv_source_layers':'kv_source_layer_ids','index_source_layers':'index_source_layer_ids',
             'candidate_source_layer':'candidate_source_layer_id','moe_inter_dim':'moe_intermediate_size'}
    for target,source in aliases.items():
        if source in c:
            if target in c and c[target]!=c[source]:raise ValueError(f'Conflicting config fields: {target}/{source}')
            c[target]=c[source]
    return c

def mappings(config, root=None):
    llm = Path(os.environ.get("MINDSPEED_LLM_ROOT", ROOT / "MindSpeed-LLM")) if root is None else root / "MindSpeed-LLM"
    bridge_root = Path(os.environ.get("MBRIDGE_ROOT", ROOT / "mbridge")) if root is None else root / "mbridge"
    arch=module('v41_audit_arch',llm/'mindspeed_llm/tasks/models/transformer/deepseek41/architecture.py')
    bridge=module('v41_audit_mapping',bridge_root/'mbridge/models/deepseek_v41_mapping.py')
    plans=arch.build_layer_plan(config)
    entries=[]
    def add(src,dst,shape,kind='parameter'):
        entries.append(dict(source=src,destination=dst,shape=list(shape),kind=kind))
    d=config['dim'];inter=config['moe_inter_dim'];ne=config['n_routed_experts']
    if config['n_shared_experts']!=1:raise ValueError('Current training MoE supports one shared expert')
    for p in plans:
        for e in bridge.single_rank_adapter_mappings(config,p):
            add(e.source,'decoder.'+e.destination,e.shape)
        source=f'layers.{p.layer_id}.ffn.';dest=f'decoder.backbone.layers.{p.layer_id}.moe.'
        add(source+'gate.weight',dest+'gate.weight',(ne,d))
        add(source+'gate.bias',dest+'gate.bias',(ne,),'buffer')
        for expert in [f'experts.{i}' for i in range(ne)]+['shared_experts']:
            for part,shape in [('w1',(inter,d)),('w2',(d,inter)),('w3',(inter,d))]:
                add(source+expert+'.'+part+'.weight',dest+expert+'.'+part+'.weight',shape)
    add('embed.weight','embedding.weight',(config['vocab_size'],d))
    add('head.weight','output_layer.weight',(config['vocab_size'],d))
    add('norm.weight','decoder.backbone.norm.weight',(d,))
    for key in ('source','destination'):
        if len({e[key] for e in entries})!=len(entries):raise ValueError(f'Duplicate {key}')
    return entries
