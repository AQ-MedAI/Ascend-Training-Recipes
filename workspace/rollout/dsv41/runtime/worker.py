"""Eager-only named RPC: replicated cropped Engram with original hashes and zero misses.

Cropped-table adapter in the training original basis, with full text weight
stream loading. Does not allocate the original full tables.
"""
from pathlib import Path
import json
from types import MethodType, SimpleNamespace
import torch
from torch import nn
from safetensors import safe_open
from .components import component
from .common import sha
from .slot_mapping import install_per_group_slot_mapping

# Installed when vLLM imports this worker extension, before runner initialization.
install_per_group_slot_mapping()


def read_payload(crop, source, snapshot=None):

    meta=json.loads((crop/'manifest.json').read_text())
    if meta['schema']!=1 or meta['status']!='prepared' or sha(crop/'engram.safetensors')!=meta['payload_sha256']:
        raise ValueError('invalid Engram crop')
    if sha(source/'config.json')!=meta['source_config_sha256']:raise ValueError('original Engram config differs')
    payload={}
    with safe_open(crop/'engram.safetensors',framework='pt',device='cpu') as f:
        for spec in meta['layers']:
            layer=str(spec['layer'])
            for name in ('rows','original_ids','q_weight','k_weight','wkv.weight'):
                payload[layer+'.'+name]=f.get_tensor(layer+'.'+name)
    version=meta.get('policy_version','crop_initial')
    if snapshot:
        manifest=json.loads((snapshot/'manifest.json').read_text())
        md=manifest['metadata']
        if manifest['schema']!=1 or manifest['status']!='complete' or md['crop_sha256']!=meta['payload_sha256'] or md['config_sha256']!=meta['model_config_sha256']:
            raise ValueError('actor snapshot/crop identity mismatch')
        required={}
        for spec in meta['layers']:
            layer=str(spec['layer'])
            required[f'model/engrams.{layer}.rows']=layer+'.rows'
            for name in ('q_weight','k_weight','wkv.weight'):required[f'model/engrams.{layer}.gate.{name}']=layer+'.'+name
            required[f'extra/{layer}.original_ids']=layer+'.original_ids'
        seen=set()
        for part in manifest['parts']:
            if not set(part['keys']) & set(required):continue
            path=snapshot/part['file'];path.resolve().relative_to(snapshot.resolve())
            if sha(path)!=part['sha256']:raise ValueError('actor snapshot checksum mismatch')
            tensors=torch.load(path,map_location='cpu',weights_only=True)
            for key in set(tensors)&set(required):
                dest=required[key];value=tensors[key]
                if key in seen or value.shape!=payload[dest].shape:raise ValueError('actor Engram layout mismatch')
                if dest.endswith('original_ids') and not torch.equal(value,payload[dest]):raise ValueError('actor row mapping mismatch')
                payload[dest]=value;seen.add(key)
            del tensors
        if seen!=set(required):raise ValueError('incomplete actor Engram state')
        version=md['policy_version']
    if any(not torch.isfinite(v).all() for v in payload.values()):raise ValueError('nonfinite Engram payload')
    return meta,payload,version


class CroppedTable(nn.Module):
    def __init__(self,rows,original_ids,device):
        super().__init__();self.ops=component('cropped_engram_lookup')
        self.ids=self.ops.validate_original_ids(original_ids,len(rows))
        self.rows=nn.Parameter(rows.to(device=device,dtype=torch.bfloat16),requires_grad=False)
        self.force_missing=False;self.reset_stats()

    def reset_stats(self):self.stats=dict(requested=0,hits=0,misses=0,nonzero_missing=0)

    def forward(self,hashes):
        indices=self.ops.remap_original_ids(self.ids,hashes)
        if self.force_missing:indices.fill_(-1)
        valid=indices>=0;values=self.ops.lookup_cropped_rows(self.rows,indices.to(self.rows.device))
        self.stats['requested']+=indices.numel();self.stats['hits']+=int(valid.sum());self.stats['misses']+=int((~valid).sum())
        self.stats['nonzero_missing']+=int((values[~valid.to(values.device)]!=0).sum())
        return values


def prepare_cropped(self,input_ids,positions):
    from vllm.forward_context import get_forward_context
    from vllm_ascend.models.deepseek_v41.engram_hash import engram_history_metadata
    state=self._cropped_engram_state
    hashes=torch.empty((0,len(state['layers']),state['columns']),dtype=torch.int64,device='cpu')
    mask=torch.empty(0,dtype=torch.bool,device='cpu')
    metadata=get_forward_context().attn_metadata
    if metadata is not None:
        first=self.layers[0].self_attn.dsa_attn.swa_cache_layer
        boundaries,blocks,size=engram_history_metadata(metadata[first.prefix])
        n=int(boundaries[-1]);requests=torch.repeat_interleave(torch.arange(len(boundaries)-1,device="cpu"),boundaries.diff())
        hashes,mask=self.engram_history.update(input_ids[:n].cpu().long(),positions[:n].cpu().long(),requests,blocks,size)
    values={layer:self.layers[layer].engram.embed(hashes[:,slot]).flatten(1) for slot,layer in enumerate(state['layers'])}
    return values,mask.to(positions.device)


class CroppedEngramWorkerExtension:
    def monkey_patch_model(self, vocab_size):
        # Reuse verl's OOV protection. V4.1 MoE loading uses its checked adapter,
        # not the legacy architecture-specific MoE patch.
        from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits
        monkey_patch_compute_logits(self.get_model(), vocab_size)

    @torch.no_grad()
    def update_weights_from_ipc(self, peft_config=None, base_sync_done=False, use_shm=True):
        """V4's worker RPC entry; retain only V4.1 model-loading adaptation."""
        if peft_config is not None or not use_shm:
            raise ValueError('V4.1 native adapter requires full weights over SHM')
        from vllm.distributed import get_tensor_model_parallel_world_size
        from verl.workers.rollout.vllm_rollout.utils import get_device_uuid
        from .inventory import targets
        from .buckets import content_digest
        core=self.get_model().language_model.model
        state=core._cropped_engram_state
        crop=Path(state['crop']) if state.get('mode') != 'full' else None
        model=Path(self.vllm_config.model_config.model)
        expected=targets(model,crop,source=state['source'])
        identity=dict(config_sha256=sha(model/'config.json'),
            crop_sha256=json.loads((crop/'manifest.json').read_text())['payload_sha256'] if crop else None,
            inventory_sha256=content_digest(expected),policy_version='native_verl')
        endpoint=f'ipc:///tmp/rl-colocate-zmq-{get_device_uuid(self.device.index)}.sock'
        endpoints=[endpoint]*get_tensor_model_parallel_world_size()
        receipt=self.dsv41_receive_buckets(endpoints,expected,identity,1800,native_sources=[endpoints])
        self.dsv41_commit_buckets(identity['inventory_sha256'],'native_verl')
        return receipt

    def dsv41_full_engram_install_managed(self, source):
        from vllm_ascend.device_allocator.camem import CaMemAllocator
        from .full_engram import install
        allocator = CaMemAllocator.get_instance()
        self._dsv41_original_weights_pool = allocator.allocator_and_pools.get('weights')
        with allocator.use_memory_pool(tag='weights'):
            return install(self, source)

    def dsv41_memory_status(self):
        import os
        torch.npu.synchronize()
        free,total = torch.npu.mem_get_info()
        return dict(pid=os.getpid(), allocated_bytes=torch.npu.memory_allocated(),
                    reserved_bytes=torch.npu.memory_reserved(), device_free_bytes=free, device_total_bytes=total)

    def dsv41_crop_install_managed(self, *args):
        from vllm_ascend.device_allocator.camem import CaMemAllocator
        allocator = CaMemAllocator.get_instance()
        # A second weights pool replaces the allocator's dictionary entry. Keep
        # the original pool alive as required by the upstream allocator.
        self._dsv41_original_weights_pool = allocator.allocator_and_pools.get('weights')
        with allocator.use_memory_pool(tag='weights'):
            return self.dsv41_crop_install(*args)

    @torch.no_grad()
    def dsv41_crop_install(self,crop_dir,source_dir,snapshot_dir=None):
        from transformers import AutoTokenizer
        from vllm_ascend.models.deepseek_v41.engram_hash import PagedNgramHistory
        from .engram_config import hash_arguments
        crop,source=Path(crop_dir),Path(source_dir)
        meta,payload,version=read_payload(crop,source,Path(snapshot_dir) if snapshot_dir else None)
        model=self.get_model().language_model.model
        if hasattr(model,'_cropped_engram_state'):raise ValueError('cropped Engram already installed')
        if any(layer.engram is not None for layer in model.layers):raise ValueError('base model must have Engram disabled')
        if not self.vllm_config.model_config.enforce_eager:raise ValueError('cropped diagnostic requires enforce_eager')

        if sha(Path(self.vllm_config.model_config.model)/'config.json')!=meta['model_config_sha256']:
            raise ValueError('rollout model config differs from crop')
        raw=json.loads((source/'config.json').read_text());config=raw.get('text_config',raw)
        args=hash_arguments(config,meta['global_tokens'])
        for name in ('image_token_id','image_pad_token_id'):
            value=config.get(name,raw.get(name,getattr(model.config,name,None)))
            if value is None and name=='image_pad_token_id':value=args.image_token_id+1
            if value is None:raise ValueError('missing original hash image marker '+name)
            setattr(args,name,value)
        tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
        with torch.device('cpu'):history=PagedNgramHistory(args,tokenizer)
        layers=[int(x['layer']) for x in meta['layers']]
        if layers!=list(args.engram_layer_ids):raise ValueError('hash layout layer order mismatch')
        device=next(model.parameters()).device;dim=model.config.hidden_size;hc=model.hc_mult
        if dim!=meta['hidden_size'] or hc!=meta['hc_mult']:raise ValueError('Engram residual layout mismatch')
        modules={}
        for layer in layers:
            prefix=str(layer)+'.';w=payload[prefix+'wkv.weight']
            if w.shape[0]!=dim*(hc+1):raise ValueError('Engram projection shape mismatch')
            module=nn.Module();module.embed=CroppedTable(payload[prefix+'rows'],payload[prefix+'original_ids'],device)
            with torch.device('meta'):module.wkv=nn.Linear(w.shape[1],w.shape[0],bias=False)
            module.wkv.to_empty(device=device);module.wkv.to(dtype=torch.bfloat16)
            module.wkv.weight.copy_(w.to(device=device,dtype=torch.bfloat16));module.wkv.weight.requires_grad_(False)
            for name in ('q_weight','k_weight'):
                value=payload[prefix+name]
                if value.shape!=(hc,dim):raise ValueError('Engram channel weight shape mismatch')
                setattr(module,name,nn.Parameter(value.float().to(device),requires_grad=False))
            modules[layer]=module
        # This adapter intentionally matches the current training gate's original basis.
        # It makes no equivalence claim for the upstream rotated/quantized model.
        model.engram_rotation.copy_(torch.eye(model.engram_rotation.shape[0],device=device))
        for layer,module in modules.items():model.layers[layer].engram=module
        model.engram_history=history;model._engram_input_buffers=None
        model._cropped_engram_state=dict(layers=layers,columns=(args.engram_max_ngram_size-1)*args.engram_n_heads,
                                        version=version,crop=str(crop),source=str(source))
        model.prepare_engram=MethodType(prepare_cropped,model)
        return dict(installed=True,layers=layers,version=version,missing_policy='zero_per_hash_row',
                    basis='training_original_basis_identity_rotation',scope='Engram_only_not_full_actor_sync')

    @torch.no_grad()
    def dsv41_crop_mode(self,force_missing=False):
        model=self.get_model().language_model.model
        for layer in model._cropped_engram_state['layers']:
            table=model.layers[layer].engram.embed
            if model._cropped_engram_state.get('mode') != 'full':
                table.force_missing=bool(force_missing);table.reset_stats()
            elif force_missing:
                raise ValueError('Full tables have no missing-row fallback')
        model.engram_history.pages.clear();model._engram_input_buffers=None
        return dict(force_missing=bool(force_missing))

    def dsv41_crop_stats(self):
        model=self.get_model().language_model.model
        return {str(layer):dict(model.layers[layer].engram.embed.stats) for layer in model._cropped_engram_state['layers']}

    @torch.no_grad()
    def dsv41_crop_replace(self,crop_dir,source_dir):
        model=self.get_model().language_model.model
        meta,payload,version=read_payload(Path(crop_dir),Path(source_dir))
        layers=[int(x['layer']) for x in meta['layers']]
        if layers!=model._cropped_engram_state['layers']:raise ValueError('cannot change Engram layer topology')
        for layer in layers:
            module=model.layers[layer].engram;prefix=str(layer)+'.'
            if not torch.equal(module.embed.ids,payload[prefix+'original_ids']):raise ValueError('cannot change Engram row map in hot reload')
            pairs=[(module.embed.rows,payload[prefix+'rows']), (module.wkv.weight,payload[prefix+'wkv.weight']),
                   (module.q_weight,payload[prefix+'q_weight']), (module.k_weight,payload[prefix+'k_weight'])]
            for param,value in pairs:
                if param.shape!=value.shape:raise ValueError('Engram shape changed')
                param.copy_(value.to(param.device,param.dtype))
        model._cropped_engram_state.update(crop=crop_dir,version=meta.get('policy_version',version))
        self.dsv41_crop_mode(False)
        return dict(version=model._cropped_engram_state['version'],layers=layers)

    def dsv41_full_digest(self):
        import hashlib
        import torch.distributed as dist
        model=self.get_model();digest=hashlib.sha256();count=0
        for name,param in sorted(model.named_parameters()):
            digest.update(name.encode());digest.update(str((tuple(param.shape),param.dtype)).encode())
            digest.update(param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes());count+=1
        version=getattr(self,'_full_actor_version',None)
        if version is None:
            manifest=Path(self.vllm_config.model_config.model)/'actor_export.json'
            version=json.loads(manifest.read_text())['identity']['policy_version'] if manifest.exists() else 'base'
        from .derived_weights import derived_digest
        return dict(rank=dist.get_rank(),parameter_tensors=count,sha256=digest.hexdigest(),policy_version=version,derived_weights=derived_digest(model))

    @torch.no_grad()
    def dsv41_full_actor_load(self,model_dir,crop_dir,source_dir,is_actor=True):
        """Quiescent eager native-mesh load. On failure terminate the engine."""
        from .full_sync import verify_bundle, iter_weights
        root=Path(model_dir);verify_bundle(root,require_actor=is_actor)
        core=self.get_model().language_model.model

        if sha(root/'config.json')!=sha(Path(self.vllm_config.model_config.model)/'config.json'):
            raise ValueError('full reload cannot change model topology')
        from .reload_layout import expert_reload_parameters, native_expert_reload, cpu_expert_weight
        experts=expert_reload_parameters(core)
        self._full_actor_version='loading'
        # Use the same native expert mapping and storage-preserving adapter as IPC.
        with native_expert_reload(experts):
            loaded=set(self.get_model().load_weights(iter_weights(root)))
        from .derived_weights import refresh_derived_weights
        derived_refresh=refresh_derived_weights(self.get_model())
        self.dsv41_crop_replace(crop_dir,source_dir)
        expected={name for name,_ in self.get_model().named_parameters() if name.startswith('language_model.') and '.engram.' not in name}
        missing=sorted(expected-loaded)
        if missing:raise ValueError('full actor loader omitted parameters: '+str(missing[:20]))
        torch.npu.synchronize()
        version=json.loads((root/'actor_export.json').read_text())['identity']['policy_version'] if is_actor else 'base'
        self._full_actor_version=version
        return dict(loaded_parameters=len(loaded),expected_text_parameters=len(expected),missing_parameters=missing,
                    policy_version=version,engram_included=True,derived_refresh=derived_refresh,parameter_digest=self.dsv41_full_digest())

    @torch.no_grad()
    def dsv41_receive_buckets(self,endpoints,expected,identity,timeout=300,native_sources=None):
        """V4 verl SHM receiver; generation is forbidden until collective commit."""
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        from .buckets import Assembler, transport_classes, socket_timeout
        from .derived_weights import refresh_derived_weights
        rank=get_tensor_model_parallel_rank();tp=get_tensor_model_parallel_world_size()
        if len(endpoints)!=tp:raise ValueError('receiver TP endpoint mismatch')
        model=self.get_model();core=model.language_model.model

        if identity['config_sha256']!=sha(Path(self.vllm_config.model_config.model)/'config.json'):
            raise ValueError('stream model topology mismatch')
        state=core._cropped_engram_state
        if state.get('mode') != 'full':
            crop_meta=json.loads((Path(state['crop'])/'manifest.json').read_text())
            if identity['crop_sha256']!=crop_meta.get('parent_payload_sha256',crop_meta['payload_sha256']):
                raise ValueError('stream Engram row-map identity mismatch')
        if not self.vllm_config.model_config.enforce_eager:raise ValueError('stream adapter requires eager mode')
        from .reload_layout import expert_reload_parameters, native_expert_reload, cpu_expert_weight
        experts=expert_reload_parameters(core)
        self._full_actor_version='loading';self._bucket_ready=None
        loaded=set()
        # Per-tensor barriers are diagnostic only; V4 synchronizes each bucket.
        import os
        trace=os.environ.get('V41_TRACE_WEIGHT_LOAD')=='1'
        def synchronize():
            if trace:torch.npu.synchronize()
        def apply(name,value):
            # Diagnose asynchronous faults at the receiver, not sender queue progress.
            phase='before_load'
            detail=dict(tensor=name,rank=rank,source_shape=list(value.shape),
                        source_dtype=str(value.dtype),source_device=str(value.device))
            try:
                synchronize()
                if name.startswith('engram/'):
                    layer,suffix=name[7:].split('.',1)
                    if int(layer) not in state['layers']:raise ValueError('unknown Engram layer')
                    module=core.layers[int(layer)].engram
                    if state.get('mode') == 'full' and suffix == 'embed.original_ids':
                        from .full_engram import verify_original_ids
                        verify_original_ids(module.embed, value)
                        return
                    if state.get('mode') == 'full' and suffix.startswith('embed.weight.rows.'):
                        from .full_engram import apply_rows
                        start,end=map(int,suffix[len('embed.weight.rows.'):].split('.'))
                        apply_rows(module.embed,start,end,value)
                        synchronize()
                        return
                    table={'rows':getattr(module.embed,'rows',None),'wkv.weight':module.wkv.weight,
                           'q_weight':module.q_weight,'k_weight':module.k_weight}
                    param=table[suffix]
                    detail.update(destination_shape=list(param.shape),destination_stride=list(param.stride()),
                                  destination_dtype=str(param.dtype),destination_device=str(param.device),
                                  destination_storage_bytes=param.untyped_storage().nbytes())
                    if param.shape!=value.shape:raise ValueError('Engram stream shape mismatch')
                    phase='engram_host_to_device'
                    incoming=value.to(device=param.device,dtype=param.dtype)
                    synchronize()
                    phase='engram_parameter_copy'
                    param.copy_(incoming)
                    synchronize()
                else:
                    phase='native_model_load'
                    loaded.update(model.load_weights([(name,value)]))
                    synchronize()
            except Exception as exc:
                detail.update(phase=phase,error=str(exc))
                print('V41 weight receiver failure: '+json.dumps(detail),flush=True)
                raise RuntimeError('V41 weight receiver failure: '+json.dumps(detail)) from exc
        with native_expert_reload(experts):
            if native_sources is None:
                assembler=Assembler(expected,apply);_,Receiver=transport_classes()
                receiver=Receiver(endpoints[rank],device=torch.device('cpu'),use_shm=True)
                socket_timeout(receiver,timeout)
                receiver.receive_weights(assembler.bucket)
                receipt=assembler.finish()
            else:
                from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver
                import time
                started = time.perf_counter()
                stats = dict(mode=state.get('mode', 'crop'), buckets=0,
                    received_bytes=0, engram_chunks=0, engram_cpu_bytes=0,
                    engram_local_copy_bytes=0, engram_nonlocal_bytes=0, expert_cpu_bytes=0, expert_cpu_tensors=0,
                    load_callback_seconds=0.0)
                row_selections = {}
                if state.get('mode') == 'full':
                    from .engram_transfer import select_engram_rows
                    layers = {int(name[7:].split('.')[0]) for name in expected
                              if name.startswith('engram/') and '.embed.weight.rows.' in name}
                    expected, row_selections = select_engram_rows(expected,
                        {layer: core.layers[layer].engram.embed for layer in layers})
                seen=set()
                def load_bucket(weights):
                    load_start = time.perf_counter()
                    stats['buckets'] += 1
                    for name,value in weights:
                        if name in seen or name not in expected or list(value.shape)!=expected[name]:
                            raise ValueError('native bucket coverage/shape mismatch: '+name)
                        stats['received_bytes'] += value.numel() * value.element_size()
                        if cpu_expert_weight(name):
                            if value.device.type != 'cpu':
                                raise RuntimeError('Expert checkpoint must remain on CPU for native EP/TP slicing')
                            stats['expert_cpu_bytes'] += value.numel() * value.element_size()
                            stats['expert_cpu_tensors'] += 1
                        if (state.get('mode') == 'full' and name.startswith('engram/')
                                and '.embed.weight.rows.' in name):
                            layer, interval = name[7:].split('.embed.weight.rows.')
                            start, end = map(int, interval.split('.'))
                            table = core.layers[int(layer)].engram.embed
                            local_rows = max(0, min(end, table.end) - max(start, table.start))
                            local_bytes = local_rows * table.width * value.element_size()
                            if value.device.type != 'cpu':
                                raise RuntimeError('Full Engram chunk must remain in CPU shared memory')
                            stats['engram_chunks'] += 1
                            stats['engram_cpu_bytes'] += value.numel() * value.element_size()
                            stats['engram_local_copy_bytes'] += local_bytes
                            stats['engram_nonlocal_bytes'] += value.numel() * value.element_size() - local_bytes
                        apply(name,value)
                        seen.add(name)
                    stats['load_callback_seconds'] += time.perf_counter() - load_start
                for source in native_sources:
                    receiver=BucketedWeightReceiver(source[rank],device=self.device,use_shm=True)
                    socket_timeout(receiver,timeout)
                    receiver.receive_weights(load_bucket, row_selections=row_selections, keep_on_cpu=lambda name: cpu_expert_weight(name) or (
                        state.get("mode") == "full" and name.startswith("engram/")
                        and ".embed.weight.rows." in name))
                if seen!=set(expected):raise ValueError('native buckets omitted weights')
                stats['receive_seconds'] = time.perf_counter() - started
                stats['timing_note'] = 'receive includes sender waits and bucket barriers; callback is host wall time'
                print('V41_WEIGHT_SYNC ' + json.dumps(dict(rank=rank, **stats)), flush=True)
                receipt=dict(transfer_stats=stats,tensors=len(seen),sha256=identity['inventory_sha256'],
                             validation='source_inventory_and_receiver_coverage; no payload rehash')
        required={name for name,_ in model.named_parameters() if name.startswith('language_model.') and '.engram.' not in name}
        if required-loaded:raise ValueError('stream omitted model parameters: '+str(sorted(required-loaded)[:20]))
        derived=refresh_derived_weights(model);self.dsv41_crop_mode(False);torch.npu.synchronize()
        receipt.update(rank=rank,loaded_parameters=len(loaded),derived_refresh=derived,policy_version=identity['policy_version'])
        self._bucket_ready=receipt
        return receipt

    def dsv41_commit_buckets(self,digest,version):
        ready=getattr(self,'_bucket_ready',None)
        if ready is None or ready['sha256']!=digest or ready['policy_version']!=version:
            raise ValueError('cannot commit unverified stream')
        self._full_actor_version=version
        self.get_model().language_model.model._cropped_engram_state['version']=version
        return dict(policy_version=version,sha256=digest)
