"""Source-independent full actor synchronization using the V4 bucket transport."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
from pathlib import Path
from queue import Queue, Empty, Full
import tempfile
import threading
import time

import torch
from .common import sha, tensor_sha, atomic_json
from .inventory import targets
from .buckets import transport_classes, socket_timeout, chunks, content_digest

def sync_weights(llm,identity,producer,model,crop,tp,local_tmp='/tmp',bucket_mb=64,timeout=300,progress_path=None):
    if bucket_mb<1 or timeout<=0:raise ValueError('invalid transfer limits')
    if Path('/dev/shm').is_dir() and shutil.disk_usage('/dev/shm').free < tp*(bucket_mb<<20):
        raise ValueError('insufficient /dev/shm for receiver buckets; reduce --bucket-mb or enlarge container shared memory')
    model=Path(model);crop=Path(crop)
    expected=targets(model,crop);Sender,_=transport_classes()
    crop_meta=json.loads((crop/'manifest.json').read_text())
    if identity['config_sha256']!=sha(model/'config.json') or identity['crop_sha256']!=crop_meta['payload_sha256']:
        raise ValueError('snapshot identity does not match model/crop')
    started=time.monotonic();last_log=[0.0];latest={}
    def progress(stage, **details):
        latest.clear();latest.update(stage=stage,seconds=time.monotonic()-started,**details)
        if progress_path is not None:
            atomic_json(progress_path,dict(status='running',**latest))
        if time.monotonic()-last_log[0]>=15:
            print('Weight sync: '+json.dumps(latest),flush=True);last_log[0]=time.monotonic()
    progress('prepare_inventory')
    queues=[Queue(maxsize=2) for _ in range(tp)];stop=threading.Event();sent={}
    # Keep IPC paths short and local; the vLLM workers are on this host.
    with tempfile.TemporaryDirectory(prefix='v41-ipc-',dir='/tmp') as ipc, tempfile.TemporaryDirectory(prefix='v41-stream-',dir=local_tmp) as scratch:
        endpoints=[f'ipc://{ipc}/{i}.sock' for i in range(tp)]
        def source(q):
            while not stop.is_set():
                try:item=q.get(timeout=1)
                except Empty:continue
                if item is None:return
                yield item
            raise RuntimeError('weight transfer aborted')
        def send(i):
            sender=Sender(endpoints[i],bucket_size_mb=bucket_mb,use_shm=True)
            socket_timeout(sender,timeout)
            asyncio.run(sender.async_send_weights(source(queues[i])))
        with ThreadPoolExecutor(max_workers=tp+1) as pool:
            receiving=None;sending=[]
            def start_transport():
                nonlocal receiving,sending
                if receiving is not None:return
                # Start the receive timeout only once the first tensor is ready.
                receiving=pool.submit(llm.collective_rpc,'dsv41_receive_buckets',
                                      args=(endpoints,expected,identity,timeout),timeout=None)
                sending=[pool.submit(send,i) for i in range(tp)]
            def put(q,item):
                while not stop.is_set():
                    if receiving.done():receiving.result();raise RuntimeError('receiver ended before transfer')
                    for future in sending:
                        if future.done() and future.exception() is not None:future.result()
                    try:q.put(item,timeout=1);return
                    except Full:continue
                raise RuntimeError('weight transfer aborted')
            def sink(name,value):
                if name in sent or name not in expected:raise ValueError('unexpected sender tensor '+name)
                sent[name]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value))
                for item in chunks(name,value,max(16,(bucket_mb<<20)//2)):
                    start_transport()
                    progress('send_buckets',tensor=name,offset=json.loads(item[0])['offset'],completed_tensors=len(sent)-1)
                    for q in queues:put(q,item)
            try:
                report=producer(sink,timeout,progress,Path(scratch))
                if set(sent)!=set(expected):raise ValueError('incomplete sender inventory')
                for q in queues:put(q,None)
                for future in sending:future.result()
                received=receiving.result()
                digest=content_digest(sent)
                if len(received)!=tp or any(r['sha256']!=digest or r['tensors']!=len(expected) for r in received):
                    raise ValueError('receiver content differs from sender')
                progress('commit',tensors=len(sent))
                committed=llm.collective_rpc('dsv41_commit_buckets',args=(digest,identity['policy_version']))
                if len(committed)!=tp or any(r['policy_version']!=identity['policy_version'] for r in committed):raise ValueError('policy commit failed')
                if progress_path is not None:
                            atomic_json(progress_path,dict(status='complete',seconds=time.monotonic()-started,tensors=len(sent)))
                return dict(schema=1,status='complete',scope='source_to_verl_bucketed_SHM_full_actor',
                    identity=report['identity'],config_sha256=report['config_sha256'],transport='verl_BucketedWeightSender_Receiver_SHM',
                    tensors=len(sent),sha256=digest,receivers=received,committed=committed,
                    export_performance=report['export_performance'],
                    limitation='Full logical tensors streamed before rollout TP loading.')
            except BaseException as exc:
                failure=dict(status='error',error=f'{type(exc).__name__}: {exc}',last_progress=dict(latest))
                # Emit before executor cleanup, which may wait on receiver timeouts.
                print('Weight sync FAILED: '+json.dumps(failure),flush=True)
                if progress_path is not None:
                            atomic_json(progress_path,failure)
                raise
            finally:stop.set()


def sync_resident(llm, source, model, crop, tp, **kwargs):
    from .resident import inspect_resident, stream_resident
    identity, _ = inspect_resident(source)
    def produce(sink, timeout, progress, scratch):
        return stream_resident(source, model, crop, sink, timeout, progress)
    return sync_weights(llm, identity, produce, model, crop, tp, **kwargs)
