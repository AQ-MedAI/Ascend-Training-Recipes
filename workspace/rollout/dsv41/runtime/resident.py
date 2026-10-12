"""Native mbridge CPU weight services and readers; no optimizer ownership."""
import hashlib
import json
import os
import secrets
import threading
import time
import re
from pathlib import Path
import torch
import zmq
from .common import atomic_json, tensor_sha

class RankWeightService:
    def __init__(self,model,identity,path,timeout=7200,transfer_timeout=300):
        self.values={n:p.detach() for n,p in model.state_dict(keep_vars=True).items()}
        if any(v.device.type!='cpu' for v in self.values.values()):raise ValueError('park model before serving')
        self.context=zmq.Context();self.socket=self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER,0);self.socket.setsockopt(zmq.SNDTIMEO,int(transfer_timeout*1000))
        port=self.socket.bind_to_random_port('tcp://127.0.0.1');self.stop=threading.Event();self.error=None
        self.token=secrets.token_hex(32);self.timeout=timeout;self.transfer_timeout=transfer_timeout
        tensors={n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tensor_sha(v)) for n,v in self.values.items()}
        self.manifest=dict(schema=1,status='ready',pid=os.getpid(),identity=identity,
                           endpoint=f'tcp://127.0.0.1:{port}',token=self.token,tensors=tensors)
        self.path=Path(path)
        self.thread=threading.Thread(target=self.serve,daemon=True);self.thread.start()
        atomic_json(self.path,self.manifest);self.path.chmod(0o600)

    def serve(self):
        deadline=time.monotonic()+self.timeout
        current=None
        try:
            while not self.stop.is_set():
                if time.monotonic()>deadline:raise TimeoutError('resident weight service timed out')
                if not self.socket.poll(500):continue
                request=self.socket.recv_json()
                if request.get('token')!=self.token or request.get('name') not in self.values:
                    self.socket.send_json({'error':'invalid weight request'});continue
                current=request['name']
                value=self.values[current].contiguous()
                raw=value.reshape(-1).view(torch.uint8).numpy()
                self.socket.send(memoryview(raw),copy=False,track=True).wait(timeout=self.transfer_timeout)
        except BaseException as exc:
            self.error=exc
            failure=dict(status='error',rank=self.manifest['identity']['rank'],tensor=current,
                         error=f'{type(exc).__name__}: {exc}')
            print('Resident weight service FAILED: '+json.dumps(failure),flush=True)
            atomic_json(self.path.with_suffix('.error.json'),failure)
        finally:self.socket.close();self.context.term()

    def close(self):
        self.stop.set();self.thread.join(timeout=2*self.transfer_timeout+5)
        if self.thread.is_alive():raise TimeoutError('resident sender did not stop')
        self.values.clear()
        if self.error:raise self.error


def inspect_resident(path):
    path=Path(path);files=sorted(p for p in path.glob('rank*.json') if p.stem[4:].isdigit())
    if not files:raise ValueError('no resident rank manifests')
    rows=[json.loads(p.read_text()) for p in files];rows.sort(key=lambda m:m['identity']['rank'])
    identity={k:v for k,v in rows[0]['identity'].items() if k not in ('rank','tp_rank')}
    if identity.get('source') == 'native_mbridge_partitioned':
        if len(rows) != identity['world'] or [r['identity']['rank'] for r in rows] != list(range(identity['world'])):
            raise ValueError('incomplete native partitioned source')
        for row in rows:
            if row['status'] != 'ready' or not row['endpoint'].startswith('tcp://127.0.0.1:'):
                raise ValueError('invalid native partition endpoint/status')
            if {k:v for k,v in row['identity'].items() if k not in ('rank','tp_rank')} != identity:
                raise ValueError('mixed native policy identity')
        return identity, rows
    if identity.get('source') == 'native_mbridge_merged':
        if len(rows) != 1 or rows[0]['identity']['rank'] != 0 or rows[0]['status'] != 'ready':
            raise ValueError('invalid native merged source')
        if not rows[0]['endpoint'].startswith('tcp://127.0.0.1:'):
            raise ValueError('native source must be same-host')
        return identity, rows
    raise ValueError("unsupported native resident source")

def canonical(name):
    match = re.fullmatch(r'layers\.(\d+)\.engram\.(rows|q_weight|k_weight|wkv\.weight|embed\.original_ids)', name)
    return 'engram/'+match[1]+'.'+match[2] if match else name


def stream_merged(identity, manifest, model, crop, sink, timeout, progress, subset=False):
    import zmq
    from .inventory import targets
    from .common import tensor_sha
    expected = targets(model, crop)
    if not subset and set(manifest['tensors']) != set(expected):
        raise ValueError('Native mbridge source coverage mismatch')
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout*1000))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout*1000))
    socket.connect(manifest['endpoint'])
    try:
        for name, info in sorted(manifest['tensors'].items()):
            progress('native_mbridge_read', tensor=name)
            if info['shape'] != expected[name]:
                raise ValueError('Native source shape mismatch: '+name)
            dtype = {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32,
                     'torch.float16': torch.float16}.get(info['dtype'])
            if dtype is None:
                raise ValueError('Unsupported weight dtype: '+name)
            socket.send_json(dict(token=manifest['token'], name=name))
            frame = socket.recv(copy=False)
            value = torch.frombuffer(frame.buffer, dtype=dtype).reshape(info['shape']).clone()
            if tensor_sha(value) != info['sha256']:
                raise ValueError('Native source hash mismatch: '+name)
            sink(name, value)
    finally:
        socket.close()
        context.term()
    return dict(identity=identity, config_sha256=identity['config_sha256'],
                export_performance=dict(source='native_engine_mbridge_CPU',
                                        snapshot_reads=0, weight_file_writes=0))


def tensor_owner(name, world):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little') % world


def stream_partitioned(identity, manifests, model, crop, sink, timeout, progress):
    from .inventory import targets
    expected = targets(model, crop)
    names = [name for row in manifests for name in row['tensors']]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError('Partitioned native source coverage mismatch')
    for row in manifests:
        stream_merged(identity, row, model, crop, sink, timeout, progress, subset=True)
    return dict(identity=identity, config_sha256=identity['config_sha256'],
                export_performance=dict(source='native_mbridge_partitioned_CPU',
                                        snapshot_reads=0, weight_file_writes=0))


def stream_resident(path, model, crop, sink, timeout=300, progress=None):
    progress = progress or (lambda *args, **kwargs: None)
    identity, rows = inspect_resident(path)
    if identity['source'] == 'native_mbridge_partitioned':
        return stream_partitioned(identity, rows, model, crop, sink, timeout, progress)
    return stream_merged(identity, rows[0], model, crop, sink, timeout, progress)
