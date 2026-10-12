import os
"""V4 SHM bucket transport and V4.1 checked tensor assembly."""
import hashlib
import json
from pathlib import Path
import torch
from .common import tensor_sha

def content_digest(tensors):
    return hashlib.sha256(json.dumps(tensors,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def chunks(name,value,chunk_bytes):
    value=value.detach().cpu().contiguous();flat=value.reshape(-1).view(torch.uint8)
    if not flat.numel():raise ValueError('empty stream weight '+name)
    width=chunk_bytes
    if width<1:raise ValueError('chunk too small')
    digest=tensor_sha(value)
    for offset in range(0,flat.numel(),width):
        header=dict(name=name,shape=list(value.shape),dtype=str(value.dtype),offset=offset,sha256=digest)
        yield json.dumps(header,separators=(',',':')),flat[offset:offset+width]


class Assembler:
    """One tensor at a time; reject missing, duplicate, reordered or corrupt chunks."""
    def __init__(self, expected, apply):
        self.expected=expected;self.apply=apply;self.seen={};self.current=None;self.value=None;self.offset=0

    def bucket(self,weights):
        for encoded,piece in weights:
            h=json.loads(encoded);name=h['name']
            if self.current is None:
                if name in self.seen or name not in self.expected or h['shape']!=self.expected[name] or h['offset']!=0:
                    raise ValueError('unexpected stream tensor '+name)
                dtype=getattr(torch,h['dtype'].removeprefix('torch.'),None)
                if dtype not in (torch.float32,torch.bfloat16,torch.float16) or piece.dtype!=torch.uint8:raise ValueError('stream dtype mismatch')
                self.current={k:v for k,v in h.items() if k!='offset'}
                self.value=torch.empty(h['shape'],dtype=dtype,device='cpu');self.offset=0
            if {k:v for k,v in h.items() if k!='offset'}!=self.current or h['offset']!=self.offset or piece.dtype!=torch.uint8:
                raise ValueError('stream chunk order/layout mismatch')
            end=self.offset+piece.numel()
            if end>self.value.nbytes:raise ValueError('stream chunk overflow')
            self.value.reshape(-1).view(torch.uint8)[self.offset:end].copy_(piece.reshape(-1));self.offset=end
            if end==self.value.nbytes:
                if tensor_sha(self.value)!=h['sha256'] or not torch.isfinite(self.value).all():raise ValueError('stream tensor checksum/finite failure')
                self.apply(name,self.value)
                self.seen[name]=dict(shape=h['shape'],dtype=h['dtype'],sha256=h['sha256'])
                self.current=None;self.value=None

    def finish(self):
        if self.current is not None or set(self.seen)!=set(self.expected):raise ValueError('incomplete weight stream')
        return dict(tensors=len(self.seen),sha256=content_digest(self.seen))


def transport_classes():
    # Loading the package normally executes old ServerAdapter/FP8 imports, which
    # are incompatible with the isolated V4.1 vLLM. Load the same transport source
    # with private dependency imports instead; never replace global verl modules.
    import ast
    import builtins
    import importlib.util
    import types
    root=Path(os.environ.get('VERL_ROOT', Path(__file__).resolve().parents[4]/'third_party/verl'))/'verl'
    def load_file(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    device=load_file('_v41_verl_device',root/'utils/device.py')
    utils_path=root/'workers/rollout/utils.py'
    tree=ast.parse(utils_path.read_text(),filename=str(utils_path))
    functions=[node for node in tree.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='ensure_async_iterator']
    if len(functions)!=1:raise RuntimeError('V4 ensure_async_iterator source not found')
    utils=types.ModuleType('_v41_verl_iterator')
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(utils_path),'exec'),utils.__dict__)
    dependencies={'verl.utils.device':device,'verl.workers.rollout.utils':utils}
    original_import=builtins.__import__
    def isolated_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level==0 and name in dependencies:return dependencies[name]
        return original_import(name,globals,locals,fromlist,level)
    path=root/'workers/rollout/vllm_rollout/bucketed_weight_transfer.py'
    module=types.ModuleType('_v41_verl_bucketed_transport')
    module.__file__=str(path)
    module.__dict__['__builtins__']=dict(vars(builtins),__import__=isolated_import)
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module.BucketedWeightSender,module.BucketedWeightReceiver


def socket_timeout(transport, seconds):
    original=transport._init_socket
    def initialize():
        import zmq
        original()
        transport.socket.setsockopt(zmq.SNDTIMEO,int(seconds*1000))
        transport.socket.setsockopt(zmq.RCVTIMEO,int(seconds*1000))
        transport.socket.setsockopt(zmq.LINGER,0)
    transport._init_socket=initialize
