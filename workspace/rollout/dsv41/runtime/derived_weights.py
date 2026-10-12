"""Derived router weights used by Ascend forward, separate from parameters."""
import hashlib
import torch


def derived_weights(model):
    for name,module in sorted(model.named_modules()):
        cache=getattr(module,'weight_fp32',None)
        if not getattr(module,'precast_fp32_weight',False) and cache is None:continue
        weight=getattr(module,'weight',None)
        if not isinstance(cache,torch.Tensor) or not isinstance(weight,torch.Tensor):
            raise ValueError('missing derived FP32 weight: '+name)
        if cache.dtype!=torch.float32 or cache.shape!=weight.shape or cache.device!=weight.device:
            raise ValueError('invalid derived FP32 weight: '+name)
        yield name,weight,cache


@torch.no_grad()
def refresh_derived_weights(model):
    rows=[]
    for name,weight,cache in derived_weights(model):
        expected=weight.detach().float()
        changed=not torch.equal(cache,expected)
        # Preserve existing NPU storage format and references held by forward.
        cache.copy_(expected)
        if not torch.equal(cache,expected):raise ValueError('derived weight refresh failed: '+name)
        rows.append(dict(name=name,changed=changed))
    return dict(checked=len(rows),changed=sum(x['changed'] for x in rows),exact=True)


def derived_digest(model):
    digest=hashlib.sha256();count=0;stale=[]
    for name,weight,cache in derived_weights(model):
        digest.update(name.encode());digest.update(str((tuple(cache.shape),cache.dtype)).encode())
        digest.update(cache.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes());count+=1
        if not torch.equal(cache,weight.detach().float()):stale.append(name)
    return dict(sha256=digest.hexdigest(),tensors=count,consistent=not stale,stale=stale)
