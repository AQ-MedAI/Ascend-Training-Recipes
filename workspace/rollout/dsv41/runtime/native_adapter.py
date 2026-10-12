"""V4 ServerAdapter with only V4.1 checkpoint naming and unchanged text bias."""
from pathlib import Path
import json
import re
from safetensors import safe_open
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
from .resident import canonical


class V41ServerAdapter(ServerAdapter):
    async def _server(self):
        if self.server_handle is None:
            import ray
            self.server_handle=ray.get_actor(f'vllm_server_{self.replica_rank}_{self.node_rank}')
        return self.server_handle

    async def resume(self, tags):
        if self.config.free_cache_engine and self.rollout_rank==0:
            server=await self._server()
            await server.wake_up.remote(tags=tags)

    async def release(self):
        if self.config.free_cache_engine and self.rollout_rank==0:
            server=await self._server()
            await server.sleep.remote()

    async def update_weights(self, weights, global_steps=None, **kwargs):
        root=Path(self.model_config.path)
        def checkpoint_weights():
            seen=set()
            for name,value in weights:
                name=canonical(name)
                if name in seen:raise ValueError('duplicate exported weight '+name)
                seen.add(name)
                yield name,value
            from .components import component
            index=component('full_engram_io').index(root)
            for name,file in sorted(index.items()):
                if name not in seen and re.fullmatch(r'layers\.\d+\.ffn\.gate\.bias_vl',name):
                    with safe_open(root/file,framework='pt',device='cpu') as f:
                        yield name,f.get_tensor(name)
        # Keep V4's sender, endpoint, RPC, bucket sizing and post-update cache reset.
        self.use_shm=True
        await super().update_weights(checkpoint_weights(),global_steps=global_steps,**kwargs)
