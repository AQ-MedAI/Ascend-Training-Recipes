"""V4.1 model hooks on the existing verl Ray vLLM server and replica."""
import os
import sys
import ray
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class V41HttpServer(vLLMHttpServer):
    def _get_worker_extension_cls(self):
        return 'workspace.rollout.dsv41.runtime.worker.CroppedEngramWorkerExtension'

    def _post_init(self, cuda_visible_devices):
        # Same port isolation as the training/rollout boundary, subdivided per replica.
        for key in ('HCCL_NPU_SOCKET_PORT_RANGE', 'HCCL_HOST_SOCKET_PORT_RANGE'):
            lo, hi = map(int, os.environ[key].split('-'))
            width = (hi-lo+1)//4
            if width < 32:
                raise ValueError('HCCL range too narrow for four rollout replicas')
            local_slot = self.replica_rank % 4
            os.environ[key] = f'{lo+local_slot*width}-{lo+(local_slot+1)*width-1}'
        super()._post_init(cuda_visible_devices)

    async def run_server(self, args):
        # verl runs one frontend per replica and headless engines on remote nodes.
        # New vLLM interprets even start_rank=0 as hybrid DP load balancing.
        # Omit it on the frontend; keep remote headless start ranks unchanged.
        if args.data_parallel_size > 1:
            if args.data_parallel_external_lb or args.data_parallel_hybrid_lb:
                raise ValueError('V41 replica requires internal DP load balancing with remote headless engines')
            if args.data_parallel_start_rank not in (None, 0):
                raise ValueError('V41 replica frontend must start at DP rank 0')
            args.data_parallel_start_rank = None
        return await super().run_server(args)

    async def launch_server(self, *args, **kwargs):
        result=await super().launch_server(*args, **kwargs)
        if os.environ.get('V41_NATIVE_TRAINER')=='1' and self.node_rank==0:
            if os.environ.get('ENGRAM_CROP'):
                await self.engine.collective_rpc('dsv41_crop_install_managed',
                    args=(os.environ['ENGRAM_CROP'],os.environ['ENGRAM_SOURCE']))
            else:
                await self.engine.collective_rpc('dsv41_full_engram_install_managed',
                    args=(os.environ.get('ENGRAM_SOURCE') or self.model_config.path,))
        return result

    async def _sleep_hybrid(self):
        # V4.1 exports text weights only. Retain unchanged vision allocations.
        # New vLLM Executor.wake_up() skips workers unless Executor.sleep()
        # set is_sleeping/sleeping_tags. Direct worker RPC bypasses that state.
        await self.engine.sleep(level=1)
        await self.engine.reset_encoder_cache()


class V41Replica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(V41HttpServer)

    def _get_server_runtime_env(self, env_vars):
        if os.environ.get('V41_NATIVE_TRAINER')=='1':
            from pathlib import Path
            root=Path(__file__).resolve().parents[4]
            image=Path(os.environ.get('DEPENDENCY_ROOT','/opt/ascend-training'))
            python=os.environ.get('ROLLOUT_PYTHON',sys.executable)
            paths=[os.environ.get(key,str(image/name))
                   for key,name in (('VLLM_ROOT','vllm'),('VLLM_ASCEND_ROOT','vllm-ascend'))]
            paths += [os.environ.get('VERL_ROOT',str(root/'third_party/verl')),str(root)]
            paths += os.environ.get('PYTHONPATH','').split(os.pathsep)
            env={key:os.environ[key] for key in ('MINDSPEED_LLM_ROOT','MBRIDGE_ROOT','VERL_ROOT','V41_CONFIG_SOURCE',
                'DEPENDENCY_ROOT','ROLLOUT_PYTHON','VLLM_ROOT','VLLM_ASCEND_ROOT',
                'ENGRAM_CROP','ENGRAM_SOURCE','ROLLOUT_ENGRAM_SHARDS','ASCEND_HOME_PATH',
                'ASCEND_OPP_PATH','ASCEND_AICPU_PATH','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH') if key in os.environ}
            if os.environ.get('V41_CSA2_ISOLATE_ROLLOUT_ENV') == '1':
                # Set at actor process creation, before its dynamic loader runs.
                # Restore even empty original values; never inherit training vendor priority.
                env['LD_LIBRARY_PATH'] = os.environ['V41_CSA2_ROLLOUT_LD_LIBRARY_PATH']
                env['ASCEND_CUSTOM_OPP_PATH'] = os.environ['V41_CSA2_ROLLOUT_CUSTOM_OPP_PATH']
                env['V41_CSA2_NATIVE_LIBRARY'] = ''
                env['V41_CSA2_FORWARD_BACKEND'] = 'triton'
                env['V41_CSA2_BACKWARD_BACKEND'] = 'triton'
            from .ports import hccl_environment
            ports=hccl_environment(os.environ,'rollout')
            env.update({k:ports[k] for k in ('HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE')})
            env.update(V41_NATIVE_TRAINER='1',PYTHONPATH=os.pathsep.join(dict.fromkeys(paths)))
            return {'py_executable':python,'env_vars':{**env_vars,**env}}
        return super()._get_server_runtime_env(env_vars)
