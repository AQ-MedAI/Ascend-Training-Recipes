"""Engram row groups within vLLM's synchronized internal DP domain."""
import os

import torch.distributed as dist


def group_plan(world_size, tp, dp, shards):
    """Pinned vLLM rank order with PP=PCP=1: external replica, DP, TP."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
           for n in (world_size, tp, dp, shards)):
        raise ValueError('Engram group sizes must be positive integers')
    domain = tp * dp
    if world_size % domain or domain % shards:
        raise ValueError('Engram shards must divide one rollout TP*DP domain')
    if shards % tp and tp % shards:
        raise ValueError('Engram shards must divide TP or be a multiple of TP')
    return [list(range(start, start + shards))
            for start in range(0, world_size, shards)]


def rollout_group(worker):
    """Reuse EP/TP or create row groups before generation starts.

    Cross-TP lookup relies on the existing MoE DP dummy-batch loop. Never
    group independently scheduled external replicas together. Keep the vLLM
    coordinator alive across sleep/wake; sleeping only releases table storage.
    """
    from vllm.distributed.parallel_state import (
        get_tp_group, get_ep_group, get_world_group, init_model_parallel_group,
    )
    config = worker.vllm_config.parallel_config
    tp = config.tensor_parallel_size
    selection = os.environ.get('ROLLOUT_ENGRAM_SHARDS', '0').strip().lower()
    reuse_ep = selection == 'ep'
    shards = None if reuse_ep else int(selection)
    if shards == 0:
        shards = tp
    if shards == tp:
        return get_tp_group().device_group
    if (config.pipeline_parallel_size != 1
            or getattr(config, 'prefill_context_parallel_size', 1) != 1
            or getattr(config, 'decode_context_parallel_size', 1) != 1
            or getattr(config, 'enable_elastic_ep', False)
            or getattr(config, 'enable_dbo', False)
            or getattr(config, 'ubatch_size', 0) > 1
            or getattr(config, 'data_parallel_external_lb', False)
            or getattr(config, 'data_parallel_hybrid_lb', False)):
        fields = ('pipeline_parallel_size', 'prefill_context_parallel_size',
                  'decode_context_parallel_size', 'enable_elastic_ep', 'enable_dbo',
                  'ubatch_size', 'data_parallel_external_lb', 'data_parallel_hybrid_lb')
        actual = {name: getattr(config, name, None) for name in fields}
        raise ValueError('Independent Engram groups require PP=PCP=DCP=1, internal DP load balancing, no elastic EP or microbatch overlap; actual=' + str(actual))
    if (reuse_ep or shards > tp) and not config.enable_expert_parallel:
        raise ValueError('Cross-TP Engram groups require synchronized vLLM MoE EP/DP execution')
    if not worker.vllm_config.model_config.enforce_eager:
        raise ValueError('Independent Engram groups require eager execution')
    if reuse_ep:
        # Borrow vLLM's group: do not allocate, cache or destroy a second one.
        return get_ep_group().device_group
    world = get_world_group()
    plan = group_plan(dist.get_world_size(), tp, config.data_parallel_size, shards)
    cached = getattr(worker, '_dsv41_engram_group', None)
    if cached is not None:
        if worker._dsv41_engram_group_plan != plan:
            raise ValueError('Cannot change Engram groups in a running engine')
        return cached.device_group
    coordinator = init_model_parallel_group(
        plan, world.local_rank, dist.get_backend(world.device_group),
        group_name='v41_engram', use_device_communicator=False,
    )
    worker._dsv41_engram_group = coordinator
    worker._dsv41_engram_group_plan = plan
    return coordinator.device_group
