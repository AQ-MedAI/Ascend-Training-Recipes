"""CPU checks using the retained v0.23 cache classes and V4.1 backport."""

import importlib.util, sys, types, enum, math, logging
from pathlib import Path
from types import SimpleNamespace as NS
import torch, pytest
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--framework-root", type=Path, required=True)
args = parser.parse_args()
ROOT = args.framework_root


# Run as a standalone process: hardware import stubs must not leak into other tests.
def stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__path__ = []
    sys.modules[name] = m
    return m


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


for name in ["vllm", "vllm.v1", "vllm.v1.core", "vllm.utils", "vllm_ascend", "vllm_ascend.core"]:
    stub(name)
stub("vllm.logger", init_logger=logging.getLogger)
stub("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b), round_up=lambda a, b: -(-a // b) * b)
stub(
    "vllm.utils.torch_utils",
    get_dtype_size=lambda d: d.itemsize,
    nvfp4_kv_cache_full_dim=lambda x: x,
)
stub(
    "vllm.v1.attention.backends.registry",
    MambaAttentionBackendEnum=enum.Enum("MambaAttentionBackendEnum", {"MAMBA2": 1, "GDN_ATTN": 2}),
)
registry = load("vllm.v1.kv_cache_spec_registry", ROOT / "vllm/vllm/v1/kv_cache_spec_registry.py")
legacy = load("vllm.v1.kv_cache_interface", ROOT / "vllm/vllm/v1/kv_cache_interface.py")
stub(
    "vllm.config",
    VllmConfig=object,
    CUDAGraphMode=enum.Enum("CUDAGraphMode", {"NONE": 0, "FULL": 1, "FULL_DECODE_ONLY": 2}),
)
stub(
    "vllm.v1.core.single_type_kv_cache_manager",
    FullAttentionManager=object,
    SlidingWindowManager=object,
)
stub("vllm_ascend.core.single_type_kv_cache_manager", CompressAttentionManager=object)
stub("vllm_ascend.utils", AscendDeviceType=NS(A5=5), get_ascend_device_type=lambda: 3)
stub(
    "vllm.v1.core.kv_cache_utils",
    may_override_num_blocks=lambda c, n: (
        getattr(c.cache_config, "num_gpu_blocks_override", None) or n
    ),
)
ascend = load(
    "vllm_ascend.core.kv_cache_interface",
    ROOT / "vllm-ascend/vllm_ascend/core/kv_cache_interface.py",
)
load("vllm_ascend.core.circular_buffer", ROOT / "vllm-ascend/vllm_ascend/core/circular_buffer.py")
v41 = load("vllm_ascend.core.deepseek_v41", ROOT / "vllm-ascend/vllm_ascend/core/deepseek_v41.py")


def test_legacy_descriptor_unchanged():
    t = legacy.KVCacheTensor(size=128, shared_by=["v4"])
    assert t.size == 128 and t.shared_by == ["v4"]
    assert not hasattr(t, "block_stride")


def test_v41_compressed_page_and_padding():
    s = v41.DeepseekV41FullSpec(
        block_size=64, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, compress_ratio=2
    )
    assert s.storage_block_size == 32
    assert s.page_size_bytes == 512
    from dataclasses import replace

    assert replace(s, page_size_padded=1024).page_size_bytes == 1024
    with pytest.raises(ValueError):
        _ = replace(s, page_size_padded=256).page_size_bytes


def test_shared_page_views_do_not_overlap():
    s = v41.DeepseekV41FullSpec(
        block_size=64, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, compress_ratio=2
    )
    i = v41.DeepseekV41IndexerSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        compress_ratio=2,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    raw = torch.zeros(2 * 1024, dtype=torch.int8)
    kv = v41.reshape_cache(raw, s, num_blocks=2, offset=0, block_stride=1024)
    key, scale = v41.reshape_cache(raw, i, num_blocks=2, offset=512, block_stride=1024)
    kv.fill_(1)
    key.fill_(3)
    scale.fill_(2)
    assert torch.all(kv == 1) and torch.all(key == 3) and torch.all(scale == 2)
    assert kv.stride()[0] == 512 and key.stride()[0] == 1024
    with pytest.raises(ValueError):
        v41.reshape_cache(raw, i, num_blocks=2, offset=1000, block_stride=1024)


def test_state_spec_one_private_page():
    s = v41.DeepseekV41CompressorStateSpec(
        block_size=32, num_kv_heads=1, head_size=16, dtype=torch.float32
    )
    assert s.page_size_bytes == 2048
    assert s.max_memory_usage_bytes(NS()) == 2048
    assert not s.prefix_cacheable


def test_mixed_ratio_full_group():
    common = dict(block_size=64, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    specs = {
        "a": v41.DeepseekV41FullSpec(**common, compress_ratio=1),
        "b": v41.DeepseekV41FullSpec(**common, compress_ratio=2),
    }
    assert legacy.UniformTypeKVCacheSpecs.from_specs(specs) is not None


def test_circular_manager_retained_scheduler_signature():
    from collections import defaultdict

    manager_class = sys.modules["vllm_ascend.core.circular_buffer"].AscendCircularBufferManager
    manager = object.__new__(manager_class)
    manager.req_to_blocks = defaultdict(list)
    manager.block_pool = NS(get_new_blocks=lambda n: [object() for _ in range(n)])
    assert (
        manager.get_num_blocks_to_allocate("request", 64, [], 0, 64, apply_admission_cap=False) == 1
    )
    manager.allocate_new_computed_blocks("request", [], 0, 0)
    assert len(manager.allocate_new_blocks("request", 64, 64)) == 1
    assert manager.get_num_blocks_to_allocate("request", 128, [], 64, 128) == 0
    assert manager.allocate_new_blocks("request", 128, 128) == []
    with pytest.raises(NotImplementedError):
        manager.allocate_new_computed_blocks("other", [], 0, 64)


def test_runtime_scope_rejects_unsupported_settings():
    config = NS(
        compilation_config=NS(cudagraph_mode=sys.modules["vllm.config"].CUDAGraphMode.NONE),
        model_config=NS(enforce_eager=True),
        speculative_config=None,
        cache_config=NS(enable_prefix_caching=False, cache_dtype="auto"),
        parallel_config=NS(),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
    )
    v41.validate_cache_runtime(config)
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(NotImplementedError):
        v41.validate_cache_runtime(config)
    config.cache_config.enable_prefix_caching = False
    config.speculative_config = object()
    with pytest.raises(NotImplementedError):
        v41.validate_cache_runtime(config)


def test_v41_rms_keeps_fp32_routing_values():
    import ast

    path = ROOT / "vllm-ascend/vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41DecoderLayer"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "rms_norm_cast"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16)
    output, routing = namespace["rms_norm_cast"](
        NS(norm_eps=1e-6, post_attention_layernorm=NS(weight=weight)), values
    )
    expected = (
        values.float()
        * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    )
    torch.testing.assert_close(routing, expected)
    assert routing.dtype == torch.float32
    torch.testing.assert_close(output, expected.to(torch.bfloat16))


if __name__ == "__main__":
    # Collect this already-loaded module without executing its CLI a second time.
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
