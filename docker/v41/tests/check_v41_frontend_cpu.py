from pathlib import Path
import importlib.util, sys, types, pytest
import argparse

parser = argparse.ArgumentParser(
    description="CPU checks for V4.1 config and encoding; no NPU imports."
)
parser.add_argument("--framework-root", type=Path, required=True)
args = parser.parse_args()
root = args.framework_root / "vllm-ascend/vllm_ascend"
donor = args.framework_root / "vllm-ascend/tests/ut"


def pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m
    return m


pkg("vllm_ascend", root)
pkg("vllm_ascend.patch", root / "patch")
pkg("vllm_ascend.patch.platform", root / "patch/platform")
pkg(
    "vllm_ascend.patch.platform.patch_deepseek_v41_frontend",
    root / "patch/platform/patch_deepseek_v41_frontend",
)
vllm = pkg("vllm", "/nonexistent")
vllm.ModelRegistry = types.SimpleNamespace(register_model=lambda *a: None)
# Model registry is loaded without model/device imports; actual config uses Transformers 5.14.1.
spec = importlib.util.spec_from_file_location("vllm_ascend.models", root / "models/__init__.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
raise SystemExit(
    pytest.main(
        [
            "-q",
            "--confcutdir=" + str(donor / "models"),
            str(donor / "models/test_deepseek_v41_release_config.py"),
            str(donor / "patch/platform/deepseek_v41/test_encoding.py"),
        ]
    )
)
