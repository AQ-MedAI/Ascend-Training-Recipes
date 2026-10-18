#!/usr/bin/env bash
# Source after selecting Python and verl, before launching the K26 training driver.
export MEGATRON_ROOT="${MEGATRON_ROOT:-/opt/ascend-training/Megatron-LM}"
export MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-/opt/ascend-training/Megatron-Bridge}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}/src:${MEGATRON_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN:-python3}" - <<'CHECK'
import os
import sys
from pathlib import Path
for key, relative in [('MEGATRON_ROOT', 'megatron/core'),
                      ('MEGATRON_BRIDGE_ROOT', 'src/megatron/bridge')]:
    expected = Path(os.environ[key]) / relative
    if not expected.is_dir():
        raise SystemExit(f'Missing K26 source: {expected}; prepare the matching K26 environment')
# MindSpeed adaptation must precede Core/Bridge imports in a fresh process.
import mindspeed.megatron_adaptor
import megatron.core
import megatron.bridge
for module, key in [(megatron.core, 'MEGATRON_ROOT'),
                    (megatron.bridge, 'MEGATRON_BRIDGE_ROOT')]:
    actual = Path(module.__file__).resolve()
    if not actual.is_relative_to(Path(os.environ[key]).resolve()):
        raise SystemExit(f'Unexpected {module.__name__}: {actual}')
    print(f'{module.__name__}: {actual}', flush=True)
# Exercise all Bridge imports used by verl; preserve the underlying traceback.
from verl.models.mcore.bridge import AutoBridge
# The registry propagates the first optional-engine import failure without retrying.
from verl.workers.engine import EngineRegistry
engine_cls = EngineRegistry.get_engine_cls('language_model', 'megatron')
print(f'K26 Megatron engine: {engine_cls.__module__}.{engine_cls.__name__}', flush=True)
print(f'K26 training Python: {sys.executable}', flush=True)
CHECK
