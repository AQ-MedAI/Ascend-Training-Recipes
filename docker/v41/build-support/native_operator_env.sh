#!/usr/bin/env bash
# Source before starting Python: dlopen dependency paths must exist at process startup.
# Caller supplies ROOT and PYTHON_BIN. Shared by training, CSA2 profiling and QLI smoke.
# Isolated discovery prevents training PYTHONPATH from selecting the base V4 package.
        ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-${PYTHON_BIN:-python3}}"
        native_package="$("$ROLLOUT_PYTHON" -I - <<'PY_PACKAGE'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('vllm_ascend')
if spec is None or spec.origin is None:
    raise SystemExit('Missing rollout vllm_ascend installation')
print(Path(spec.origin).resolve().parent)
PY_PACKAGE
)"
        export V41_CSA2_NATIVE_LIBRARY="${V41_CSA2_NATIVE_LIBRARY:-$("$PYTHON_BIN" - "$native_package" <<'PY_LIBRARY'
from pathlib import Path
import sys
libs = list(Path(sys.argv[1]).glob('vllm_ascend_C*.so'))
if len(libs) != 1:
    raise SystemExit(f'Expected one built vllm_ascend_C library, found {libs}')
print(libs[0])
PY_LIBRARY
)}"
        [[ -f "$V41_CSA2_NATIVE_LIBRARY" ]] || { echo "Missing native library: $V41_CSA2_NATIVE_LIBRARY" >&2; exit 1; }
        vendor="$native_package/_cann_ops_custom/vendors/custom_transformer"
        [[ -d "$vendor/op_api/lib" ]] || { echo "Missing vendor: $vendor" >&2; exit 1; }
        export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
        export LD_LIBRARY_PATH="$vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        if [[ -n "${ASCEND_HOME_PATH:-}" && -d "$ASCEND_HOME_PATH/python/site-packages/cann_ops_transformer" ]]; then
            export PYTHONPATH="$ASCEND_HOME_PATH/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
        fi
