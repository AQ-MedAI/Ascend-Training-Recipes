#!/usr/bin/env bash
# Run inside a matching training image; install recipe-owned sources without resolving dependencies.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-v41}"
[[ $# -le 1 && "$PROFILE" =~ ^(v41|v4)$ ]] || { echo 'Usage: bash tools/setup_environment.sh [v41|v4]' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" "$ROOT/tools/check_environment.py" --profile "$PROFILE"
for name in MindSpeed MindSpeed-LLM mbridge verl; do
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$ROOT/third_party/$name"
done
if ! "$PYTHON_BIN" -m pip check; then
    echo 'Dependency metadata conflicts reported above; continuing with the fixed deployment versions.' >&2
fi
printf 'Sources installed for %s. Run: source tools/activate.sh %s\n' "$PROFILE" "$PROFILE"
