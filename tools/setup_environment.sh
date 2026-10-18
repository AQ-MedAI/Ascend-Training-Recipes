#!/usr/bin/env bash
# Validate the preinstalled image; source activation before starting Ray.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-v41}"
[[ $# -le 1 && "$PROFILE" =~ ^(v41|v4)$ ]] || { echo 'Usage: bash tools/setup_environment.sh [v41|v4]' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" "$ROOT/tools/check_environment.py" --profile "$PROFILE"
printf 'Image environment checked for %s. Run: source tools/activate.sh %s\n' "$PROFILE" "$PROFILE"
