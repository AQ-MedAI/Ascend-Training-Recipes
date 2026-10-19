#!/usr/bin/env bash
# Resolve on the target platform without installing packages in the host interpreter.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-build}"
[[ $# -le 1 && "$PROFILE" =~ ^(build|runtime)$ ]] || {
    echo 'Usage: bash docker/v41/locks/compile.sh [build|runtime]' >&2; exit 2;
}
UV_BIN="${UV_BIN:-uv}"
# Match the resolver used for the checked-in lock.
[[ "$($UV_BIN --version)" == 'uv 0.11.6'* ]] || {
    echo 'Use uv 0.11.6 to regenerate these locks.' >&2; exit 2;
}
cd "$ROOT"
# Preserve the last successful lock on failure. Reuse it as constraints when regenerating.
temporary="$(mktemp "$ROOT/.${PROFILE}.XXXXXX")"
trap 'rm -f "$temporary"' EXIT
constraints=()
[[ "$PROFILE" != runtime ]] || constraints+=(--overrides runtime-overrides.txt)
[[ ! -f "$PROFILE.lock" ]] || constraints+=(--constraint "$PROFILE.lock")
"$UV_BIN" --no-config pip compile "$PROFILE.in"     --default-index https://pypi.org/simple     --python-version 3.11.15 --python-platform aarch64-manylinux_2_35     --only-binary :all: --generate-hashes --no-annotate --no-python-downloads     --custom-compile-command "bash docker/v41/locks/compile.sh $PROFILE"     "${constraints[@]}" --output-file "$temporary"
mv "$temporary" "$PROFILE.lock"
