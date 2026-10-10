#!/usr/bin/env bash
set -euo pipefail

# Usage: bash pre-commit.sh [--all-files | --files path ...]
# Install the commit hook once per clone with: pre-commit install
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v pre-commit >/dev/null 2>&1; then
    printf '%s\n' 'pre-commit is required. Install it with: uv tool install pre-commit' >&2
    exit 127
fi

exec pre-commit run --config .pre-commit-config.yaml "$@"
