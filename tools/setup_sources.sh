#!/usr/bin/env bash
# Fetch the four recipe-pinned forks. Nested upstream dependencies are installed separately.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# != 0 )); then
    printf 'Usage: bash tools/setup_sources.sh\n' >&2
    exit 2
fi
paths=(third_party/MindSpeed third_party/MindSpeed-LLM third_party/mbridge third_party/verl)
# Refuse to change an existing checkout containing local edits.
for path in "${paths[@]}"; do
    if [[ -e "$ROOT/$path/.git" ]]; then
        if [[ -n "$(git -C "$ROOT/$path" status --porcelain --untracked-files=normal)" ]]; then
            printf 'Local changes in %s; commit or stash them before updating sources.\n' "$path" >&2
            exit 1
        fi
    fi
done
git -C "$ROOT" submodule sync -- "${paths[@]}"
git -C "$ROOT" -c submodule.recurse=false submodule update --init --checkout -- "${paths[@]}"
git -C "$ROOT" submodule status -- "${paths[@]}"
