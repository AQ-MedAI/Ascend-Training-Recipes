#!/usr/bin/env bash
# Compatibility alias: all task execution uses the native main_ppo entry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/dsv41_flash_main_ppo_16npu.sh" "$@"
