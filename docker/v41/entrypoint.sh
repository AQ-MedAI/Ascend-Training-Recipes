#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
_recipe_root="${RECIPE_ROOT:-/opt/ascend-training/Ascend-Training-Recipes}"
if [[ ! -f "$_recipe_root/tools/activate.sh" ]]; then
    printf 'Missing recipe activation script: %s/tools/activate.sh\n' "$_recipe_root" >&2
    exit 1
fi
source "$_recipe_root/tools/activate.sh" "${RECIPE_PROFILE:-v41}"
unset _recipe_root
# Operator vendor paths are configured by the selected training launcher.
exec "$@"
