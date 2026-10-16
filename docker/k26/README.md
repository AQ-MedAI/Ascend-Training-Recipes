# K26 image adaptation

This is an independent **source preparation candidate**, not a runnable or validated
K26 training image. The original internal patches are unavailable. Public commits
are pinned in `sources.json`; they do not reproduce the former internal runtime.

```bash
docker build --platform linux/arm64 --target common \
  -f docker/v41/Dockerfile -t ascend-training-base:dev .
docker build --platform linux/arm64 \
  -f docker/k26/Dockerfile -t ascend-training-k26:sources .
```

The existing V4/V4.1 build remains separate. Its `common` target is reused provisionally
for CANN/Python/Torch; compatibility with K26 is not established yet. Do not base K26
on the final V4.1 image or inherit its patched Core.

The candidate fetches Core and Bridge into `/opt/ascend-training/Megatron-LM` and
`/opt/ascend-training/Megatron-Bridge`. It does not pip-install them or claim import
success. Next steps before publishing a runtime image:

1. Restore required Ascend adaptations against the pinned public sources; store patches
   and their hashes with the source manifest.
2. Resolve and lock K26 Python dependencies against the selected hardware stack.
3. Pin and build K26-compatible vLLM, vLLM-Ascend and native operators.
4. Install Core/Bridge, run the complete verl Bridge imports, then validate four-layer
   RL and MOPD including optimizer updates, rollout synchronization and checkpoint restore.

Use `aq/k26-integration` for the recipe's verl fork. All Ray nodes must use the same
runtime and absolute paths. `recipes/kimi-k2.6/runtime_env.sh` selects the intended
source paths and checks the actual Core/Bridge imports before launching training.
