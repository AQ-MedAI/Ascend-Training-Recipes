# K26 training image

Build from the Python 3.11.15 V4/V4.1 image produced by `docker/v41/Dockerfile`.
The K26 image reuses its compiled vLLM, vLLM-Ascend, CANN operators and Python
packages. It replaces the training sources and bundles the matching recipe.
The two-layer K26 RL and MOPD recipes have passed runtime tests; the rebuilt
image must still pass the same smoke tests before publication.

From the recipe repository root:

```bash
docker build --progress=plain --platform linux/arm64 \
  --build-arg BASE_IMAGE=ascend-training-v41:dev \
  --build-arg RECIPE_REVISION="$(git rev-parse HEAD)" \
  -f docker/k26/Dockerfile -t ascend-training-k26:dev .
```

Replace `BASE_IMAGE` with the actual local V4/V4.1 image tag, or a published image
reference pinned by digest. The base must contain the standardized
`/opt/ascend-training/third_party` layout and the source revisions declared in
`dependencies/framework-sources.json`. The `common` target contains no compiled
rollout runtime and cannot be used as this Dockerfile's base.

Source downloads default to GitHub's public commit archives, with retries and
full inventory verification. Set `--build-arg FRAMEWORK_SOURCE_TRANSPORT=git`
to use shallow Git fetches with HTTP/1.1 and retries instead.

## Contents

| Directory under `/opt/ascend-training` | Contents |
| --- | --- |
| `Ascend-Training-Recipes` | Recipe snapshot, launchers, reward routing, validation tools and patches |
| `third_party/Megatron-LM` | Core `96ea1d1c1a7ba8c2e0f7bbd06083e82b41187472` plus the complete K26 Core series |
| `third_party/Megatron-Bridge` | Bridge `c15303e05b3d48505e4b1b0047faa0c321368856` plus K26 text conversion and shared-filesystem export patches |
| `third_party/MindSpeed` | The shared pinned baseline plus the complete K26 compatibility series |
| `third_party/verl` | K26 profile, engine registration and full-vocabulary RPC fixes |
| `third_party/vllm`, `third_party/vllm-ascend` | Compiled inference sources inherited from the base; not rebuilt |
| `workspace` | Shared compiled native operators inherited from the base |
| `manifest` | Source inventories, patches, package versions, recipe revision and inherited base receipts |

All four training dependencies are installed editable with `--no-deps` and
`--no-build-isolation`. This preserves the base image's tested Python packages
and avoids installing CUDA dependencies from upstream package metadata. No new
Python dependency lock is claimed: the base image determines the inherited
package versions, and the build verifies they do not change. Select a versioned
base image or digest when publishing a release.

MindSpeed-LLM and mbridge belong to the V4/V4.1 training stack; their editable
installations and source directories are removed from this K26 image. The base
image itself remains available under its original tag. Shared native operator
paths are retained so installed libraries continue to resolve correctly.
The image is a K26 text-training environment; it does not enable multimodal actor
training or FP8 training.

## Verification and use

The build checks Python 3.11.15, hardware/data package versions, inherited rollout
source pins and source-file hashes, the complete K26 source inventories, editable
installation paths and adapted Python syntax. It replaces the inherited recipe
snapshot, so deleted files cannot survive by being merged through `COPY`.
The Docker context excludes development checkouts, model/data files, caches,
credentials and local environment inventories.

Start the image with the same Ascend device mappings, driver-library mounts,
shared-memory settings and shared model/data storage used for the validated job.
The entrypoint initializes CANN/ATB and selects the K26 source paths.
`source tools/activate.sh k26` additionally runs the same import/interface preflight
used by the launchers. No source
clone or patch application is needed inside the container.

```bash
cd /opt/ascend-training/Ascend-Training-Recipes

# Uses the supplied BF16 checkpoint, preparing a separate two-layer crop if needed.
FULL_MODEL_PATH=/models/Kimi-K2.6-bf16 \
MODEL_PATH=/models/Kimi-K2.6-2layers \
DATA_ROOT=/data \
OUTPUT_DIR=/outputs LOG_DIR=/logs \
EXP_NAME=kimi_k26_mopd_image_smoke \
bash recipes/kimi-k2.6/kimi_k26_mopd_16npu_cutlayer.sh
```

The default MOPD smoke test uses 8 student NPUs plus two TP4 teachers on another
8 NPUs, with TP4/PP1/CP2/EP8/ETP1 on the actor. It takes 16 training prompts,
batch size 8 and two steps in one epoch. MATH and GSM8K rewards share the same
`score`, `acc`, `pred` fields. Set the data paths and shared HF cache as documented
in the [K26 recipe](../../recipes/kimi-k2.6/README.md). Use a fresh experiment/output
path for the image acceptance run to avoid resuming an earlier smoke checkpoint.

Before publishing, check two optimizer steps, teacher hidden-state coverage,
checkpoint save and rollout weight updates in the rebuilt image. Checkpoint
resume is a separate acceptance test. Build-time checks do not run NPU kernels.
