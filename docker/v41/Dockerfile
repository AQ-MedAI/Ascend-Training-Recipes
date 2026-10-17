# Use the bundled BuildKit Dockerfile frontend to avoid an extra Docker Hub pull.
# Candidate only: required public inputs are documented in README.md.
# CANN base from upstream verl Ascend Dockerfile; pin its digest for release.
ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-ubuntu22.04-py3.12
FROM ${BASE_IMAGE} AS common
ARG BASE_IMAGE
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV BASH_ENV=/dev/null \
    DEBIAN_FRONTEND=noninteractive \
    PIP_CONFIG_FILE=/dev/null \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL="" \
    PIP_CONSTRAINT="" \
    UV_INDEX_URL=https://pypi.org/simple \
    UV_EXTRA_INDEX_URL="" \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    GIT_TERMINAL_PROMPT=0
WORKDIR /opt/ascend-training

# Check the CANN development environment before installing the pinned Python stack.
RUN test "$(uname -m)" = aarch64 && \
    python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "This image requires Python 3.12"' && \
    test -f /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    test -f /usr/local/Ascend/nnal/atb/set_env.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates patch rsync file unzip zip tar gzip xz-utils \
        build-essential pkg-config libnuma-dev cmake && \
    rm -rf /var/lib/apt/lists/*
# Ubuntu's default python3-dev may target a different Python; require the base's matching headers.
RUN python3 -c 'import pathlib, sysconfig; assert (pathlib.Path(sysconfig.get_path("include")) / "Python.h").is_file(), "Python 3.12 development headers are required"'

# Version-pinned bootstrap, not a complete hashed environment lock.
COPY docker/v41/hardware-requirements.txt docker/v41/hardware-constraints.txt /tmp/ascend-image-build/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --only-binary=:all: \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        --constraint /tmp/ascend-image-build/hardware-constraints.txt \
        -r /tmp/ascend-image-build/hardware-requirements.txt && \
    python3 -c 'import importlib.metadata as m; expected={"torch":"2.10.0","torch-npu":"2.10.0.post4","triton-ascend":"3.2.2","triton":"3.5.0"}; actual={k:m.version(k).split("+",1)[0] for k in expected}; assert actual == expected, (expected, actual)'

# Shared V4/V4.1 framework layer; retain the target name for existing build commands.
FROM common AS v41

ARG SOC_VERSION=ascend910_9382
ARG MAX_JOBS=4
ARG DEPENDENCY_OFFLINE=0
ARG PYTHON_DEPENDENCY_MODE=resolve
ENV DEPENDENCY_ROOT=/opt/ascend-training \
    MEGATRON_ROOT=/opt/ascend-training/Megatron-LM \
    SOC_VERSION=${SOC_VERSION}
# Reconstruct all three sibling source trees from public commits and checked patches.
COPY dependencies/framework-sources.json /tmp/ascend-image-build/dependencies/framework-sources.json
COPY patches/ /tmp/ascend-image-build/patches/
COPY tools/prepare_framework_sources.py /tmp/ascend-image-build/tools/prepare_framework_sources.py
COPY docker/v41/Megatron-LM-LICENSE /tmp/ascend-image-build/
RUN python3 /tmp/ascend-image-build/tools/prepare_framework_sources.py \
    --root /opt/ascend-training --manifest /tmp/ascend-image-build/dependencies/framework-sources.json

# Only image-owned sources are built here; recipe dependencies/runtime are installed after cloning.
COPY dependencies/registry.json /tmp/ascend-image-build/registry.json
# A complete runtime lock is optional for development builds; fixed inputs are included.
COPY docker/v41/locks/ /tmp/ascend-image-build/locks/
COPY docker/v41/build-support/prepare_native_deps.py /opt/ascend-training/workspace/rollout/dsv41/
COPY docker/v41/build-support/prepare_mhc_premix.py docker/v41/build-support/build_mhc_premix.sh docker/v41/build-support/native_operator_env.sh /opt/ascend-training/workspace/tests/dsv41/
COPY docker/v41/build-runtime.sh docker/v41/collect_environment.py docker/v41/download_dependencies.py docker/v41/native-dependencies.json /tmp/ascend-image-build/
RUN --mount=type=cache,target=/root/.cache/ascend-training-recipes \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    PIP_CACHE_DIR=/root/.cache/pip UV_CACHE_DIR=/root/.cache/uv MAX_JOBS="${MAX_JOBS}" DEPENDENCY_OFFLINE="${DEPENDENCY_OFFLINE}" PYTHON_DEPENDENCY_MODE="${PYTHON_DEPENDENCY_MODE}" bash /tmp/ascend-image-build/build-runtime.sh

# Training and rollout share Python dependencies; native loader settings remain role-specific.
ENV VLLM_ROOT=/opt/ascend-training/vllm \
    VLLM_ASCEND_ROOT=/opt/ascend-training/vllm-ascend \
    V41_MHC_PRE_BACKEND=premix \
    V41_INDEXER_BACKEND=lightning \
    V41_RMSNORM_BACKEND=native \
    V41_ROPE_BACKWARD=explicit \
    V41_ROPE_CACHE=1 \
    V41_CSA2_METADATA_CACHE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn
COPY docker/v41/entrypoint.sh /usr/local/bin/v41-entrypoint
ENTRYPOINT ["/bin/bash", "/usr/local/bin/v41-entrypoint"]
WORKDIR /workspace/Ascend-Training-Recipes
CMD ["/bin/bash"]
