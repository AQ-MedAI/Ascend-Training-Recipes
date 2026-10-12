# syntax=docker/dockerfile:1
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
COPY docker/v41/hardware-requirements.txt docker/v41/hardware-constraints.txt /opt/v41-image/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --only-binary=:all: \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        --constraint /opt/v41-image/hardware-constraints.txt \
        -r /opt/v41-image/hardware-requirements.txt && \
    python3 -c 'import importlib.metadata as m; expected={"torch":"2.10.0","torch-npu":"2.10.0.post4","triton-ascend":"3.2.2","triton":"3.5.0"}; actual={k:m.version(k).split("+",1)[0] for k in expected}; assert actual == expected, (expected, actual)'

# V4 may reuse --target common; this framework layer is specific to V4.1.
FROM common AS v41

# This source must include the V4.1 optimizer/DDP adaptations, not just upstream HEAD.
ARG MEGATRON_REPOSITORY=https://github.com/NVIDIA/Megatron-LM
ARG MEGATRON_COMMIT=a845aa7e12b3a117e24c2352b9e3e60bad2e3a17
ARG SOC_VERSION=ascend910_9382
ARG MAX_JOBS=4
ARG DEPENDENCY_OFFLINE=0
ENV DEPENDENCY_ROOT=/opt/ascend-training \
    MEGATRON_ROOT=/opt/ascend-training/Megatron-LM \
    SOC_VERSION=${SOC_VERSION}
RUN [[ "$MEGATRON_REPOSITORY" == https://* ]] && \
    [[ "$MEGATRON_COMMIT" =~ ^[0-9a-f]{40}$ ]] && \
    git init Megatron-LM && \
    git -C Megatron-LM remote add origin "$MEGATRON_REPOSITORY" && \
    git -C Megatron-LM fetch --depth=1 origin "$MEGATRON_COMMIT" && \
    git -C Megatron-LM checkout --detach FETCH_HEAD && \
    test "$(git -C Megatron-LM rev-parse HEAD)" = "$MEGATRON_COMMIT"

# Reconstruct the reviewed V4.1 source tree from an upstream commit plus a local patch.
COPY docker/v41/megatron-v41.patch docker/v41/megatron-source.json docker/v41/Megatron-LM-LICENSE /opt/v41-image/
RUN python3 - <<'VERIFY_MEGATRON'
import hashlib
import json
import subprocess
from pathlib import Path
manifest = json.loads(Path('/opt/v41-image/megatron-source.json').read_text())
root = Path('/opt/ascend-training/Megatron-LM')
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == manifest['base_commit']
patch = Path('/opt/v41-image') / manifest['patch']
assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest['patch_sha256']
subprocess.run(['git', '-C', str(root), 'apply', '--check', str(patch)], check=True)
subprocess.run(['git', '-C', str(root), 'apply', str(patch)], check=True)
for name, expected in manifest['patched_files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
VERIFY_MEGATRON

# Only image-owned sources are built here; recipe forks/runtime are installed after cloning.
COPY dependencies/registry.json /opt/v41-image/registry.json
# Full Python dependency locks are pending; no external runtime source payload is required.
COPY docker/v41/locks/ /opt/v41-inputs/
COPY docker/v41/build-support/prepare_native_deps.py /opt/ascend-training/workspace/rollout/dsv41/
COPY docker/v41/build-support/prepare_mhc_premix.py docker/v41/build-support/build_mhc_premix.sh docker/v41/build-support/native_operator_env.sh /opt/ascend-training/workspace/tests/dsv41/
COPY docker/v41/build-runtime.sh docker/v41/collect_environment.py docker/v41/download_dependencies.py docker/v41/native-dependencies.json /opt/v41-image/
RUN --mount=type=cache,target=/root/.cache/ascend-training-recipes \
    MAX_JOBS="${MAX_JOBS}" DEPENDENCY_OFFLINE="${DEPENDENCY_OFFLINE}" bash /opt/v41-image/build-runtime.sh

# Training and rollout share Python dependencies; native loader settings remain role-specific.
ENV VLLM_ROOT=/opt/ascend-training/workspace/rollout/dsv41/build-src/vllm \
    VLLM_ASCEND_ROOT=/opt/ascend-training/workspace/rollout/dsv41/build-src/vllm-ascend \
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
