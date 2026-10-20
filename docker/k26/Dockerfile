# Reuse the compiled Python 3.11 V4/V4.1 runtime; replace the training sources for K26.
ARG BASE_IMAGE=ascend-training-v41:dev
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG FRAMEWORK_SOURCE_TRANSPORT=archive
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/ascend-training
ENV BASH_ENV=/dev/null \
    PIP_CONFIG_FILE=/dev/null \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL="" \
    PIP_CONSTRAINT="" \
    UV_INDEX_URL=https://pypi.org/simple \
    UV_EXTRA_INDEX_URL="" \
    DEPENDENCY_ROOT=/opt/ascend-training \
    RECIPE_ROOT=/opt/ascend-training/Ascend-Training-Recipes \
    RECIPE_PROFILE=k26 \
    TRAINING_PROFILE=k26 \
    PYTHON_BIN=/usr/local/python3.11.15/bin/python3 \
    ROLLOUT_PYTHON=/usr/local/python3.11.15/bin/python3 \
    MINDSPEED_ROOT=/opt/ascend-training/third_party/MindSpeed \
    MEGATRON_ROOT=/opt/ascend-training/third_party/Megatron-LM \
    MEGATRON_BRIDGE_ROOT=/opt/ascend-training/third_party/Megatron-Bridge \
    VERL_ROOT=/opt/ascend-training/third_party/verl \
    VLLM_ROOT=/opt/ascend-training/third_party/vllm \
    VLLM_ASCEND_ROOT=/opt/ascend-training/third_party/vllm-ascend \
    ASCEND_CHECKPOINT_WRITER=inprocess
# Keep an explicit interpreter even if the base image was built with another PATH.
ENV PATH="/usr/local/python3.11.15/bin:${PATH}" \
    PYTHONPATH=""
RUN test "$(uname -m)" = aarch64 && \
    "$PYTHON_BIN" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
COPY dependencies/framework-sources.json dependencies/verl-sources.json /tmp/k26-image-build/dependencies/
COPY patches/ /tmp/k26-image-build/patches/
COPY docker/k26/sources.json /tmp/k26-image-build/docker/k26/
COPY tools/prepare_framework_sources.py tools/prepare_k26_image_sources.py /tmp/k26-image-build/tools/
COPY docker/k26/build-runtime.sh docker/k26/verify_runtime.py docker/v41/collect_environment.py /tmp/k26-image-build/
RUN --mount=type=cache,target=/root/.cache/pip \
    FRAMEWORK_SOURCE_TRANSPORT="${FRAMEWORK_SOURCE_TRANSPORT}" \
    bash /tmp/k26-image-build/build-runtime.sh
# Verify the installed training sources and inherited package versions before bundling the recipe.
RUN "$PYTHON_BIN" /tmp/k26-image-build/verify_runtime.py

# Replace the inherited recipe snapshot before copying; removed files must not survive COPY.
RUN rm -rf /opt/ascend-training/Ascend-Training-Recipes
ARG RECIPE_REVISION=unspecified
COPY . /opt/ascend-training/Ascend-Training-Recipes/
RUN test ! -e "$RECIPE_ROOT/third_party" && test ! -e "$RECIPE_ROOT/.git" && \
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$RECIPE_ROOT/tools/record_recipe.py" \
        --root "$RECIPE_ROOT" --revision "$RECIPE_REVISION" \
        --output /opt/ascend-training/manifest/recipe.json && \
    bash -n "$RECIPE_ROOT/recipes/kimi-k2.6/runtime_env.sh" \
        "$RECIPE_ROOT/recipes/kimi-k2.6/kimi_k26_rl_8npu_cutlayer.sh" \
        "$RECIPE_ROOT/recipes/kimi-k2.6/kimi_k26_mopd_16npu_cutlayer.sh" && \
    rm -rf /tmp/k26-image-build
COPY docker/k26/entrypoint.sh /usr/local/bin/k26-entrypoint
LABEL org.opencontainers.image.revision="${RECIPE_REVISION}" \
    org.opencontainers.image.base.name="${BASE_IMAGE}"
ENTRYPOINT ["/bin/bash", "/usr/local/bin/k26-entrypoint"]
WORKDIR /opt/ascend-training/Ascend-Training-Recipes
CMD ["/bin/bash"]
