# Adaptation candidate: this builds the training-source layer, not a validated K26 runtime.
# Build the common base with docker/v41/Dockerfile --target common first.
ARG BASE_IMAGE=ascend-training-base:dev
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/ascend-training
ENV DEPENDENCY_ROOT=/opt/ascend-training \
    MEGATRON_ROOT=/opt/ascend-training/Megatron-LM \
    MEGATRON_BRIDGE_ROOT=/opt/ascend-training/Megatron-Bridge \
    TRAINING_PROFILE=k26
COPY docker/k26/sources.json /opt/ascend-training/manifest/k26-sources.json
COPY docker/k26/build-runtime.sh /tmp/k26-build-runtime.sh
RUN bash /tmp/k26-build-runtime.sh && rm /tmp/k26-build-runtime.sh
ENV PYTHONPATH=/opt/ascend-training/Megatron-Bridge/src:/opt/ascend-training/Megatron-LM
WORKDIR /workspace/Ascend-Training-Recipes
CMD ["/bin/bash"]
