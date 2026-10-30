# syntax=docker/dockerfile:1

# Keep the toolkit/cuBLAS version aligned with the proven warp3090 image.
FROM nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311 AS build
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH \
    # Link against the driver stub without requiring a GPU during the build.
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake nasm curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.0
WORKDIR /src
COPY . .
# Bound nvcc parallelism on smaller build hosts; kernels still target sm_86.
ARG W34_NVCC_JOBS=4
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/target \
    W34_NVCC_JOBS=${W34_NVCC_JOBS} cargo build --locked --release --features cuda --bin w34 \
    && cp target/release/w34 /w34

# Do not ship the multi-GB CUDA runtime image, toolkit, or driver stubs.
# The host driver is injected by NVIDIA Container Toolkit with --gpus.
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /home/w34 w34
# COPY dereferences these SONAME symlinks. Copying versioned paths as well
# would duplicate the large cuBLAS payloads in the final image.
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13 \
    /usr/local/cuda/lib64/
RUN echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf && ldconfig
COPY --from=build /w34 /usr/local/bin/w34
ENV HOME=/home/w34 \
    HF_HUB_CACHE=/home/w34/.cache/huggingface/hub \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER w34
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/usr/local/bin/w34"]
# No implicit model load: without arguments, print CLI usage.
CMD []
