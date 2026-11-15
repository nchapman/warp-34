# syntax=docker/dockerfile:1

# The frontend tool versions match .mise.toml. Only dist crosses into Rust;
# neither Node nor pnpm is present in the runtime image.
FROM node:24.21.0-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20 AS dashboard
RUN npm install --global pnpm@12.10.1
WORKDIR /dashboard
COPY dashboard/package.json dashboard/pnpm-lock.yaml ./
RUN --mount=type=cache,id=warp34-pnpm,target=/pnpm/store,sharing=locked \
    pnpm install --frozen-lockfile --store-dir=/pnpm/store
COPY dashboard/ ./
RUN pnpm build

# Keep the toolkit/cuBLAS version aligned with the proven warp3090 image.
FROM nvidia/cuda:13.4.1-devel-ubuntu26.04@sha256:fe678162c7114158e170f2f727a7582162a3a55d7090ed31fac04f2dbc173378 AS build
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH \
    # Link against the driver stub without requiring a GPU during the build.
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake nasm curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.0
WORKDIR /src
# Frontend sources are not Rust build inputs; only the built assets are embedded.
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ ./.cargo/
COPY crates/ ./crates/
COPY --from=dashboard /dashboard/dist ./dashboard/dist
# Bound nvcc parallelism on smaller build hosts; kernels build as an sm_80 + sm_86 + sm_89 fatbinary by default.
ARG WARP34_NVCC_JOBS=4
ARG WARP34_ARCH=sm_80,sm_86,sm_89
# Compress the embedded device code, not the executable with a runtime packer.
# Keep native SASS for both GPU generations and the existing PTX fallback.
ARG WARP34_NVCC_EXTRA=--compress-mode=size
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    WARP34_ARCH=${WARP34_ARCH} WARP34_NVCC_JOBS=${WARP34_NVCC_JOBS} WARP34_NVCC_EXTRA=${WARP34_NVCC_EXTRA} \
    cargo build --locked --release --features cuda --bin warp34 \
    && cp target/release/warp34 /warp34

# Do not ship the multi-GB CUDA runtime image, toolkit, or driver stubs.
# The host driver is injected by NVIDIA Container Toolkit with --gpus.
FROM ubuntu:26.04@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /home/warp34 warp34 \
    && install -d -o warp34 -g warp34 /home/warp34/.cache/huggingface/hub
# Keep CUDA libraries in a separate layer so app updates can reuse the download.
# COPY dereferences SONAME symlinks; also copying versioned files duplicates them.
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13 \
    /usr/local/cuda/lib64/
RUN echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf && ldconfig
COPY --from=build /warp34 /usr/local/bin/warp34
ENV HOME=/home/warp34 \
    HF_HUB_CACHE=/home/warp34/.cache/huggingface/hub \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0"
USER warp34
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/usr/local/bin/warp34"]
# No implicit model load: without arguments, print CLI usage.
CMD []
