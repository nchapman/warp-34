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
# Layer 1: Pre-build all third-party Rust & C++/CMake dependencies (cached until Cargo.lock changes).
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ ./.cargo/
COPY crates/warp34-kernels/Cargo.toml ./crates/warp34-kernels/Cargo.toml
COPY crates/warp34-host/Cargo.toml ./crates/warp34-host/Cargo.toml
RUN --mount=type=cache,id=warp34-cargo-registry,target=/root/.cargo/registry,sharing=locked \
    mkdir -p crates/warp34-kernels/src crates/warp34-host/src \
    && echo "fn main() {}" > crates/warp34-kernels/build.rs \
    && echo "" > crates/warp34-kernels/src/lib.rs \
    && echo "" > crates/warp34-host/src/lib.rs \
    && echo "fn main() {}" > crates/warp34-host/src/main.rs \
    && cargo build --locked --release --features cuda --bin warp34 \
    && rm -rf crates/warp34-kernels/build.rs crates/warp34-kernels/src \
        target/release/.fingerprint/warp34-* \
        target/release/deps/libwarp34_* \
        target/release/deps/warp34* \
        target/release/build/warp34-* \
        target/release/warp34

# Layer 2: Pre-build warp34-kernels sm_80 + sm_86 + sm_89 CUDA fatbinary (cached until crates/warp34-kernels changes).
# Compress the embedded device code, not the executable with a runtime packer.
ARG WARP34_NVCC_JOBS=8
ARG WARP34_ARCH=sm_80,sm_86,sm_89
ARG WARP34_NVCC_EXTRA=--compress-mode=size
ENV WARP34_ARCH=${WARP34_ARCH} \
    WARP34_NVCC_EXTRA=${WARP34_NVCC_EXTRA}
COPY crates/warp34-kernels/ ./crates/warp34-kernels/
RUN --mount=type=cache,id=warp34-cargo-registry,target=/root/.cargo/registry,sharing=locked \
    touch crates/warp34-kernels/build.rs crates/warp34-kernels/src/lib.rs \
    && WARP34_NVCC_JOBS=${WARP34_NVCC_JOBS} \
    cargo build --locked --release --features cuda -p warp34-kernels

# Layer 3: Build warp34-host with embedded dashboard assets and remove /src/target
# so the final layer only exports the linked /warp34 binary.
COPY crates/warp34-host/ ./crates/warp34-host/
COPY --from=dashboard /dashboard/dist ./dashboard/dist
RUN --mount=type=cache,id=warp34-cargo-registry,target=/root/.cargo/registry,sharing=locked \
    touch crates/warp34-host/src/lib.rs crates/warp34-host/src/main.rs \
    && cargo build --locked --release --features cuda --bin warp34 \
    && cp target/release/warp34 /warp34 \
    && rm -rf target

# Do not ship the multi-GB CUDA runtime image, toolkit, or driver stubs.
# The host driver is injected by NVIDIA Container Toolkit with --gpus.
FROM ubuntu:26.04@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /root/.cache/huggingface/hub
# Keep CUDA libraries in a separate layer so app updates can reuse the download.
# COPY dereferences SONAME symlinks; also copying versioned files duplicates them.
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13 \
    /usr/local/cuda/lib64/
RUN echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf && ldconfig
COPY --from=build /warp34 /usr/local/bin/warp34
COPY LICENSE NOTICE /usr/share/doc/warp34/
ENV WARP34_HOST=0.0.0.0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/usr/local/bin/warp34"]
# No implicit model load: without arguments, print CLI usage.
CMD []
