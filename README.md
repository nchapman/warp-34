# warp-34

Clean-room Rust + CUDA inference engine for **Qwen3.5 hybrid models** on
RTX 3090 Ti (`sm_86`, 24 GB, 1008 GB/s), architected around **`vLLM`** from day
one.

## Core Design Principles

1. **Single-source `vLLM` alignment:** Kernel layouts, fusion boundaries, and
   scheduling semantics follow `vLLM` (`8352b2427` / `0.31.0`) and `vllm.cpp`
   (`50d425bec`) unless a measured end-to-end win proves otherwise.
2. **End-to-end `f16` activation spine:** Hidden states, residuals, norms, RoPE,
   attention, GDN, and Marlin GEMMs consume and produce `f16` directly — zero
   `f32 ↔ f16` adapter casts between ops.
3. **One unified forward path:** Single-GPU, `TP=2`, and speculative decode
   share one layer execution spine and weight residency policy.

## Workspace Layout

```
crates/
  w34-kernels/   CUDA kernels + C/Rust FFI seam (built incrementally via nvcc)
  w34-host/      Rust runtime, safetensors loader, scheduler, CLI, and server
```

## Quick Commands

```sh
export PATH="/opt/cuda/bin:$PATH"
mise run build          # cargo build --release --features cuda
mise run check-quick    # fmt --check + clippy (with and without cuda)
mise run check          # fmt + clippy + release test suite
```
