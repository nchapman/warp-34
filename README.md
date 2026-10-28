# warp-34

Clean-room Rust + CUDA inference engine for **Qwen3.5 hybrid models** on
RTX 3090 Ti (`sm_86`, 24 GB, 1008 GB/s), architected around **`vLLM`** from day
one.

## Core Design Principles

1. **Explicit Reference Donor per Subsystem ([`docs/ROADMAP.md`](docs/ROADMAP.md)):**
   - **`vLLM 0.31.0` (`8352b2427`)**: Primary donor for the `FP16` model spine, Marlin GEMMs (`W4A16` / `W8A16` / `W4A8` / `moe_wna16`), layer fusions (`GemmaRMSNorm`, `fused_qk_rmsnorm_rope_gate`, GDN `chunk` + `fused_recurrent`), native `MTP`, 9-step GPU sampler, and Qwen3 reasoning/tool/vision parsers.
   - **`SGLang v0.5.19` (`0bcd822377`)**: Primary donor for **Hybrid Prefix Caching (`MambaRadixCache` + `HybridReqToTokenPool` `extra_buffer` strategy)** and **`DFlash2` (`DFlashWorkerV2` `M=8` speculative verify)** — the two areas where `vLLM 0.31.0` cannot combine hybrid prefix caching with draft models or breaks on `TP=2` (`tau=1.000`).
   - **`warp3090` (`sm_86` proven IP)**: Primary donor for **TurboQuant 4-bit KV (`TQ4`)** (`block_tq4_0` 4.125 bpw, Rademacher+WHT-256 rotation, Beta(256) Lloyd-Max codebook, `FirstLast2` `FP16` boundary layers, fused `fattn_mma_tq` + GQA-8 `fattn_vec_tq`) and **Pinned-Sysmem PCIe Gen4 x8 `TP=2` AllReduce** (`~18 µs`).
2. **End-to-end `f16` activation spine:** Hidden states, residuals, norms, RoPE,
   attention, GDN, and Marlin GEMMs consume and produce `f16` directly — zero
   `f32 ↔ f16` adapter casts between ops.
3. **Continuous Oracle & `nsys` Scoreboard from Phase 0:** Every op and phase is
   gated by `w34-oracle` (`dump-diff` tensor parity, `greedy-diff` token parity,
   and `profile-diff` side-by-side `nsys` kernel timing against `vLLM` / `SGLang`).

## Workspace Layout

```
crates/
  w34-kernels/   CUDA kernels + C/Rust FFI seam (built incrementally via nvcc)
  w34-host/      Rust runtime, safetensors loader, scheduler, CLI, and server
docs/
  ROADMAP.md     Phased engineering roadmap & vLLM / SGLang / warp3090 source map
```

## Quick Commands

```sh
export PATH="/opt/cuda/bin:$PATH"
mise run build          # cargo build --release --features cuda
mise run check-quick    # fmt --check + clippy (with and without cuda)
mise run check          # fmt + clippy + release test suite
```
