# warp-34 — Project Agent Guide

> **Posture: beat `vLLM` and `SGLang` on 1–2× RTX 30-series (`sm_86`) and 40-series (`sm_89`) GPUs (primary reference targets: RTX 3090 / 3090 Ti and RTX 4090).** Clean-room Rust + CUDA inference engine for Qwen 3.5 / 3.6 / 3.8 hybrid models (GDN + full attention + MoE + Vision + MTP/DFlash2 + Paged TQ4 KV).

## Project

- **Layout:** CUDA kernels in `crates/w34-kernels/csrc/`, FFI seam in `crates/w34-kernels/src/lib.rs`, Rust runtime/CLI/HTTP server in `crates/w34-host/`.
- **Explicit Donor per Subsystem (`docs/ROADMAP.md` §2, pins in `BASELINE.txt`):**
  - **`vLLM 0.31.0` (`8352b2427`)**: Installed at `~/.local/share/mise/installs/python/3.12.12/lib/python3.12/site-packages/vllm/` (sparse tree at `/home/nchapman/Code/warp3090/vendor/vllm`). Donor for `FP16` model spine, Marlin GEMMs (`W4A16`/`W8A16`/`W4A8`/`moe_wna16`), `GemmaRMSNorm`, `fused_qk_rmsnorm_rope_gate`, `Qwen3_5MTP`, 9-step GPU sampler, and `Qwen3Parser` (reasoning/tools/vision).
  - **`vllm.cpp` (`50d425bec`)**: `/home/nchapman/Code/warp3090/vendor/vllm.cpp`. Donor for native CUDA C++ GDN kernels (`src/vt/cuda/cuda_gdn.cu`), verified against `vLLM` FLA golden dumps.
  - **`SGLang v0.5.19` (`0bcd822377`)**: `/home/nchapman/Code/warp3090-dflash/vendor/sglang` & Docker `lmsysorg/sglang:v0.5.19`. Donor for `MambaRadixCache` + `HybridReqToTokenPool` (`extra_buffer` ping-pong GDN checkpoint donation) and `DFlashWorkerV2` (`M=8` verify on 27B, `M=16` on 4B).
  - **`warp3090` (`/home/nchapman/Code/warp3090`)**: Donor for TurboQuant 4-bit KV (`TQ4`: `block_tq4_0` 132B/row, Rademacher+WHT-256, Beta(256) codebook, rotated `Q8_0` boundaries + LSQ `TQ4` interior `3.07×`, `fattn_vec_tq` + `fattn_mma_tq`) and pinned-sysmem / NVLink P2P `TP=2` & `TP=4` collectives (`op_collectives.cu`).
- **Key Docs:** `docs/STATE.md` (current status & verified benchmarks), `docs/ROADMAP.md` (architecture & donor reference map, seam contracts, & core invariant tests), `docs/HARDWARE.md` (30-series `sm_86` & 40-series `sm_89` rooflines & ISA traps).

## Commands

Prefix GPU commands with `export PATH="/opt/cuda/bin:$PATH"` and select free card(s) via `CUDA_VISIBLE_DEVICES`.

```sh
nvidia-smi --query-compute-apps=pid,used_memory --format=csv              # check GPU squatters first
mise run test-prep                                                        # verify/download pinned test models & oracle fixtures
mise run build                                                            # cargo build --release --features cuda
mise run check-quick                                                      # fmt --check + clippy (with and without cuda)
mise run check                                                            # fmt + clippy (both states) + release test suite
mise run test-quick                                                       # cargo test --release --features cuda --lib
mise run test-cpu                                                         # cargo test --lib (fast CPU unit tests)
mise run coverage                                                         # cargo llvm-cov --features cuda --lib --summary-only
cargo test --release --features cuda --test <target> -- --test-threads=1 --exact <test>
for p in $(pgrep -x w34); do kill "$p"; done                              # stop w34 processes safely
```

## Conventions

- **Session ritual:** At start, read `docs/STATE.md` and `docs/ROADMAP.md`, check `git status -sb` / `git log --oneline -5`, and check `nvidia-smi` for GPU squatters. At end, update `docs/STATE.md`.
- **Hardware traps (`docs/HARDWARE.md`):** Kernels build as a multi-arch fatbinary (`sm_86,sm_89` + PTX by default; override via `W34_ARCH`) and query cached per-device properties (`CudaDeviceProps` / `w34_device_props`) at runtime so all 30-series (`28–84` SMs, `3–6 MiB` L2) and 40-series (`24–142` SMs, `24–96 MiB` L2) GPUs work without hardcoded `84`-SM or device-`0` assumptions. Primary targets are **1–2× RTX 3090 / 3090 Ti (`sm_86`, 82–84 SMs, 6 MB L2, 936–1008 GB/s)** and **1–2× RTX 4090 (`sm_89`, 128 SMs, 72 MB L2, 1008 GB/s, 165 TF F16-mma FP32-acc / 330 TF FP16-acc / 661 TOPS INT8 IMMA)**. Cross-GPU `TP=2` uses pinned-sysmem two-hop with system-scope fences (`kMaxBlocks = 144`, clamped to `min(peer_sm_cap, cap)`). GeForce FP32-accumulate tensor math is half-rate; INT32 is half-rate and shares issue with FP32. Shared `sm_86`/`sm_89` ceiling: `cp.async`, `ldmatrix`, dense `mma.sync`, `mbarrier`; no TMA, `wgmma`, `stmatrix`, clusters, or `mma.sp` (CUDA 13.4 `ptxas` rejects `sm_86a`).
- **Foundational Seam Invariants (`docs/ROADMAP.md` §1.5):**
  - **Precision policy:** Target model spine is end-to-end `FP16` (`__half`); `DFlash2` drafters preserve native `BF16` (`__nv_bfloat16`) where required (4B drafter overflows `FP16`); persistent GDN `ssm_state` (`72 MiB/rank` per 27B `TP=2` snapshot) and reductions are `FP32` in global memory.
  - **KV addressing:** Logical `page_id`s are decoupled from physical storage via per-layer `KvLayerLayout` descriptors (`fp16`, `q8_0` `272 B` rows, `tq4` `3.07×` rotated `Q8_0` boundaries + `132 B` LSQ `TQ4` interior, `tq4_all` `3.88×`, and `DFlashDraftContext` namespaces).
  - **Cold-start `TP=2` init:** Always register new kernels in `prepare_execution_plan()` so both ranks warm all specializations and `cudaDeviceSynchronize` before collective warmup or graph capture (preventing CUDA 13 lazy-loading deadlocks).
  - **Continuous `w34` oracle verification:** Every op and subsystem must pass `dump-diff` (tensor parity), `greedy-diff` (token parity), and `profile-diff` (`nsys` SQLite category `ms/round` vs `vLLM`/`SGLang`).

## Boundaries

- **Never run `pkill -f`** — it self-matches and kills the agent harness; kill `w34` by PID loop (`pgrep -x w34`).
- **Never trust a featureless `cargo test` or `cargo build`** — GPU tests require `--features cuda`.
- **Never reintroduce a `llama.cpp` `f32` activation spine or `f32 ↔ f16` glue casts** between `FP16` target layers.
- **Never transport token IDs or global-selection counters through `FP16` collectives** — use typed `INT32` / `FP32` collective entrypoints.
- **Never commit temporary debug printfs or experiment env vars** before stripping them.
