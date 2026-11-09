# RTX 30-Series (`sm_86`) & RTX 40-Series (`sm_89`) — Hardware Reference

Every number an optimization decision might depend on, each with its source.
`warp-34` builds a multi-architecture fatbinary (`sm_86` + `sm_89` + PTX) by
default and queries per-device hardware properties (`CudaDeviceProps` /
`w34_device_props`) at startup so every GeForce RTX 30-series (`sm_86`,
28–84 SMs, 3–6 MiB L2) and RTX 40-series (`sm_89`, 24–142 SMs, 24–96 MiB L2)
GPU runs natively without hardcoded SM or L2 constants. Primary optimization
references are **1–2× RTX 3090 / 3090 Ti (`sm_86`)** and **1–2× RTX 4090
(`sm_89`)**.

Sources are tagged inline:

- **[BOX]** queried on this machine (2× RTX 3090 Ti, driver 615.71.09, CUDA 13.4)
- **[VENDOR]** reference `llama.cpp` (`cb7934c5`) block structs / kernel dispatch for format internals
- **[WP]** NVIDIA Ampere GA102 GPU Architecture Whitepaper (v1/v2)
- **[ADA]** NVIDIA Ada Whitepaper Appendix A (official RTX 3090 Ti performance table)
- **[CC]** CUDA C++ Programming Guide, Compute Capabilities appendix (CUDA 13.x)
- **[BPG]** CUDA C++ Best Practices Guide §12.1 throughput table (CUDA 13.4)
- **[PTX]** PTX ISA 9.4 documentation
- **[ATG]** CUDA Ampere Tuning Guide
- **[TPU]** TechPowerUp GPU database
- **[PTXAS]** verified by assembling PTX with this box's `ptxas` (CUDA 13.4)

> **Primary `sm_86` reference machine:** `nvidia-smi` reports **2×
> GeForce RTX 3090 Ti** (GA102-350, fully-enabled die). All numbers in §§1–11
> are for the 3090 Ti. Differences vs RTX 3090 (`GA102-300`): 84 SMs (vs 82),
> 336 tensor cores (vs 328), 21 Gbps GDDR6X (`1008 GB/s` vs `936 GB/s`), boost
> 1860 MHz (vs 1695 MHz), 450W stock TDP — and **no NVLink** (removed on 3090 Ti;
> RTX 3090 retains 2-way NVLink). Both share `sm_86`.

---

## 1. Card identity

| Property | Value | Source |
|---|---|---|
| GPU / die | GA102-350, Samsung 8N, 28.3B transistors, 628 mm² | [TPU] |
| Compute capability | 8.6 (`sm_86`) | [BOX] |
| SMs | **84** | [BOX] |
| CUDA (FP32) cores | 10,752 (128/SM) | [WP][TPU] |
| Tensor cores | 336 (4/SM, 3rd gen) | [WP] |
| RT cores | 84 (2nd gen; irrelevant to us) | [TPU] |
| Base / boost clock | 1560 MHz reference (1670 MHz FE) / **1860 MHz** | [TPU][NV] |
| Max SM clock (driver cap) | 2100 MHz | [BOX] |
| Memory | 24 GB GDDR6X, 384-bit, 21 Gbps (mem clock 10501 MHz [BOX]) | [TPU][BOX] |
| Theoretical DRAM BW | **1008 GB/s** | [NV][TPU] |
| L2 cache | 6,291,456 B = **6 MB** | [BOX] |
| L1/shared per SM | 128 KB unified | [WP][CC] |
| Register file per SM | 64K × 32-bit (256 KB) | [WP][ATG] |
| TDP | 450W stock — ran power-capped at 275W during early perf research; **both cards at stock 450 W since 2026-10-03** (ops, commit `ac9b9cd`; §8 measured: the cap does not bind); pre-lift IMMA/power ceilings in perf notes are conservative | [TPU][BOX] |
| PCIe | Gen4 x16 card; **this box wires each card at x8** (2-GPU bifurcation) | [TPU][BOX] |
| GPU0↔GPU1 | `nvidia-smi topo -m` = **PHB** (same host bridge) — **P2P is disabled on this box**: `cudaDeviceCanAccessPeer` = 0 both directions, `cudaDeviceEnablePeerAccess` → "peer access is not supported" (AMD Ryzen root complex). Cross-GPU traffic uses pinned-sysmem two-hop when P2P is unavailable and 1-hop NVLink VRAM when P2P is available (`crates/w34-host/src/cuda/collectives.rs` & `crates/w34-kernels/csrc/ops/op_collectives.cu`); contract in `docs/contracts/collectives.md` | measured 2026-10-04; P2P probe 2026-10-14 |
| NVLink | **None.** 3090 Ti removed it (RTX 3090 had 2-way NVLink) | NVIDIA dev forums; note some vendor spec sheets still list "2-way NVLink" — template noise, the PCB has no NVLink fingers |
| ECC | No | [BOX][TPU] |
| Copy engines (`asyncEngineCount`) | 2 → H2D and D2H can overlap | [BOX] |

---

## 2. SM microarchitecture

GA10x SM [WP]:

- **4 processing blocks (partitions)**, each with one warp scheduler + one
  dispatch port: **each SM issues at most 4 warp instructions per clock** (1
  per scheduler) [CC].
- Each partition has **two 16-lane datapaths**: one FP32-only, one
  FP32+INT32 shared. Per SM per clock: **128 FP32 ops, or 64 FP32 + 64 INT32**
  — INT32 work (address math, predicates, masks) displaces FP32 issue, and
  pure-INT32 code runs at half the FP32 rate [WP].
- Per partition: 1 tensor core, 4 SFU, 4 LD/ST units.
- Arithmetic result latency ≈ 4 clocks; with the 4-scheduler model you need
  roughly **16 resident warps/SM issuing independently** to hide ALU
  latencies, more for memory [CC].
- Warp size 32; max 48 resident warps (1536 threads)/SM [CC][ATG].

---

## 3. Execution-unit throughput (per SM per clock, and card totals @ 1860 MHz)

Rates from [BPG] table ("8.6/8.9" column), [WP], and [CC]; card totals = rate
× 84 SMs × 1.86 GHz.

| Operation | ops/clk/SM | Card total | Notes |
|---|---|---|---|
| FP32 add/mul/mad | 128 | 40.0 TFLOPS | 2× sm_80 (A100 has 64) [ATG] |
| FP16 `f16x2` add/mul/mad | 64 | 40.0 TFLOPS (½-pairs) | **Half A100's rate**; FLOPs equal FP32 → CUDA-core FP16 buys bandwidth, not math [BPG] |
| FP16/BF16 non-tensor TFLOPS | — | 40.0 | equals FP32 [ADA] |
| INT32 add/mul/shift/logical | 64 | 20.0 TOPS | shared datapath w/ FP32 [WP][BPG] |
| 64-bit integer add | 32 | — | [BPG] |
| FP64 | 2 | 0.625 TFLOPS | FP32:FP64 = **64:1** on 8.6 [CC][BPG] |
| SFU: rcp/rsqrt/lg2/ex2/sin/cos | 16 | — | `tanh.approx.f32`-class ops; use `__expf`-style intrinsics [BPG] |
| `clz`, `popc` | 16 | — | [BPG] |
| cvt 8/16-bit int ↔ 32-bit | 64 | — | [BPG] |
| cvt f32 ↔ f16 | 64 | — | [BPG] |
| cvt involving int ↔ f32 | 64 | — | [BPG] |
| cvt involving f64 | 2 | — | [BPG] |
| `shfl.sync` | 32 | — | [BPG] |
| DP4A/DP2A | 64 | ≈80 TOPS int8-equiv | CUDA 12.x guide table (row dropped from CUDA 13 table) |
| Shared-memory access | 128 B/clk | ~20 TB/s | 32 banks × 4 B; bank conflicts serialize; broadcast free [CC] |
| `__syncthreads()` | 16 ops/clk | — | [BPG] |

---

## 4. Tensor cores (3rd gen, 4 per SM)

Official 3090 Ti numbers at 1860 MHz boost — dense / with 2:4 sparsity [ADA]:

| Mode | TFLOPS/TOPS | FMA per SM per clock |
|---|---|---|
| FP16 in, **FP16 acc** | **160 / 320** | 512 |
| FP16 in, **FP32 acc** | **80 / 160** | 256 |
| BF16 in, FP32 acc | 80 / 160 | 256 |
| TF32 | 40 / 80 | 128 |
| INT8 (s8/u8) | 320 / 640 | 1024 |
| INT4 (s4/u4) | 640 / 1280 | 2048 |
| FP64 tensor | **not supported** (GA10x has no FP64 tensor input; A100 only) | — |
| FP8 | not supported (sm_89+) | — |

Decisions these numbers force:

1. **GeForce nerf: FP32 accumulate is HALF rate.** On GA10x, FP16×FP16 with
   FP32 accumulation runs at 80 TF while FP16 accumulation runs at the full
   160 TF [ADA][WP]. A100 runs both at full rate. llama.cpp's MMA kernels
   (and ours, ported) accumulate in FP32 → our dense-F16 GEMM ceiling is
   **80 TF**, not 160. FP16 accumulation is a potential 2× prefill win if
   accuracy testing ever justifies it.
2. **TF32 is ¼ rate** (40 TF = same as CUDA-core FP32, but no bandwidth
   savings) — never worth using on this chip [ADA; NVIDIA dev forums confirm
   3090 TF32 = 35.6 dense = ¼ of its 142.7 FP16-acc rate].
3. **INT8 IMMA is 4× the FP32-acc FP16 rate** (320 vs 80): for Q4_K weights,
   dequant→s8 + `mma.m16n8k32.s8` is the fast lane (llama.cpp MMQ's approach).
4. Marketing "Tensor performance" numbers NVIDIA publishes for 30-series
   (e.g. "285 TFLOPS" for 3090, "320" for 3090 Ti) are **sparse FP16-acc**
   — divide by 2, and by 2 again for FP32-acc, when rooflining.

---

## 5. Memory hierarchy

| Level | Size / rate | Notes |
|---|---|---|
| DRAM | 24 GB @ **1008 GB/s** theoretical | **Measured: 959.5 GB/s read stream (95.2%)** — see §11 |
| L2 | 6 MB (6,291,456 B) [BOX] | Residency control available on 8.0+ [CC]: `cudaStreamSetAttribute` access-policy window; max window 128 MB [BOX]; **max persisting set-aside 4,325,376 B ≈ 4.125 MB** [BOX] |
| L1 / shared | 128 KB unified per SM | carveout options 0/8/16/32/64/100 KB shared [CC] |
| Shared per block | 48 KB static; **99 KB (101,376 B) with dynamic-smem opt-in** (`cudaFuncAttributeMaxDynamicSharedMemorySize`) | driver reserves 1 KB/block → ≤1 block/SM at full 99 KB [CC][ATG][BOX] |
| Shared banks | 32 banks × 4 B, 128 B/clk/SM | broadcast/multicast conflict-free; 64-bit accesses split | [CC] |
| Constant | 64 KB total, 8 KB/SM cache | uniform (same-address) reads broadcast, divergent reads serialize [CC] |
| Local (spill) | 512 KB/thread | spills go to L1/L2/DRAM at 128 B granularity — keep register pressure ≤ 255/thread instead |
| L2 fetch granularity | `cudaLimitMaxL2FetchGranularity` 32/64/128 B | useful for KV/streaming reads |
| PTX cache hints | `ld.global.nc` (read-only/`__ldg` path), `evict_first/last/unchanged` (sm_70+), `.L2::128B/.L2::256B` prefetch (sm_80+), `createpolicy`+`.L2::cache_hint` (sm_80+) | [PTX] |

---

## 6. Occupancy & limits (all [BOX] + [CC])

| Limit | Value |
|---|---|
| Threads per SM | 1,536 |
| Warps per SM | 48 |
| Blocks per SM | 16 |
| Threads per block | 1,024 |
| Registers per SM / per thread | 65,536 / 255 |
| Shared per SM / per block (opt-in) | 102,400 B / 101,376 B |
| Grid x / y,z | 2³¹−1 / 65,535 |
| Concurrent resident grids | 128 [CC] |
| Streams | stream priorities supported [BOX] |

Worked occupancy anchors:

- Full 1,536-thread occupancy with 128-thread blocks → 12 blocks × ≤40 regs
  (65536/1536 = 42.7); with 256-thread blocks → 6 blocks, same register cap.
- MMA tile kernels using ~99 KB dynamic smem → **1 block/SM**; pair with
  ≥512 threads/block for ≥16 warps to cover cp.async + HMMA issue.
- 255-reg kernels → 8 warps/SM (1×256-thread block) ≈ 17% occupancy; only
  viable with heavy ILP.

---

## 7. Instruction / feature availability on sm_86

Verified against PTX ISA 9.4 [PTX] and, where marked, by assembling with
this box's `ptxas` [PTXAS].

| Feature | sm_86 | Notes |
|---|---|---|
| `mma.sync` dense: f16 m16n8k16, bf16 m16n8k8/k16, tf32 m16n8k4/k8, s8 m16n8k16/k32, s4 m16n8k32/k64 | ✅ (all sm_80+) | [PTX][PTXAS] |
| `ldmatrix` (incl. `.trans`, b16) | ✅ sm_75+ | [PTX][PTXAS] |
| `cp.async` (global→shared, 4/8/16 B) + commit/wait groups | ✅ sm_80+ | [PTX]; LDGSTS |
| `mbarrier` init/arrive/wait/`test_wait` | ✅ sm_80+ | [PTX][PTXAS] |
| `mbarrier.try_wait` | ❌ sm_90+ | poll `test_wait.parity` instead [PTX] |
| `mbarrier.expect_tx` | ❌ sm_90+ (TMA companion) | [PTX] |
| `redux.sync` (u32/s32 add/min/max/and/or/xor) | ✅ sm_80+ | f32/f16 variants are sm_100a+ [PTX][PTXAS] |
| `mma.sp` (2:4 sparse) | ⛔ requires arch-specific target `sm_86a` | **CUDA 13.4 `ptxas` rejects `sm_86a` entirely** (`Value 'sm_86a' is not defined`) → sparsity is unreachable with our toolchain [PTXAS]. Irrelevant for dense LLM weights. |
| `stmatrix` | ❌ sm_90+ | "Feature 'stmatrix' requires .target sm_90" [PTXAS] |
| `wgmma`, `cp.async.bulk(.tensor)` (TMA), thread-block clusters, DSMEM, async mbarrier tx | ❌ sm_90+ | [PTX][CCCL docs] |
| FP8 (`e4m3`/`e5m2`) mma | ❌ sm_89+ | [PTX] |
| `tanh.approx.f16/f16x2` | ✅ sm_75+ | bf16 variant is sm_90+ [PTX] — GELU/tanh activations |
| L2 cache hints (`.L2::256B`, `cache_hint`, eviction priority) | ✅ sm_80+ | [PTX] |
| `nanosleep`, async barriers (`barrier.cluster` aside) | ✅ | — |
| **Compile target** | `-arch=sm_86` | use `sm_86`, not `sm_80` binaries: 8.0 code runs but leaves the doubled FP32 path on the table [ATG] |

Practical takeaway (matches llama.cpp's design): sm_86's ceiling for GEMM is
**manually pipelined `cp.async` + `ldmatrix` + `mma.sync` with double-buffered
smem and `test_wait`-style barriers** — Hopper niceties don't exist here.

---

## 8. Clocks & power reality on this box

[BOX] Both cards:

- Power limit **450 W (stock) since 2026-10-03** (was 275 W during initial
  hardware bringup — see §11).
- **Measured (2026-10-03, GPU 1, synthetic loads): the cap does not bind.**
  DRAM streaming held 1950 MHz at ~305 W; pure-mma tensor loads held
  1950–1965 MHz at only 205–238 W. So §4 peak tables scale by 1950/1860
  (+4.8%) at these operating points, not down.
- Real kernels mix pipes + DRAM and may draw more than any synthetic —
  sustained clocks under actual GEMM/decode are verified via `w34 profile-diff`
  and `nsys` traces rather than assumed from synthetic loads.
- GDDR6X junction temps run hot; `nvidia-smi` doesn't expose memory
  temperature on this driver — watch for memory-clock throttling during long
  decode runs.

---

## 9. Roofline mapping to our workloads

`warp-34` executes Safetensors checkpoints in unified Marlin **`W4A16` / `W8A16`**
tile layout (`docs/MODELS.md`, `crates/w34-kernels/csrc/ops/op_marlin.cu`). The
GGUF `Q4_K_M` figures below are included as reference math for cross-engine
roofline comparisons (`benchmarks/perf/`). Bandwidth model uses the **measured
959.5 GB/s** read ceiling (§11; 95.2% of 1008 theory).

- **Decode (batch 1) is DRAM-bound**: per-token traffic ≈ all weights + KV
  row reads.
  - 9B `W4A16` / `Q4_K_M`: ~5.3–5.4 GB/token → ceiling ≈ 959.5/5.3 ≈ **181 tok/s**
    (weights only). Tensor cores idle.
  - 27B `W4A16` / `Q4_K_M` (production target): ~14.0–14.5 GB/token streamed →
    ceiling ≈ **66–68 tok/s** at measured BW (`warp-34` achieves `54.5 tok/s` end-to-end on 1× 3090 Ti).
  - 0.8B `W4A16` / `Q4_K_M`: ~508 MB/token → ceiling ≈ **1890 tok/s** theory-bound —
    but at ~30 kernels/layer the measured 1.36 µs pipelined launch
    overhead (§11) dominates long before that: **CUDA graphs / kernel
    fusion is the lever for the 0.8B** (graphs amortize to 0.58 µs/node).
- **Prefill is tensor-bound**. Ceilings: **83.8 TF measured sustained**
  (F16 mma, FP32-acc, 1950 MHz) / **337 TOPS measured** (s8 IMMA) / ~80
  TOPS (dp4a path). Arithmetic-intensity crossover vs measured DRAM:
  F16-TC 83.8e12/959.5e9 ≈ **87 FLOP/B**; IMMA ≈ **351 OP/B**.
- **L2 (6 MB) cannot hold a layer** (9B: ~190 MB/layer average) — decode
  bandwidth math on `sm_86` must assume DRAM for weights. The 4.125 MB persisting
  set-aside fits hot scratch activations or a hot KV slice (whereas on RTX 4090
  `sm_89`, the 72 MB L2 + 49.5 MB persisting window holds full activation matrices
  and scratch arenas — see §12).
- **KV cache per token** (full-attention layers):
  `2 × n_attn_layers × n_kv_heads × head_dim × bytes` (`docs/contracts/cache-layout.md`).
- **Two GPUs, no NVLink, no P2P** (§1): tensor-parallel sharding over both
  cards splits weights in half per rank; cross-card traffic goes over PCIe Gen4 x8
  via pinned sysmem (`docs/contracts/collectives.md`) in ~18–22 µs per all-reduce,
  delivering `211.7 tok/s` on 9B `TP=2` and `90.8 tok/s` on 27B `TP=2` (vs `vLLM` `57.2 tok/s`).
- Model load over PCIe Gen4 x8 ≈ 16 GB/s → 5.3 GB in ~0.35 s; `asyncEngineCount=2`
  lets H2D and D2H overlap [BOX].

---

## 10. Choosing quantization formats — hardware decision variables

A quant format prices three hardware resources: **(a)** bits per weight →
DRAM bytes per token (decode speed) and model size (capacity), **(b)** which
compute path the format feeds (prefill speed), **(c)** dequant ALU cost —
whether the decode kernel can actually saturate DRAM. Quality-vs-bpw is
model-dependent and priced elsewhere (perplexity runs / llama.cpp's tables);
this section only prices the speed/size side.

### (a) Bits per weight → decode speed and capacity

Exact bpw computed from the vendored block structs (`ggml-common.h`,
QK_K = 256, QK8_0 = 32, K_SCALE_SIZE = 12) [VENDOR]; `_M`/`_S` variants mix
in Q6_K/Q5_K rows, so use measured file sizes when exact [BOX: Q4_K_M 9B =
5.3 GB]. Weights-only decode ceiling at 850 GB/s (batch 1), for the 9B
parity model and the 27B production target:

| Format | bpw | 9B bytes | 9B tok/s | 27B bytes | 27B tok/s |
|---|---|---|---|---|---|
| F16/BF16 | 16 | 18.0 GB | ~47 | 54.0 GB | fits nowhere |
| Q8_0 | 8.5 | 9.6 GB | ~88 | 28.7 GB | ~30 (2 cards) |
| Q6_K | 6.5625 | 7.4 GB | ~114 | 22.1 GB | ~38 |
| Q5_K_M | ≈5.69 | 6.3 GB | ~130 | 19.2 GB | ~44 |
| Q4_K_M | ≈4.83 | 5.3 GB [BOX] | ~155 | 16.3 GB | **~52** |
| IQ4_XS | ≈4.25 | 4.8 GB | ~172 | 14.3 GB | ~59 |
| IQ3_S | 3.44 | 3.9 GB | ~208 | 11.6 GB | ~72 |
| IQ3_XXS | 3.06 | 3.4 GB | ~234 | 10.3 GB | ~81 |
| Q2_K | 2.63 | 2.9 GB | ~272 | 8.9 GB | ~94 |
| IQ2_S | 2.56 | 2.9 GB | ~279 | 8.6 GB | ~96 |
| IQ2_XS | 2.31 | 2.6 GB | ~309 | 7.8 GB | ~107 |
| IQ2_XXS | 2.06 | 2.3 GB | ~346 | 6.9 GB | ~120 |

All rows are loadable formats (the retired GGUF lane's Tier 1 + Tier 2
coverage — historical reference; the engine reads GPTQ g128 today). unsloth UD mixes land between rows by element share
(UD-IQ4_XS 27B: 14.25 GB ≈ 4.22 effective bpw). §10c's ALU caveat applies
below 3 bpw: grid-decode formats stream below the DRAM ceiling until
ncu says otherwise.

`tok/s ≈ achieved_BW × 8 / (params × bpw + KV bytes/token)`. F16 9B fits
24 GB but leaves thin KV headroom; quant buys it back linearly. **For 27B
the ladder hits a wall at the top**: Q4_K_M leaves ~6 GB for KV+context on
one card; Q6_K leaves ~1 GB (unusable); Q8_0 needs both cards; F16 27B fits
not even 2×24 GB. Q4_K_M is the best quality that keeps a usable KV budget
on one 24 GB card.

### (b) Format class → prefill compute path

- **Integer formats ≤ 8 bit** (every `Q*_0/Q*_1/Q*_K/IQ_*` in the vendored
  tree, plus MXFP4/NVFP4) have MMQ paths: dequant → s8 + IMMA → ceiling
  **320 TOPS** [VENDOR: `ggml-cuda/mmq.cu` type dispatch].
- **F16/BF16** → HMMA with FP32 accumulate → **80 TF**.
- Anything forced onto scalar pipes (dp4a / FFMA) → ~80 TOPS / 40 TF.

Prefill spread between formats is up to 4× and decided by path availability,
not bpw: pick formats with MMQ coverage for long-prompt work.

### (c) Dequant ALU budget at streaming rate — the hidden constraint

During batch-1 decode the tensor cores are idle and the scalar pipes are all
available for dequant: 128 FP32 + 64 INT32 lane-ops/clk/SM ≈ 27 Tops/s at
1860 MHz. Per byte streamed at 850 GB/s that is ≈ **32 lane-ops/byte**, i.e.
≈ 4 ops per weight at b bpw: Q8_0 ≈ 34, Q6_K ≈ 26, Q4_K ≈ 18, IQ2-class ≈ 11
ops/weight. K-quant dequant costs ~3–6 ops/weight (mask/shift + scale MAD,
scales amortized over 256-weight super-blocks) → ≥3× margin down to 4-bit.
**Sub-3-bit codebook formats (IQ2_*, Q2_K: grid lookups, extra shared loads
+ address math) approach the budget — verify with ncu before assuming a
2-bit model streams at DRAM speed.** Shared-memory staging is never the
limit (84 SM × 128 B/clk ≈ 18 TB/s ≫ DRAM).

### (d) Capacity ledger

24 GB = weights + KV (`2 × n_attn_layers × n_kv_heads × head_dim × bytes ×
ctx`) + activations + CUDA context (~0.5–1 GB). Two cards give 48 GB via
layer-split only (no NVLink, §1) — and layer-split streams each card's half
**sequentially**, so it buys capacity, not speed: two-card Q8_0 27B ≈
30 tok/s is slower than one-card Q4_K_M ≈ 52 tok/s.

## 11. Measured on this box (calibration 2026-10-03, GPU 1)

1. **DRAM bandwidth — MEASURED**: read stream **959.5 GB/s** (95.2% of
   theory; p10–p90 spread 0.04%), memcpy D2D 909 GB/s effective.
2. **Sustained clocks / TC peaks at 450 W — MEASURED**: cap does not bind
   for pure loads (DRAM stream 305 W, pure mma ≤ 238 W, both at 1950+ MHz).
   Sustained **83.8 TF** F16-TC FP32-acc and **337 TOPS** INT8 IMMA — 100%
   of §4 tables scaled to 1950 MHz.
3. **Kernel launch overhead — MEASURED**: 1.36 µs pipelined back-to-back,
   4.02 µs launch+sync round trip, **0.58 µs/node in CUDA graphs**, 2.05 µs
   1-node graph replay. Implication: at ~150 launches/token the 0.8B pays
   ~37% of its weight-bound budget in dispatch — graphs/fusion are
   first-class for 0.8B; host-side syncs are ruinous.
4. **P2P between the two cards — RESOLVED (disabled on AMD Ryzen PHB)**:
   `cudaDeviceCanAccessPeer` = 0 both directions; `cudaDeviceEnablePeerAccess`
   returns "peer access is not supported between these two devices"
   (`docs/contracts/collectives.md`). The TP seam runs pinned-sysmem two-hop on
   non-P2P PCIe and 1-hop NVLink VRAM when P2P is available.

Re-verification one-liner:

```sh
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current --format=csv
```

---

## 12. RTX 4090 (`AD102-300`, `sm_89`) & Full 30/40-Series Lineup

### 12.1 Primary Reference Comparison: RTX 3090 Ti (`sm_86`) vs RTX 4090 (`sm_89`)

| Property | RTX 3090 (`GA102-300`) | RTX 3090 Ti (`GA102-350`) | RTX 4090 (`AD102-300`) | Ratio (`4090 / 3090 Ti`) |
|---|---|---|---|---|
| Compute capability | `8.6` (`sm_86`) | `8.6` (`sm_86`) | `8.9` (`sm_89`) | — |
| SMs | **82** | **84** | **128** (of 144 on full AD102) | **1.524×** |
| CUDA (FP32) cores | 10,496 (128/SM) | 10,752 (128/SM) | 16,384 (128/SM) | 1.524× |
| Tensor cores | 328 (3rd gen, 4/SM) | 336 (3rd gen, 4/SM) | 512 (4th gen, 4/SM) | 1.524× |
| Base / boost clock | 1395 / 1695 MHz | 1560 / **1860 MHz** | 2235 / **2520 MHz** | **1.355×** |
| Memory | 24 GB GDDR6X, 19.5 Gbps | 24 GB GDDR6X, 21 Gbps | 24 GB GDDR6X, 21 Gbps | 1.00× |
| Theoretical DRAM BW | **936 GB/s** | **1008 GB/s** | **1008 GB/s** | **1.00×** |
| **L2 cache** | **6 MB** | **6 MB** (`6,291,456 B`) | **72 MB** (`75,497,472 B`) | **12.0×** |
| L1 / shared per SM | 128 KB unified | 128 KB unified | 128 KB unified | 1.00× |
| Max opt-in shared/block | 99 KB (`101,376 B`) | 99 KB (`101,376 B`) | 99 KB (`101,376 B`) | 1.00× |
| Register file per SM | 64K × 32-bit (256 KB) | 64K × 32-bit (256 KB) | 64K × 32-bit (256 KB) | 1.00× |
| Dense FP32 TFLOPS | 35.6 TF | 40.0 TF | **82.6 TF** | **2.065×** |
| Dense Tensor FP16/BF16 (FP32-acc) | 71.2 TF | **80.0 TF** | **165.2 TF** | **2.065×** |
| Dense Tensor FP16 (FP16-acc) | 142.4 TF | **160.0 TF** | **330.3 TF** | **2.065×** |
| Dense Tensor INT8 (`IMMA`) | 284.7 TOPS | **320.0 TOPS** | **660.6 TOPS** | **2.065×** |
| TDP | 350 W | 450 W | 450 W | 1.00× |

### 12.2 Architectural Implications of `sm_89` vs `sm_86`

1. **Identical HBM bandwidth (1008 GB/s), 2.065× compute & 12× L2 cache:**
   - Single-request `M=1` weight-streaming decode is HBM-bound on both 3090 Ti and 4090 (~1008 GB/s), but **prefill (`M >= 16`), multi-token verify (`M = 8..16` for `MTP` / `DFlash2`), batched serving (`N = 8..32`), and L2-resident operations** benefit directly from 4090's **2.065× tensor/ALU throughput** and **72 MB L2 cache**.
   - On 4090 (`l2_cache_bytes >= 32 MiB`), `op_marlin.cu` raises the wide-`N` (`N > 4096`) `M`-parallelism cap from `max_par = 16` (`M <= 1024`) to `max_par = 64` (`M <= 4096`) because the activation matrix `A` stays resident in the 72 MB L2 cache across all `N` tiles without re-reading `B` weights from HBM.
2. **Per-SM limits are identical between `sm_86` and `sm_89`:**
   - Both architectures have `128 KB` unified L1/shared per SM, `101,376 B` (`99 KB`) max opt-in dynamic shared memory per block, `65,536` 32-bit registers per SM (`255` max per thread), `48` max warps (`1536` threads) per SM, and `16` max blocks per SM.
   - All `sm_80+` PTX instructions used in `warp-34` (`cp.async`, `ldmatrix`, `mma.sync.aligned.m16n8k16`, `redux.sync`, `mbarrier`) assemble natively to both `sm_86` and `sm_89` with `0` register spills (`0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`).
3. **Full 30-Series & 40-Series Lineup (Runtime `CudaDeviceProps` Scaling):**
   - **30-Series (`sm_86`):** RTX 3060 (28 SMs, 3 MB L2), RTX 3060 Ti (38 SMs, 4 MB L2), RTX 3070 / 3070 Ti (46–48 SMs, 4 MB L2), RTX 3080 / 3080 Ti (68–80 SMs, 5–6 MB L2), RTX 3090 (82 SMs, 6 MB L2), RTX 3090 Ti / RTX A6000 (84 SMs, 6 MB L2).
   - **40-Series (`sm_89`):** RTX 4060 / 4060 Ti (24–34 SMs, 24–32 MB L2), RTX 4070 / Super / Ti / Ti Super (46–66 SMs, 36–48 MB L2), RTX 4080 / Super (76–80 SMs, 64 MB L2), RTX 4090 D (114 SMs, 72 MB L2), RTX 4090 (128 SMs, 72 MB L2), RTX 6000 Ada / L40S (142 SMs, 96 MB L2).
   - **Dynamic runtime adaptation:**
     - `w34_get_current_device_props` / `CudaDeviceProps` caches `compute_major`, `compute_minor`, `multi_processor_count`, `max_shared_mem_per_block_optin`, and `l2_cache_bytes` per device.
     - `op_collectives.cu` sizes `Signal` for `kMaxBlocks = 144` and clamps `tp2_block_cap` to `min(peer_sm_cap, cap)` so GPUs with `< 84` SMs (e.g. 82-SM RTX 3090, 80-SM RTX 4080 Super, 24-SM RTX 4060) never launch more spin-waiting CTAs than physical SMs, while 128-SM RTX 4090 and 142-SM L40S scale up to full SM count on large reductions.
     - `op_marlin.cu` and `model/rank.rs` size the Marlin tile-lock workspace to `sm_count * 64` ints, query the active device (`cudaGetDevice`), scale wide-`N` `max_par` continuously from `l2_cache_bytes / (384 KiB)` (`[8, 128]`), and keep `M <= 16` non-`enable_m_invariant_split_k` GEMMs on `blocks = mn_tiles` (1 slice/tile) so `< 64`-SM GPUs (`34`-SM 4060 Ti, `38`-SM 3060 Ti) preserve `TP=1 == TP=2` reduction invariance.
     - `op_gdn.cu` selects single-request recurrent prefill `ROWS` in `{4, 7, 8, 11}` by minimizing wave-quantized work `ceil(num_v_heads * v_tiles / sm_count) * ROWS` directly from the card's reported `sm_count` (independent of `batch->num_reqs` so multi-request batched prefill matches single-request prefill).
     - `tq4_kv_common.cuh` and `op_qsa.cu` keep `block_tq4_0` / `qsa_block_tq4_0` (`66 B`, `alignof == 2`) naturally 2-byte aligned without `#pragma pack(push, 1)` so `ptxas` emits aligned 16-bit `STG.E.U16` / `LDG.E.U16` scale accesses and avoids the CUDA 13.0 (`ptxas V13.0.88`) 1-byte `__half` store peephole bug (`F2I.U8.F16.TRUNC.NTZ`).
4. **Live 2× RTX 4090 (`sm_89`) Empirical Findings & Kernel Tuning (`Passes 1–5`):**
   - **Cold-DRAM vs L2-Resident Marlin Trap (`op_marlin.cu`):**
     - Single-layer Marlin microbenchmarks (`<= 64 MiB` weight footprint) fit entirely inside the 4090's **`72 MiB` L2 cache** (`~2.8 TB/s` L2 BW), making 128-CTA `{128, 128, 256}` Split-K appear `28–38%` faster on wide shapes (`N = 5120`).
     - In full-model inference (`4B` = `2.6 GB`, `9B` = `5.4 GB`, `27B` = `14.5 GB`), weights stream cold from GDDR6X (`1,008 GB/s`), where `80` CTAs of 1-slice `{128, 64, 128}` already saturate DRAM controllers without paying Split-K's global-memory partial-sum write/read traffic or spin-lock serialization.
     - Conversely, for narrow `N` (`N <= 3072`, where `mn_tiles = N / 64 <= 48` CTAs cannot saturate DRAM with 1 slice), 2-slice `enable_m_invariant_split_k` (`80..96` CTAs) genuinely increases achieved DRAM bandwidth and speeds up real inference (`9B TP=2` decode `219.6 -> 230.0 tok/s`, `+4.7%`). Keeping `N = 2560, K <= 4096` on 1-slice preserves `128/128 (100.0%)` exact `4B` golden token parity across 3090 Ti (`84` SMs) and 4090 (`128` SMs).
   - **Multi-CTA PCIe Pinned-Sysmem Collectives (`op_collectives.cu`):**
     - Replacing thread-0-only peer flag polling with single-writer `barrier_epoch_release` + multi-warp read, buffering `v_reg` in registers instead of `24.7 KiB` shared memory (`SHARED: 128B`, `STACK:0 LOCAL:0`), launching `512` threads for `H >= 4096`, software-pipelining `tx_stage(stage + 1)` before `wait_stage(stage)`, and raising `tp2_block_cap` to `<= 16` CTAs cut `TP=2` collective latency by `-9%` to `-14%` across `H = 2048..5120` (`M = 1..16`) on both 2× 4090 (`5.75 -> 5.20 us` at `M=1, H=5120`; `24.48 -> 21.84 us` at `M=16, H=5120`) and 2× 3090 Ti (`3090 Ti` `35B-A3B TP=2` decode `268.2 -> 287.2 tok/s`, `+7.1%`; `27B TP=2` decode `89.3 -> 90.8 tok/s`, `+1.7%`).
   - **35B-A3B Sparse MoE `M=1` Decode (`op_moe.cu`):**
     - Adding `moe_topk_router_256_k8_f16_kernel` (`8` warps × `32` lanes = `256` experts in registers with warp-shuffle top-8 reduction, `1.73 us` vs `vLLM`'s `16.2 us` 2-stage router), plus `__launch_bounds__(128, 8)` (`64` regs/thread -> `8` blocks/SM = `32` warps/SM occupancy, `STACK:0 LOCAL:0`) and bitshift `w >> group_shift` in `moe_w4a16_gate_up_silu_f16_kernel` and `moe_w4a16_down_accum_f16_kernel<false, 32|64, 8>`, reduced 4090 `nsys` per-step MoE time by **`-11.4%`** on `35B TP=1` (`970.4 -> 859.5 us/step`) and **`-21.5%`** on `35B TP=2` (`719.2 -> 564.3 us/step`), lifting 4090 `35B-A3B` decode to **`242.4 tok/s`** (`TP=1`, `+6.1%`), **`308.4 tok/s`** (`TP=2`, `+9.7%`), and **`507.1 tok/s`** (`TP=2 MTP k=2`, `+6.5%`), and 3090 Ti `35B-A3B` decode to **`213.6 tok/s`** (`TP=1`, `+4.9%`), **`287.2 tok/s`** (`TP=2`, `+7.1%`), and **`466.4 tok/s`** (`TP=2 MTP k=2`).
   - **Persisting L2 Cache Window (`op_init.cu`, `model/rank.rs`):**
     - On GPUs with `persisting_l2_max_bytes >= 16 MiB` (RTX 4090 reports `51,904,512 B` = `49.5 MiB`; RTX 3090 Ti reports `4,718,592 B` = `4.5 MiB`), `w34_stream_set_persisting_l2_window` pins a `16 MiB` persisting L2 window (`hitRatio = 1.0`, `hitProp = cudaAccessPropertyPersisting`, `missProp = cudaAccessPropertyNormal`) on `scratch_arena` so hot decode/verify ping-pong hidden states and QKV/MoE scratch buffers remain L2-resident across weight-streaming GEMMs without thrashing `M=512` prefill activations (`missProp = Streaming` with `36 MiB` degraded `35B` prefill `4706 -> 4417 tok/s`; `missProp = Normal` with `16 MiB` keeps `35B TP=2` prefill at `4868 tok/s`).
   - **HGPTQ MoE `APPLY_HAD64` Occupancy Trap & `Flash-Next` `n_interm=640` Specialization (`op_moe.cu`):**
     - Fusing `H_64` inside `moe_w4a16_gate_up_silu_f16_kernel<..., true>` (`M = 1..3`) and `moe_sparse_vec4_kernel<true, 64, true>` (`M = 4..7`) collapsed `grid.x` to `n_interm / 64` (`8` CTAs/pair = `64` CTAs on `35B TP=2`, `10` CTAs/pair = `~50` active CTAs on `Flash-Next TP=2` across `128` SMs), costing `44.64 µs` vs `6.69 µs` (`6.67×` slower) at `M = 1` and `48.85 µs` vs `22.35 µs` at `M = 4` (`83.14 µs` on `Flash-Next TP=2`). Running full-occupancy `APPLY_HAD64 = false` (`grid.x = n_interm / 4` or `n_interm / 16`) + L1/L2-resident `w34_block_hadamard_inplace_f16` (`~1.1 µs`) plus `<false, 80, 10>` (`n_interm = 640, top_k = 10`) lifted `35B-HGPTQ TP=2` decode `204.9 -> 293.1 tok/s` (`+43.1%`), `MTP-2` `436.3 -> 548.9 tok/s` (`+25.8%`), `MTP-3` `534.1 -> 601.1–614.0 tok/s`, and `Flash-Next-HGPTQ TP=2` decode `66.9 -> 90.1 tok/s` (`+34.7%`) / `MTP-3` `102.3 -> 118.7 tok/s` (`+16.0%`).
   - **PCIe Gen 4 x16 Mapped-Sysmem SM-Store Bottleneck (`2.36 GB/s`) vs Copy-Engine DMA TX (`20.51 GB/s`) (`op_collectives.cu`, `collectives.rs`):**
     - On AMD EPYC hosts without `iommu=pt`, GPU SM stores (`st.global.v4.u32`) to mapped pinned host memory achieve only `2.36 GB/s` (`2,570 µs` per `M = 512` collective on `27B`), whereas Copy-Engine `cudaMemcpyAsync` D2H achieves `20.51 GB/s` and SM reads achieve `16.82 GB/s`. Using `st.global.wb.v4.u32` (write-back streaming store bypassing L2 read-allocate) for `< 64 KiB` (`9.9 µs` at `M = 1`) and Copy-Engine `cudaMemcpyAsync` TX + 2-barrier `uint4` SM RX for `>= 64 KiB` (`kDmaThresholdBytes = 65536`) across `f16`, `bf16`, `f32`, `residual_gemma_rms_norm_hadamard_f16`, and `hc_combine_tap_f16` cut `M = 512` collective time from `2,570 µs` to `443 µs` (`5.8×` faster) and `M = 8` from `46.4–64.6 µs` to `27.3–30.9 µs`, lifting `27B TP=2` prefill `1,181 -> 3,363 tok/s` (`2.82×`), `9B TP=2` prefill `3,572 -> 9,180 tok/s` (`2.57×`), `4B TP=2 DFlash2` decode `1,454.2 -> 1,645.5 tok/s` (`+13.2%`), and `27B-HGPTQ TP=2 DFlash2` decode `310.8 -> 333.6 tok/s` (`+7.3%`).

### 12.3 Measured 2× RTX 4090 (`sm_89`) vs 2× RTX 3090 Ti (`sm_86`) & `vLLM 0.31.0` / `SGLang v0.5.19` Oracles

| Model & Config | `vLLM 0.31.0` / `SGLang v0.5.19` Oracle (`3090 Ti`) | `warp-34` `3090 Ti` (`sm_86`) Decode (`tok/s`) | `warp-34` `4090` (`sm_89`) Before (`tok/s`) | `warp-34` `4090` (`sm_89`) After (`tok/s`) | `warp-34` `4090` Prefill (`tok/s`) |
|---|---:|---:|---:|---:|---:|
| `Qwen3.5-4B` (`TP=1`) | `169.3` (`vLLM`) | `223.9` | `224.8` | **`226.4`** | **`11,291.0`** |
| `Qwen3.5-4B` (`TP=1`, `MTP k=2`) | — | `344.2` | `353.3` | **`356.7`** | `5,152.3` |
| `Qwen3.5-4B` (`TP=1`, `DFlash2 M=16`) | `462.6` (`SGLang`) | `647.5` | `805.2` | **`817.6`** | `9,340.7` |
| `Qwen3.5-9B` (`TP=1`) | `102.7` (`vLLM`) | `134.9` | `136.4` | **`138.3`** | **`6,496.5`** |
| `Qwen3.5-9B` (`TP=2`) | `120.2` (`vLLM`) | `211.7` | `219.6` | **`230.0`** | **`5,771.0`** |
| `Qwen3.5-9B` (`TP=2`, `MTP k=2`) | — | `370.1` | `383.6` | **`400.7`** | `4,295.0` |
| `Qwen3.8-27B` (`TP=1`) | OOM (`vLLM` FP16 KV) | `54.5` | `55.0` | **`55.1`** | **`2,608.0`** |
| `Qwen3.8-27B` (`TP=1`, `MTP k=2`) | — | `88.7` | `93.8` | **`94.1`** | `1,510.7` |
| `Qwen3.8-27B` (`TP=2`) | `57.2` (`vLLM`) | `90.8` | `91.2` | **`91.9`** | **`2,226.5`** |
| `Qwen3.8-27B` (`TP=2`, `MTP k=2`) | — | `149.2` | `158.9` | **`159.9`** | `1,353.5` |
| `Qwen3.8-27B` (`TP=2`, `DFlash2 M=8`) | `124.8` (`SGLang`) | `164.6` | `180.4` | **`182.0`** | `1,720.1` |
| `Qwen3.6-35B-A3B MoE` (`TP=1`) | OOM (`vLLM` 1×24GB) | `213.6` | `228.5` | **`242.4`** | **`3,538.9`** |
| `Qwen3.6-35B-A3B MoE` (`TP=2`) | `158.4` (`vLLM`) | `287.2` | `281.2` | **`308.4`** | **`4,868.4`** |
| `Qwen3.6-35B-A3B MoE` (`TP=2`, `MTP k=2`) | `241.0` (`vLLM MTP`) | `466.4` | `476.3` | **`507.1`** | `2,713.6` |

### 12.4 Live 2× RTX 4090 `nsys` Per-Step Category Breakdown (`µs/step`, 64 Decode Steps)

| Config (`2× RTX 4090`, Rank 0 `µs/step`) | Marlin | MoE | Collectives (`TP=2`) | GEMV (`in_proj_ba` / `lm_head`) | GDN | Paged Attn (`TQ4`) | Norm / RoPE / SiLU | DFlash / `cuBLAS` | Total GPU `µs/step` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4B TP=1` | `1,942.5` | — | — | `272.6` | `414.5` | `94.7` | `270.4` | — | `3,004.7` |
| `4B TP=1 DFlash2 (M=16)` | `2,052.1` | — | — | `266.8` | `645.7` | `156.9` | `395.9` | `2,798.7` | `6,343.6` (`/round`) |
| `9B TP=1` | `3,855.7` | — | — | `303.6` | `471.7` | `115.6` | `293.0` | — | `5,050.2` |
| `9B TP=2` | `1,882.2` | — | `437.4` | `283.7` | `316.9` | `97.0` | `227.6` | — | `3,254.2` |
| `27B TP=1` | `9,114.2` | — | — | `584.7` | `1,151.1` | `317.7` | `529.3` | — | `11,717.0` |
| `27B TP=2` | `5,505.8` | — | `911.0` | `575.4` | `679.7` | `227.5` | `442.7` | — | `8,359.7` |
| `35B-A3B TP=1` (Before → After) | `903.4` → **`891.5`** | `970.4` → **`859.5`** (`-11.4%`) | — | `384.6` → **`377.4`** | `460.0` → **`452.9`** | `113.0` → **`110.5`** | `268.2` → **`264.6`** | — | `3,109.7` → **`2,966.1`** |
| `35B-A3B TP=2` (Before → After) | `488.3` → **`477.1`** | `719.2` → **`564.3`** (`-21.5%`) | `536.2` → **`509.1`** (`-5.1%`) | `376.4` → **`368.1`** | `331.4` → **`319.9`** | `95.6` → **`92.9`** | `238.9` → **`232.9`** | — | `2,794.7` → **`2,572.9`** |
| `35B-A3B TP=2 MTP (k=2)` (Before → After) | `541.5` → **`537.5`** | `1,358.2` → **`1,236.2`** (`-9.0%`) | `572.1` → **`546.8`** (`-4.4%`) | `395.9` → **`383.8`** | `512.0` → **`507.6`** | `124.2` → **`111.7`** | `289.2` → **`283.1`** | — | `3,803.6` → **`3,617.0`** (`/round`) |

### 12.5 Roofline Assessment & Future `sm_89`-Specific Opportunities

1. **Dense `M=1` Decode Is at the `1,008 GB/s` GDDR6X Roofline:**
   - Because RTX 4090 and RTX 3090 Ti share the exact same `384-bit @ 21 Gbps` (`1,008 GB/s`) GDDR6X memory bus, single-request `M=1` dense decode (`27B TP=1` at `55.1 tok/s` streaming `14.5 GB`/token, `9B TP=1` at `138.3 tok/s` streaming `5.4 GB`/token, `4B TP=1` at `226.4 tok/s` streaming `2.6 GB`/token) already achieves `~85–90%` of effective DRAM bandwidth (`~930 GB/s` inside the Marlin GEMMs).
   - Where 4090 pulls ahead by `1.26×–1.72×` over 3090 Ti is whenever work becomes compute-bound, L2-resident, or multi-CTA occupancy-sensitive: **prefill (`M=512`)** (`11,291 tok/s` on `4B`, `6,496 tok/s` on `9B`, `4,868 tok/s` on `35B TP=2`), **speculative verification (`M=3..16`)** (`817.6 tok/s` on `4B DFlash2`, `507.1 tok/s` on `35B TP=2 MTP`, `400.7 tok/s` on `9B TP=2 MTP`, `182.0 tok/s` on `27B TP=2 DFlash2`), and **sparse MoE decode** (`308.4 tok/s` on `35B TP=2`, `242.4 tok/s` on `35B TP=1`).
2. **Untapped `sm_89`-Only Hardware Opportunities (Future Passes):**
   - **Native Ada `FP8` (`e4m3` / `e5m2`) Tensor Core MMA (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` — `330.3 TFLOPS` FP32-acc on 4090):**
     - `sm_86` lacks hardware `FP8`, whereas `sm_89` provides `2×` Tensor Core throughput (`330.3 TF` vs `165.2 TF` FP16/BF16) and `0.5×` weight bandwidth in `FP8`. Quantizing the 5-layer `DFlash2` drafter from `BF16` (`cuBLAS`, `2,798.7 µs/round` on `4B`) to `FP8` (`W8A8-FP8` or `W4A8-FP8`) on `sm_89` would cut drafter latency in half, pushing `4B DFlash2` past `~950+ tok/s` and `27B DFlash2` past `~200+ tok/s`, and accelerating `M >= 64` prefill.
   - **Multi-Token `m_e in {2, 3}` Register-Broadcast Sparse MoE GEMV (`op_moe.cu`):**
     - During `35B-A3B` speculative verify (`M=3` for `MTP k=2`), `MoE` takes `1,236.2 µs/round` (vs `564.3 µs/step` at `M=1`) because with `M=3 × top-8 = 24` expert assignments across `256` experts, most active experts receive `m_e = 1` or `2` tokens (falling below the `m_e >= 4` `mma.sync` tile threshold and streaming weights per token row). A specialized `m_e in {2, 3}` warp-per-expert GEMV that loads each expert's packed `int32` weight once into registers and multiplies against `2` or `3` token activation vectors simultaneously would save `~200–250 µs/round` on `35B-A3B MTP` (`507 -> ~550+ tok/s`).

---

## Source index

- GA102 whitepaper (v1/v2): nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-whitepaper-v1.pdf (SM datapaths, per-SM composition, GDDR6X, NVLink3)
- Ada whitepaper Appendix A: nvidia.com/content/PDF/nvidia-ada-gpu-architecture-whitepaper-v2.pdf (official RTX 3090 Ti tensor/FLOPS table)
- CUDA C++ Programming Guide — Compute Capabilities appendix: docs.nvidia.com/cuda/cuda-programming-guide (limits, ratios, data types, carveouts)
- CUDA C++ Best Practices Guide §12.1: docs.nvidia.com/cuda/cuda-c-best-practices-guide (instruction throughput table, CUDA 13.4)
- CUDA Ampere Tuning Guide: docs.nvidia.com/cuda/archive/12.6.1/pdf/Ampere_Tuning_Guide.pdf (8.0-vs-8.6 deltas)
- PTX ISA 9.4: docs.nvidia.com/cuda/parallel-thread-execution (target gates for mma/mma.sp/ldmatrix/cp.async/mbarrier/redux/L2 hints)
- NVIDIA product pages + TechPowerUp GA102-350 entry (card-level clocks/BDW/TDP; FP64 1:64 pattern)
- NVIDIA developer forums: "TF32 TFLOPs of GeForce RTX 3090 vs A40" (GA10x TF32 rate), "NVLink port support for RTX 3090 Ti" (NVLink removal)
- Pinned llama.cpp (`vendor/llama.cpp` @ cb7934c5): `ggml-common.h` block
  structs (bpw math), `ggml-cuda/mmq.cu` type dispatch (MMQ/IMMA coverage)
