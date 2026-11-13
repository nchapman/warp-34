# CUDA Kernel Correctness & Performance Audit: `9ef1964`

## Scope and method

Full clean-room audit of all **37 CUDA C++ source and header files** (`~28.5k` LOC under `crates/w34-kernels/csrc/`), the Rust FFI seam (`crates/w34-kernels/src/ffi*.rs`, `crates/w34-kernels/src/lib.rs`), and cold-start execution-plan warmup (`crates/w34-host/src/cuda/warmup*.rs`) at baseline commit `9ef1964`.

Static review was paired with binary artifact inspection (`cuobjdump --dump-resource-usage`, PTX, and SASS) across all compiled kernel specializations for both `sm_86` (RTX 30-series, `28–84` SMs, `3–6 MiB` L2) and `sm_89` (RTX 40-series, `24–142` SMs, `24–96 MiB` L2) built with `-O3 -lineinfo -std=c++17 --expt-relaxed-constexpr -use_fast_math -extended-lambda`.

Subsystems audited:
1. **Quantized Linear, Small-N GEMM, & Multi-LoRA:** `marlin/*` (`marlin_template.h`, `dequant.h`, `marlin_mma.h`, `marlin.cuh`, `marlin_dtypes.cuh`), `ops/op_marlin.cu`, `ops/op_small_n_gemm.cu`, `ops/op_lora.cu`.
2. **KV Cache, TurboQuant (`TQ4`/`Q8_0`), Paged Attention, Paged FA2, & QSA:** `ops/tq4_kv_common.cuh`, `ops/op_kv_cache.cu`, `ops/op_paged_attn.cu`, `ops/op_paged_fa2.cu`, `ops/op_qsa.cu`, `attention/*`.
3. **Gated DeltaNet (`GDN`):** `ops/op_gdn.cu`, `ops/op_gdn_wy.cu`, `ops/op_gdn_wy_chunk64.cu`.
4. **Mixture-of-Experts (`MoE`), Router, GPU LRU, & Hybrid CPU Bridge:** `ops/op_moe.cu`, `ops/moe_sparse_inline.cuh`, `ops/op_moe_router.cu`, `ops/op_moe_cpu_bridge.cu`, `ops/moe_cpu_bridge.h`.
5. **Norms, RoPE, Activations, FWHT, HyperConnections (`HC`), & Per-Layer Embeddings (`PLE`):** `ops/op_norms_rope_act.cu`, `ops/hadamard_inline.cuh`, `ops/op_hc.cu`, `ops/op_ple.cu`.
6. **Multi-GPU Collectives (`TP=2`/`TP=4`), `DFlash2`, Rejection/MTP, GPU Sampler, Vision, & Init:** `ops/op_collectives.cu`, `ops/op_dflash2.cu`, `ops/op_rejection.cu`, `ops/op_sampler.cu`, `ops/op_vision.cu`, `ops/op_init.cu`.

---

## 1. Critical Correctness Bugs

### C1. Paged FlashAttention-2 partial-tile clamp reads past `block_tables`
- **Location:** `crates/w34-kernels/csrc/ops/op_paged_fa2.cu:38-50`
- **Root cause:** In `PagedKVManager::load_page`, threads past `*partial_block_size` clamp their row offset using `ceil_div(final_row_offset, kGmemRowsPerThread) * kGmemRowsPerThread` (`kGmemRowsPerThread = 4`, page size `16`). When `*partial_block_size == 32` (`2` pages), `final_row_offset = 31`, `ceil_div(31, 4) * 4 = 32`, and `virtual_page_idx = 32 >> 4 = 2`. Line 49 unconditionally executes `block_table[2]` before any KV load predication, reading out of bounds whenever `block_tables` is allocated tightly to `max_blocks_per_req` (also triggered for partial tile lengths `14..16`, `30..32`, `46..48`).
- **Fix:** Floor to the 4-row slice owning `final_row_offset`: `(final_row_offset / kGmemRowsPerThread) * kGmemRowsPerThread`.

### C2. Sampler block-reduction helpers reuse shared memory without a trailing `__syncthreads()`
- **Location:** `crates/w34-kernels/csrc/ops/op_sampler.cu:40-124` (call sites at `609-613`, `898-904`, `974-976`)
- **Root cause:** `block_reduce_max_pair`, `block_reduce_sum`, `block_reduce_sum_f64`, and `block_reduce_sum_u32` synchronize before returning `s_val[0]`, but do not synchronize *after* all warps read `s_val[0]`. When called back-to-back on the same `s_u32`/`s_f32` scratch array (e.g. `valid_count` followed immediately by `any_nonfinite` at `op_sampler.cu:974-976` across `1024` threads / `32` warps), warp 0 can enter the second reduction and overwrite `s_u32[0]` before warps `1..31` finish reading `valid_count`. Warps `1..31` can then observe `valid_count == 0` and take the early `return;` at line 978 while warp 0 reaches subsequent `__syncthreads()` barriers—corrupting outputs or hanging the CTA.
- **Fix:** Copy `s_val[0]` (and `s_idx[0]`) to registers, call `__syncthreads()`, and return the register values.

### C3. `TP=2` / `TP=4` SM-staged collectives clobber peer staging banks across changing CTA counts
- **Location:** `crates/w34-kernels/csrc/ops/op_collectives.cu:47-58, 101-125, 166-288, 975-1007, 1662, 1707`
- **Root cause:** Non-DMA (`< 64 KiB`) collectives use a 1-barrier, 2-bank (`even`/`odd`) protocol indexed by per-CTA `self_sg->flag[blockIdx.x] + 1`, with no post-read barrier. Different collective entrypoints launch different `gridDim.x` and partition the same staging buffer differently (`w34_tp2_all_reduce_residual_gemma_rms_norm_f16` launches `blocks = num_tokens` with row-major ownership; `w34_tp2_all_reduce_f16` launches `blocks = ceil(num_elems / 4096)` with strided ownership). On `Qwen3.6-35B-A3B` (`TP=2`, `H=2048`) at `M=4` MTP verify (`16 KiB`), alternating between a 4-CTA fused norm and a 2-CTA flat all-reduce desynchronizes `flag[0..1]` vs `flag[2..3]` and lets Rank 0's CTAs `0..1` overwrite elements `4096..8191` while Rank 1's CTAs `2..3` from the prior collective are still reading them. `TP=4` shares the same per-CTA epoch hazard.
- **Fix:** Track a communicator-wide call/bank epoch and enforce cross-rank completion before reusing staging regions when CTA counts or layouts differ (or add a lightweight post-read barrier when switching layouts/CTA counts).

### C4. GDN WY chunked kernels produce `NaN` under strong decay and poison persistent `ssm_state`
- **Location:** `crates/w34-kernels/csrc/ops/op_gdn_wy.cu:285, 312, 383, 728, 1101` and `crates/w34-kernels/csrc/ops/op_gdn_wy_chunk64.cu:176, 231, 301, 559, 606` (gate product also in `ops/op_gdn.cu:582, 719`)
- **Root cause:**
  1. Both WY chunk kernels compute cumulative FP32 prefix sums `s_g` and evaluate `expf(s_g[i] - s_g[j])` (including on the `col == row` diagonal where the exact value is `1.0f`). If `s_g` underflows/overflows to `-inf`, `-inf - (-inf)` yields `NaN`; even when finite, large negative cumulative prefixes lose low-order mantissa precision so `s_g[i] - s_g[j]` cancels to `0.0f` (`exp(0) = 1` instead of `exp(g)`).
  2. `-expf(a_log[hv]) * softplus_f32(a + dt_bias)` can form `inf * 0 = NaN` for extreme finite inputs. Any `NaN` writes directly into persistent `ssm_state` and `MambaRadixCache` checkpoints. (Additionally, `tests/backbone_and_collectives.rs` uses `.fold(0.0f32, f32::max)`, which swallows `NaN` in Rust.)
- **Fix:** Guard gate evaluation against `inf * 0`, compute exact `1.0f` on `i == j` and safe bounded segment differences `g_segment_exp(s_g, i, j)` (or per-chunk re-anchored cumulative `g` so `s_g[0]` starts near `0` within each chunk and non-finite differences clamp cleanly to `0.0f`), and assert `.is_finite()` in GDN parity tests.

---

## 2. Important Correctness, Memory-Ordering, & Contract Bugs

### I1. Marlin Split-K dangling pointer UB & missing release store on atomic Split-K init
- **Location:** `crates/w34-kernels/csrc/marlin/marlin_template.h:477, 1645-1663`
- **Root cause:**
  - `marlin_template.h:1645-1663`: `int2 tmp` / `int4 tmp` are declared inside `if/else` blocks while `c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp)` is dereferenced after those blocks end (lifetime UB).
  - `marlin_template.h:477`: When `part_use_atomic_add` is active (reachable on smaller-SM 30/40-series GPUs when `blocks <= mn_tiles` stripes across K), `locks[locks_off] = 1 - slice_count;` uses a plain store after zeroing `C[...]`, lacking a `.release.gpu` store to order zero-init writes before peer CTAs' acquire-spin + `atomicAdd`.
- **Fix:** Hoist `tmp` into the enclosing scope and publish `locks[locks_off]` with `st.release.gpu.global.s32` (preceded by `__syncthreads()`).

### I2. Marlin repack 32-bit `out_offset` overflow & `int64_t -> int` narrowing before validation
- **Location:** `crates/w34-kernels/csrc/ops/op_marlin.cu:131, 580-587, 727-802`
- **Root cause:** `int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_out_size;` overflows 32-bit signed `int` on very large matrices (`> 2^31` `uint32_t` words), and `w34_marlin_gemm_*` narrows `int64_t m, n, k` to `int` before checking bounds.
- **Fix:** Use `size_t` for `out_offset` and validate `m, n, k` in `int64_t` before narrowing.

### I3. Cold-start Marlin warmup misses `N=128/256` and `thread_m_blocks=2` (`M=17..32`) kernels
- **Location:** `crates/w34-host/src/cuda/warmup.rs:316-385`
- **Root cause:** `warmup.rs` only warms `N=64, K=128` across `M={1, 16, 40, 64}`, leaving `th_cfg = {128, 128, 128}`, `{64, 256, 256}`, and `thread_m_blocks = 2` (`M=17..32`) un-warmed until live inference—risking CUDA 13 lazy module-load serialization/deadlock under `TP=2`/`TP=4`.
- **Fix:** Warm `N=256, K=256` (plus `N=64, K=128`) across `M={1, 16, 24, 40, 64}`.

### I4. MoE grouped MMA reuses scales/zero-points across 64-element K chunks when `group_size % 64 != 0`
- **Location:** `crates/w34-kernels/csrc/ops/op_moe.cu:796-840, 987-1045, 1165-1180`
- **Root cause:** `can_use_grouped_w4a16` admits any `group_size >= 16` divisible by 16 (e.g. `96`), but the inner loop hoists one scale/ZP per 64-element `kc` tile whenever `group_size >= 64`, reading the wrong group at `kc = 64, ks = 2` (`col_base = 96`).
- **Fix:** Hoist across the 64-element chunk only when `(group_size & 63) == 0`.

### I5. GDN WY duplicate-offset checkpoint drop, init error swallowing, & conv `kernel_size` guard
- **Location:** `crates/w34-kernels/csrc/ops/op_gdn_wy.cu:171-179, 1255-1261`, `ops/op_gdn_wy_chunk64.cu:197-205`, `ops/op_gdn.cu:72, 134, 927-953`
- **Root cause:**
  - `WY` chunk planning `break`s on the first checkpoint matching `trig_off == base_s + tile_len`, dropping subsequent checkpoints at the same token offset (or stopping on a `-1` sentinel before a valid slot).
  - `op_gdn_wy.cu:1260` returns `cudaGetLastError()` after `w34_gdn_init_wy_chunk64_f16()` already consumed the error via `cudaGetLastError()`, silently returning `0` without launching the kernel.
  - `op_gdn.cu:927` does not reject `kernel_size > 4` before indexing fixed `float hist[3]; float w_reg[4];`.
- **Fix:** Emit all valid checkpoints matching a step/boundary, propagate `init_rc` directly, and validate `1 <= kernel_size <= 4`.

### I6. `TP=4` collective rank-dependent grid mismatch & `w34_get_tp2_min_sm_count`
- **Location:** `crates/w34-kernels/csrc/ops/op_collectives.cu:1033-1051, 1932-1939, 2034-2072`, `ops/op_init.cu:175-200`
- **Root cause:** `tp4_all_reduce_{f16,bf16,f32}` chooses `elems_per_block = vec8 ? 4096 : 512` using each rank's local pointer alignment (causing rank grid mismatch + deadlock if alignments differ), and `tp4_block_cap` queries `w34_get_tp2_min_sm_count()` which inspects only devices `0` and `1` rather than all 4 ranks.
- **Fix:** Compute `blocks` invariantly from `num_elems` (`4096` elements/CTA regardless of scalar/vec8 loop) and clamp `tp4_block_cap` across all 4 participating devices.

### I7. Inconsistent `valid_req_mask` & fallback validation in KV write and QSA
- **Location:** `crates/w34-kernels/csrc/ops/op_kv_cache.cu:238-263`, `ops/op_qsa.cu:249-268, 444-472, 631-653, 1130-1132, 1368-1386`
- **Root cause:**
  - `w34_kv_cache_write` (FP16) and `w34_qsa_select_blocks_f16` do not check `batch.valid_req_mask` (unlike `w34_kv_cache_write_bf16`).
  - `op_qsa.cu:1379` falls back to `qsa_sparse_paged_attn_kernel` on invalid `local_kv_heads == 0` or `local_q_heads % local_kv_heads != 0`, dividing by zero on GPU at line 1130 instead of returning `cudaErrorInvalidValue`.
  - `op_qsa.cu:249-260` executes the null-norm mean-pooling write on all 4 token CTAs instead of gating on `(slot & 3) == 3`.
- **Fix:** Enforce `valid_req_mask` consistently, validate GQA head divisibility up front, and gate null-norm `comp_k_cache` writes consistently.

### I8. Standalone `w34_gemma_rms_norm_f16` silently truncates `hidden_size > 12288`
- **Location:** `crates/w34-kernels/csrc/ops/op_norms_rope_act.cu:178-197, 813-816`
- **Root cause:** `w34_gemma_rms_norm_hadamard_f16` dispatches `gemma_rms_norm_f16_kernel<48>` (`12,288` max cols) in its `else` branch without rejecting `hidden_size > 12288` (or providing a dynamic loop fallback).
- **Fix:** Add a strided multi-pass fallback kernel for `hidden_size > 12288` (matching `op_dflash2.cu`'s generic norm fallback) and validate `hidden_size > 0`.

### I9. MoE router, GPU LRU, & Hybrid CPU bridge boundary validation
- **Location:** `crates/w34-kernels/csrc/ops/op_moe_router.cu:135, 382, 555-609, 957-997`, `ops/op_moe_cpu_bridge.cu:371-438`
- **Root cause:**
  - `w34_moe_gpu_lru_update_and_pull` and `w34_moe_hybrid_plan_and_submit` launch table-mutating kernels before validating pack dimensions/pointers.
  - `w34_moe_topk_router_f16` accepts `top_k > num_experts` and skips `norm_topk_prob` when sigmoid `w_sum <= 1e-12f` (`w_sum > 0.0f` is still normalizable in FP32).
  - `w34_moe_hybrid_wait_and_add` does not validate `num_tokens <= 8 && hidden <= 4096` before reading `MoeBridgeReqSlot::y_f32[8 * 4096]`.
- **Fix:** Validate all arguments before mutating GPU LRU tables, reject `top_k > num_experts`, normalize whenever `w_sum > 0.0f` and finite, and check `MoeBridgeReqSlot` bounds in `wait_and_add`.

### I10. CUDA `grid.y <= 65535` limits, vector alignment guards, & in-place `__restrict__` aliasing
- **Location:** `crates/w34-kernels/csrc/ops/op_norms_rope_act.cu:1160-1206, 1818-1887, 1907-1948`, `ops/op_small_n_gemm.cu:267-360`, `ops/op_hc.cu:71, 113, 248`, `ops/op_ple.cu:397`
- **Root cause:**
  - `w34_silu_and_mul_*` and `w34_apply_attn_output_gate_hadamard_f16` place `num_tokens` in `grid.y`, failing with `cudaErrorInvalidConfiguration` at `num_tokens >= 65536` (e.g. long-context prefill).
  - `w34_small_n_gemm_*` and `w34_midloop_damped_euler_f16` perform `uint4`/`__half2` loads without pointer alignment checks.
  - `hc_rmsnorm`, `hc_down_silu_inject`, `hc_combine`, `ple_signed_sqrt_gate_norm_add`, and `midloop_damped_euler` annotate pointers `__restrict__` that `w34-host` callers intentionally alias in place (`dst == src`).
- **Fix:** Put `num_tokens` in `grid.x` (`<= 2^31 - 1`) or chunk `grid.y`, check pointer alignment before vectorized entrypoints, and drop `__restrict__` on in-place aliased parameters.

---

## 3. Suspected Hazards Audited & Verified Clean

1. **`-use_fast_math` on RoPE (`op_norms_rope_act.cu:12-16, 545-548`, `op_qsa.cu:17-19`):** Calls libdevice `__nv_sinf`, `__nv_cosf`, `__nv_powf`, and `__fdiv_rn` directly, preserving full Payne–Hanek range reduction at `rope_theta = 1e7` and `262k` positions.
2. **`W2A16` (`kU2B2`) dequantization (`marlin/dequant.h:127-172`):** `0x3400` is exact normal FP16 `0.25` (not subnormal), and `__hsub2`/`__hfma2` emit packed FP16 instructions unaffected by F32 FTZ.
3. **`TQ4` (`66 B` block / `132 B` row) alignment & 64-bit KV pool addressing (`tq4_kv_common.cuh`, `op_kv_cache.cu`, `op_paged_attn.cu`):** `qs` accesses use `uint8_t` or aligned `uint16_t` loads (never misaligned `uint32_t*` at `+2 B`), and all `page_id * page_stride` arithmetic widens to 64-bit (`size_t` / `int64_t`) before multiplication.
4. **Sequential Marlin Split-K & GPU↔CPU Hybrid Mailbox ordering:** Sequential Marlin Split-K pairs `ld.global.acquire.gpu.b32` with `fence.acq_rel.gpu`; `op_moe_cpu_bridge.cu` pairs `__threadfence_system()` + `st.release.sys.global.b64` with `ld.acquire.sys.global.b64`; same-stream `cudaMemcpyAsync` DMA-TX before `st.release.sys` in `op_collectives.cu` is ordered by CUDA stream task boundaries.
5. **Vision GELU & PLE signed-sqrt / E4M3FN:** Vision tanh-GELU uses `__nv_tanhf` (immune to `exp(2x)` overflow `NaN`); PLE uses `sqrtf(fabsf(v) + eps)` (`MUFU.SQRT`).

---

## 4. High-Leverage `sm_86` & `sm_89` Performance Findings

| ID | Kernel / File | Baseline `cuobjdump` (`sm_86` / `sm_89`) | Bottleneck & Planned Optimization |
|---|---|---|---|
| **P1** | `w34_moe_topk_router_f16` (`op_moe_router.cu:324-402`) | `256/k8`: `REG=36, STACK=0`<br>`>256` generic: **`REG=40, STACK=256, SHM=4224`** | `Qwen3.8-Flash-Next` (`512` experts, `top_k=10`, `48` layers) misses the `<=256` warp path and hits the 128-thread `STACK=256` fallback. Add `moe_topk_router_512_k10_f16_kernel` (`1` warp, `16` experts/lane in regs, warp-shuffle top-10, `STACK=0`). |
| **P2** | `launch_marlin` (`op_marlin.cu:698-703`) | Every launch reserves `max_shared_mem = 101,376 B` | Unconditionally reserving `99 KiB` dynamic smem caps occupancy at **1 CTA/SM** (`4–8` warps/SM) even when small-M tiles use `< 32 KiB`, and calls `cudaFuncSetAttribute` every GEMM. Compute exact tile `smem_size` and cache `cudaFuncSetAttribute` per device/kernel. |
| **P3** | `norm_bf16_vec8_4warp` (`op_dflash2.cu:163-232`) | **`REG=46, STACK=384`** | Runtime-indexed `float vals[12][8]` spills `384 B`/thread to `.local` memory on every non-`2560` `<= 3072` BF16 RMSNorm. Template `norm_bf16_vec8_4warp<ROUNDS>` for `STACK=0`. |
| **P4** | `moe_hybrid_plan_and_submit_kernel` (`op_moe_cpu_bridge.cu:110-176`) | **`REG=40, STACK=2048, SHM=516`** | Five `int32[128]` arrays on thread 0's stack generate `2 KiB` of `.local` `STL`/`LDL` traffic. Move scratch arrays to `__shared__` (`~2.5 KiB` total smem, `STACK=0`). |
| **P5** | `sampler_step_f16_kernel` (`op_sampler.cu:257, 328, 811`) | **`REG=64, STACK=224, SHM=3736`** + `ATOMS.CAST.SPIN.64` | Move per-thread `picked_lp_ids[20]` / `picked_lp_vals[20]` (`160 B`) into `__shared__` (`STACK=0`), skip constant `0` top byte(s) in token-ID radix CDF, and warp-aggregate bin sums / replace 64-bit `double` CAS spin loops. |
| **P6** | `tp4_*` & `tp2_all_reduce_residual_...<512,12>` (`op_collectives.cu:236, 975`) | `tp4_*`: **`STACK=64..112`**<br>`tp2<512,12>`: **`REG=64, STACK=16`** | Pass pre-resolved `(my_stage, peer0, peer1, peer2)` scalars instead of runtime-indexed `Tp4Ptrs` arrays (`STACK=0`), and tune `tp2<512,12>` launch bounds (grid is `<= 16` CTAs) to eliminate the 16 B spill. |
| **P7** | `op_lora.cu` (`13, 423-478, 535-558`) | `fused_512t`: **`REG=64, STACK=16`**<br>`expand_add`: **`REG=48, STACK=32`** | Tune `lora_fused_shrink_expand_512t` unrolling and move `lora_expand_add`'s runtime-indexed `lids[8]` to `__shared__` so all LoRA kernels compile with `STACK=0`. |
| **P8** | `gdn_wy_chunk64` (`op_gdn_wy_chunk64.cu:151-155, 604`) | `REG=203, SHM=99,072, STACK=0` | Pad/swizzle shared-memory `s_H` staging in `WY64` fallback to eliminate 32-way shared-memory bank conflicts on column accesses. |
| **P9** | Sparse MoE `M=2..8` (`op_moe.cu:1843-1880`, `moe_sparse_inline.cuh:16`) | `moe_sparse_vec4` gate: **`REG=128`** | Add multi-token expert weight reuse / launch-bounds tuning for small-batch `MTP` verify (`M=2..4`) where multiple verify tokens hit the same expert or `vec4` caps occupancy at 16 warps/SM. |
| **P10** | `RoPE` & `HC` / `PLE` decode parallelism (`op_norms_rope_act.cu:545`, `op_hc.cu`, `op_ple.cu`) | `RoPE`: `REG=37, STACK=32`<br>`HC`/`PLE`: `1 CTA / token` | Compute RoPE `sin`/`cos` once per rotary pair in `__shared__` per token CTA (eliminating `20..72×` per-head libdevice `__nv_powf`/`__nv_sinf`/`__nv_cosf` redundancy), and parallelize elementwise `HC`/`PLE` projection reductions across multiple CTAs at small `M`. |

---

## 5. Remediation Status

All **24 findings** (`C1`–`C4`, `I1`–`I10`, `P1`–`P10`) are implemented, regression-tested, sanitizer/binary verified, and passing `mise run check` across `sm_86` and `sm_89`:

| Stage | Commit | Items | Measured Binary / Sanitizer / Runtime Verification |
|---|---|---|---|
| **Doc** | `147904e` | Initial audit baseline (`9ef1964`) | `cuobjdump` across `1,043` `sm_86`/`sm_89` kernels + full 37-file CUDA source review |
| **Stage 1** | `213173d` | Critical fixes (`C1`–`C4`) | `C1` (`op_paged_fa2.cu`): `compute-sanitizer --tool memcheck` **`152 -> 0` OOB errors**; `C2` (`op_sampler.cu`): `--tool racecheck` **`1,500+ -> 0` races**; `C3` (`op_collectives.cu`): communicator-wide `next_comm_bank_odd` bank parity across changing CTA counts; `C4` (`op_gdn_wy*.cu`, `op_gdn.cu`): `-104.0f` decay clamp + finite `ssm_state` parity |
| **Stage 2** | `bfea69c` | Important fixes (`I1`–`I5`) | `I1` (`marlin_template.h`): Split-K barriers + `st.release.gpu.global.s32` (`--tool racecheck` **`60 -> 0` hazards**); `I2` (`op_marlin.cu`): 64-bit `size_t` repack offsets & `int64_t` host bounds; `I3` (`warmup.rs`, `op_marlin.cu`): `w34_marlin_warmup_specializations()` warming `{128,128,128}`, `{64,256,256}`, and `thread_m_blocks=2`; `I4` (`op_moe.cu`): `(group_size & 63) == 0` hoist guard; `I5` (`op_gdn*.cu`): multi-checkpoint emit, `init_rc` propagation, `1 <= kernel_size <= 4` guard |
| **Stage 3** | `d6e88d6` | Important fixes (`I6`–`I10`) | `I6` (`op_collectives.cu`, `op_init.cu`): rank-invariant `TP=4` CTA grid + 4-device SM clamp; `I7` (`op_kv_cache.cu`, `op_qsa.cu`): `valid_req_mask` & GQA divisibility guards; `I8` (`op_norms_rope_act.cu`): `gemma_rms_norm_strided_f16_kernel` fallback for `hidden_size > 12288`; `I9` (`op_moe_router.cu`, `op_moe_cpu_bridge.cu`): validate-before-mutate LRU/bridge + `top_k <= num_experts` + `wait_and_add` bounds; `I10` (`op_norms_rope_act.cu`, `op_hc.cu`, `op_ple.cu`): `grid.x = num_tokens` (`> 65535` safe), alignment checks, in-place alias-safe pointers |
| **Stage 4** | `7c0ffaa` | Performance (`P1`–`P5`) | `P1` (`op_moe_router.cu`): `moe_topk_router_512_k10_f16_kernel` (`REG:53, STACK:0`, `8.70 -> 4.57 µs`, `1.90×`) & `warp512` (`REG:90, STACK:0`), generic `STACK:256 -> 0`; `P2` (`op_marlin.cu`): exact tile `get_kernel_cache_size` smem + cached `g_marlin_attr_warm_mask`; `P3` (`op_dflash2.cu`): templated `norm_bf16_vec8_4warp<1..=12>` (`STACK:384 -> 0`, `3.48..4.49 -> 2.43..2.84 µs`, `1.43×–1.58×`); `P4` (`op_moe_cpu_bridge.cu`): shared cold-expert scratch (`STACK:2048 -> 0`); `P5` (`op_sampler.cu`): shared `picked_lp_*`, warp-aggregated radix mass/hist, `start_pass` skip (`STACK:224 -> 32`) |
| **Stage 5** | `ca63f67` | Performance (`P6`–`P10`) | `P6` (`op_collectives.cu`): all 12 `tp4_*` (`STACK:64..112 -> 0`) & `tp2<512,12>` (`STACK:16 -> 0`) now **`STACK:0`**; `P7` (`op_lora.cu`): `lora_expand_add` (`STACK:32 -> 0`) & `lora_fused_512t` (`STACK:16 -> 0`) now **`STACK:0`** (`1.02×–1.07×`); `P8` (`op_gdn_wy_chunk64.cu`): `S_H_STRIDE = 72` (`101,120 B` smem) eliminates 32-bank `wmma` conflicts; `P9` (`moe_sparse_inline.cuh`, `op_moe.cu`): `moe_sparse_vec4` (`STACK:16 -> 0`) + 2–4 col/warp `moe_w4a16_down_accum_f16_kernel` packing (`18.06..47.12 -> 8.37..16.47 µs`, `2.16×–2.86×` on `n_interm=640`); `P10` (`op_norms_rope_act.cu`, `op_hc.cu`, `op_ple.cu`): `uint4` RoPE head I/O (`1.09×–1.14×`) + 2D `(num_tokens, col_blocks)` HC/PLE grids (`hc_combine_tap` `10.69 -> 2.30 µs` `4.64×`, `ple_dequant_f8_hadamard` `6.06 -> 1.93 µs` `3.14×`) |
| **Stage 6** | Follow-up | `P5` & `P6` A/B tuning (`op_sampler.cu`, `op_collectives.cu`) | `op_sampler.cu` (`w34_sampler_step_f16`): restored `1024`-thread launch (`REG:64, STACK:32, SHARED:28492`) with `kSamplerCandCap = 2048` shared-memory candidate compaction, `chosen_cnt == 1` early exit, `HalfVec8` loads, and `__all_sync` 5-step `FP64` warp butterfly (`248k` vocab `top_k=50`: **`2,548 -> 750 µs` (`3.4×` vs `d6e88d6`, `5.3×` vs `34d7c44`)**; `top_k=0` nucleus: **`33,998 -> 2,836 µs` (`12.0×`)**; `memcheck=0`, `racecheck=0`); `op_collectives.cu` (`tp2`/`tp4_all_reduce_hc_combine_tap_f16`): `512`-thread single-wave PCIe read for `hidden_size > 2048` + `TP=4` `uint4` vectorization (`STACK:0`, `M=1/4/8`: `11.50/13.71/17.51 -> 9.72/12.09/16.06 µs`, `Flash-Next TP=2` `MTP-3` verify `28.57 -> 25.94 ms/round` `+10.1%`, `Plain` `56.4 -> 66.1 tok/s`, `MTP-3` `92.8 -> 102.4 tok/s`) |
