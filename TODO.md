# warp-34 — Backlog & Hotspot Tracker

Prioritized performance, modularity, and reliability hotspots identified across `crates/w34-host` and `crates/w34-kernels`.

---

## 1. Performance Hotspots

### [x] 1.1 Adaptive Post-Drafter `K_hot` Expansion for Tiered-MoE (`Qwen3.8-Flash-Next`)
- **Locations:** `crates/w34-host/src/model/rank.rs:416–425`, `crates/w34-host/src/moe_offload/host_store.rs`
- **Problem:** `plan_tiered_moe_hot_per_layer` capped `K_hot` at `TIERED_MOE_TP2_DEFAULT_HOT = 112` on `TP=2` (`96` on `TP=1`) during `Qwen3_5Engine::load` to reserve VRAM for a potential `MTP` drafter (`~450 MiB`) or large KV pools. Even after loading `MTP` with `kv_pages = 640` on 2× RTX 3090 Ti, `1.49 GiB` of VRAM per GPU remained unused; with `DFlash2-W4A16-HGPTQ` (`~220 MiB` total footprint and `kv_pages = 256`), `~2.0 GiB` per GPU remained unused. Meanwhile, `DFlash2` (`M=8`, `80` `(token, expert)` pairs/layer) was bottlenecked by PCIe zero-copy cold-expert pulls (`~1.99 MiB` per expert miss).
- **Resolution:** Raised `TIERED_MOE_TP2_DEFAULT_HOT` from `112` to `120` (`240/512` experts across 2 GPUs, using `765 MiB` of the `1.49 GiB` idle VRAM per GPU while retaining `0.64–0.67 GiB` headroom after loading `MTP` with `kv_pages = 640`). Verified on 2× RTX 3090 Ti: `TP=2` Warm Target 64-tok `46.66 -> 65.81 tok/s` (`+41.0%`), `MTP-3` Warm 64-tok `85.08 -> 113.79 tok/s` (`+33.7%`), `DFlash2` Warm 64-tok `137.25 -> 152.18 tok/s` (`+10.9%`), `short-chat 128:512 WARM` `77.63 -> 91.36 tok/s` (`+17.7%`), `chat-512` 5-prompt mean `76.74 -> 81.77 tok/s` (`+6.6%`), `4096:256` prefill `1511.0 -> 1665.0 tok/s` (`+10.2%`) & decode `66.85 -> 72.88 tok/s` (`+9.0%`), `8192:1024` decode `70.00 -> 73.61 tok/s` (`+5.2%`).

### [x] 1.2 `TP=2 / EP=2` CPU MoE Bridge for `StreamedPrefill`
- **Locations:** `crates/w34-host/src/moe_offload/engine.rs`, `crates/w34-host/src/moe_cpu_bridge.rs`
- **Problem:** On `TP=1`, `StreamedPrefill` offloads sparse cold experts (`1..=8` or `1..=96` routed tokens) to the SIMD CPU worker pool (`cache.bridge_runtime`) in parallel with GPU Pass 0 + PCIe H2D streaming. On `TP=2 / EP=2`, `register_cpu_bridge` was gated on `ep_size == 1`, so 100% of cold expert misses in `128–8K` prefill streamed over PCIe H2D DMA.
- **Resolution:** Added `Pool::new_with_cpus` and `Runtime::new_prefill_only(device, ep_rank, ep_size)` in `crates/w34-host/src/moe_cpu_bridge.rs` (partitioning physical CPU cores disjointly across `ep_size` ranks without spawning a decode `moe-responder` thread) and enabled `StreamedPrefill` tail cold-expert CPU offload on `TP=2 / EP=2` (`cold.len() > ws.stage_capacity` or `cold.len() > 2 * ws.stage_capacity`) while keeping decode/spec-verify 100% on `GpuLru`. Verified on 2× RTX 3090 Ti: `short-chat 128:512 WARM` TTFT `470.52 -> 425.19 ms` (`301.0 tok/s` prefill, `91.50 tok/s` decode), `chat-512` prompt TTFTs cut by `9–24%`, and `8192:1024` prefill `1,575.5 -> 1,696.3 tok/s` (`73.88 tok/s` decode).

### [x] 1.3 `moe_sparse_mma_kernel` Double-Buffered Smem & Parallel Epilogue for `DFlash2` (`M=8`) Verify
- **Locations:** `crates/w34-kernels/csrc/ops/moe_sparse_inline.cuh`
- **Problem:** For `Qwen3.8-Flash-Next` + `DFlash2` (`M=8`, `total_pairs = 80`) and `StreamedPrefill` chunks, `moe_sparse_mma_kernel` (`regime = 2`) issued 3 inter-subtile `__syncthreads()` barriers inside the `APPLY_HAD64` (`sub < 4`) loop to reuse a single `partial[4][8][32]` shared-memory staging buffer, and serialized the `GATE == true` epilogue (both non-Hadamard SiLU*Up writeout and `APPLY_HAD64` 64-point WHT + global writeout) on `wid == 0` while warps `1..3` sat idle.
- **Resolution:** Double-buffered `partial[APPLY_HAD64 ? 2 : 1][4][GATE ? 8 : 4][32]` (`sub & 1`) to eliminate all 3 inter-subtile `__syncthreads()` barriers when `APPLY_HAD64 == true`, parallelized the `GATE == true` epilogue across all 4 warps (`const int j = wid;`) and `GATE == false` epilogue across both `warp_k` warps (`for (int jj = 0; jj < 2; ++jj)`), and preloaded `out_tw0`/`out_tw1` before the K-loop with 0-ULP parity (`h64_g64_routed_moe_all_regimes_and_tiered_parity`, `moe_sparse`, and `test_qwen4_exp_real_checkpoint_tp1_and_tp2` all PASS; `TP=1 DFlash2` Warm 64-tok `98.82 -> 104.20 tok/s`).

### [x] 1.4 Small-`M` (`M <= 16`) Tile & Split-K Tuning for `BF16` Marlin `u4b8`
- **Locations:** `crates/w34-kernels/csrc/ops/op_marlin.cu`, `crates/w34-kernels/csrc/marlin/marlin_template.h`
- **Problem:** `w34_marlin_gemm_bf16_u4b8` (used by `Qwen3.5-27B-DFlash2-W4A16-HGPTQ` and `Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ` at `M=8` and `M=16`) skipped `enable_m_invariant_split_k` for `N <= 4096, K >= 1024` (leaving `{128, 64, 128}` on 1 slice for `N = 2560, K = 2048` `o_proj` and `choose_thread_config` checking `get_marlin_kernel_f16` instead of `get_marlin_kernel_bf16u4`).
- **Resolution:** Enabled `is_bf16` grouped (`group_blocks > 0`) `enable_m_invariant_split_k` for `((prob_n / 64) * 2 <= sms) && (prob_k >= 1024)` in `op_marlin.cu`, guarded `is_valid_config` so channelwise (`group_size == -1`, `prob_n >= 128`) projections require `thread_n >= 128`, and plumbed `is_bf16` through `choose_thread_config` / `cfg128` kernel table lookups.

---

## 2. Code Quality, Modularity, & Reliability Hotspots

### [x] 2.1 Preemptive Modular Splits for Files Approaching 2,000 Lines
All non-vendor `.rs`, `.cu`, and `.cuh` files are now `< 1,850` lines:
- `crates/w34-host/tests/backbone_and_collectives.rs` (`1,987 -> 1,491 lines`) — extracted mixed-batch GDN WY and `TP=2`/`TP=4` collective tests into `tests/collectives_and_gdn_wy.rs`.
- `crates/w34-kernels/csrc/ops/op_gdn_wy.cu` (`1,958 -> 1,375 lines`) — extracted `gdn_wy_chunk64_prefill_f16_kernel` into `csrc/ops/op_gdn_wy_chunk64.cu`.
- `crates/w34-host/tests/tq4_vision_and_moe.rs` (`1,950 -> 1,658 lines`) — moved `moe_grouped_mma_exact_parity_and_tp2` into `tests/moe_sparse.rs`.
- `crates/w34-host/src/ple.rs` (`1,892 -> 1,238 lines`) — extracted CUDA FFI wrappers into `src/ple/gpu_ops.rs`.
- `crates/w34-kernels/csrc/ops/op_moe.cu` (`1,882 -> 1,732 lines`) — extracted dense `FP16` MoE reference kernels into `csrc/ops/op_moe_f16.cu`.
- `crates/w34-host/src/model/rank.rs` (`1,878 -> 1,568 lines`) — extracted post-load runtime staging and tap configuration into `src/model/rank_runtime.rs`.
- `crates/w34-host/tests/qwen4_flash_next.rs` (`1,863 -> 1,528 lines`) — extracted `test_qwen4_exp_reference_bench_suite` into `tests/qwen4_flash_next_bench.rs`.

### [x] 2.2 Per-Kernel Error Attribution in `prepare_execution_plan()` Warmup
- **Locations:** `crates/w34-host/src/cuda/warmup.rs`, `crates/w34-host/src/cuda/warmup_vision_moe.rs`
- **Problem:** Cold-start warmup previously chained FFI launches via bitwise OR (`rc |= unsafe { ffi::w34_...(...) };`), obscuring which FFI entrypoint failed.
- **Resolution:** Replaced every warmup launch in `warmup.rs` and `warmup_vision_moe.rs` with `warmup_ffi!(ffi::w34_...(...))` (`check_rc(rc, stringify!($call))?`) for exact per-kernel error attribution.

### [x] 2.3 Strict Shape & Missing-Tensor Checks in `dump-diff` Oracle Comparisons
- **Locations:** `crates/w34-host/src/cli/oracle.rs`, `crates/w34-host/src/cli/tests.rs`
- **Problem:** In `run_dump_diff`:
  1. `--compare-tp2` used `if let Some((shape2, vals2)) = dump_tp2.tensors_f32.get(name) && shape1 == shape2`, silently skipping any tensor whose shape differed between `TP=1` and `TP=2` (other than `logits`) or that was missing on `TP=1` / `TP=2`.
  2. `--golden` did not report the count of runtime-only diagnostic tensors skipped when absent from the golden archive.
- **Resolution:** Added `compare_tp1_vs_tp2_tensors` enforcing bidirectional non-`logits` tensor presence and exact `shape` + `vals.len()` agreement in `--compare-tp2`, logged `skipped_tensors={skipped}` in `[dump-diff golden]`, and added unit tests in `crates/w34-host/src/cli/tests.rs`.
