# warp-34 — Backlog & Hotspot Tracker

Prioritized performance, modularity, and reliability hotspots identified across `crates/w34-host` and `crates/w34-kernels`.

---

## 1. Performance Hotspots

### [x] 1.1 Adaptive Post-Drafter `K_hot` Expansion for Tiered-MoE (`Qwen3.8-Flash-Next`)
- **Locations:** `crates/w34-host/src/model/rank.rs:416–425`, `crates/w34-host/src/moe_offload/host_store.rs`
- **Problem:** `plan_tiered_moe_hot_per_layer` capped `K_hot` at `TIERED_MOE_TP2_DEFAULT_HOT = 112` on `TP=2` (`96` on `TP=1`) during `Qwen3_5Engine::load` to reserve VRAM for a potential `MTP` drafter (`~450 MiB`) or large KV pools. Even after loading `MTP` with `kv_pages = 640` on 2× RTX 3090 Ti, `1.49 GiB` of VRAM per GPU remained unused; with `DFlash2-W4A16-HGPTQ` (`~220 MiB` total footprint and `kv_pages = 256`), `~2.0 GiB` per GPU remained unused. Meanwhile, `DFlash2` (`M=8`, `80` `(token, expert)` pairs/layer) was bottlenecked by PCIe zero-copy cold-expert pulls (`~1.99 MiB` per expert miss).
- **Resolution:** Raised `TIERED_MOE_TP2_DEFAULT_HOT` from `112` to `120` (`240/512` experts across 2 GPUs, using `765 MiB` of the `1.49 GiB` idle VRAM per GPU while retaining `0.64–0.67 GiB` headroom after loading `MTP` with `kv_pages = 640`). Verified on 2× RTX 3090 Ti: `TP=2` Warm Target 64-tok `46.66 -> 65.81 tok/s` (`+41.0%`), `MTP-3` Warm 64-tok `85.08 -> 113.79 tok/s` (`+33.7%`), `DFlash2` Warm 64-tok `137.25 -> 152.18 tok/s` (`+10.9%`), `short-chat 128:512 WARM` `77.63 -> 91.36 tok/s` (`+17.7%`), `chat-512` 5-prompt mean `76.74 -> 81.77 tok/s` (`+6.6%`), `4096:256` prefill `1511.0 -> 1665.0 tok/s` (`+10.2%`) & decode `66.85 -> 72.88 tok/s` (`+9.0%`), `8192:1024` decode `70.00 -> 73.61 tok/s` (`+5.2%`).

### [ ] 1.2 `TP=2 / EP=2` CPU MoE Bridge for `StreamedPrefill`
- **Locations:** `crates/w34-host/src/moe_offload/engine.rs:542`, `crates/w34-host/src/moe_offload/engine.rs:1025`, `crates/w34-host/src/moe_cpu_bridge.rs`
- **Problem:** On `TP=1`, `StreamedPrefill` offloads sparse cold experts (`1..=8` or `1..=96` routed tokens) to the SIMD CPU worker pool (`cache.bridge_runtime`) in parallel with GPU Pass 0 + PCIe H2D streaming. On `TP=2 / EP=2`, `register_cpu_bridge` is gated on `ep_size == 1`, so 100% of cold expert misses in 4K–8K prefill (`bench.py 4096:256` at `1511 tok/s`, `8192:1024` at `1646 tok/s`) stream over PCIe H2D DMA.
- **Plan:**
  1. Partition physical CPU worker cores across EP ranks (`cores / ep_size` per rank, e.g. 8 physical cores per rank on a 16-core host) or share a single thread pool keyed by `(ep_rank, layer_id)`.
  2. Enable `StreamedPrefill` hybrid CPU cold-expert evaluation for `ep_size == 2` so DDR5 CPU evaluation overlaps with per-rank PCIe H2D DMA during long prefills.

### [ ] 1.3 Hybrid Bucketed Dispatch for `DFlash2` (`M=8`) Sparse MoE Verify
- **Locations:** `crates/w34-kernels/csrc/ops/op_moe.cu:1710–1750`, `crates/w34-kernels/csrc/ops/moe_sparse_inline.cuh`
- **Problem:** For `Qwen3.8-Flash-Next` + `DFlash2` (`M=8`, `total_pairs = 80` across 96–112 hot slots), `w34_moe_w4a16_mlp_ex_f16` dispatches to `regime = 2` (`moe_sparse_mma_kernel`, `16×16×16` MMA tile). Because the average touched expert has only ~1.4 routed tokens, `14 / 16` rows in each `m16n8k16` tile are wasted zero-padding, while pure `regime = 1` (`moe_sparse_vec4_kernel`) is slower (`116.5` vs `137.3 tok/s`) because it reloads weights for the top static experts hit by 3–6 tokens.
- **Plan:**
  1. In `moe_build_expert_offsets_kernel`, partition active experts into a dense bucket (`count >= 3`) and a singleton/pair list (`count <= 2`), or specialize a compact warp-per-expert `vec4` path for `count <= 2` alongside `m16` MMA for `count >= 3`.
  2. Benchmark on `Qwen3.8-Flash-Next` + `DFlash2` (`M=8`) and `Qwen3.6-35B-A3B` + `MTP-3` (`M=4`).

### [x] 1.4 Small-`M` (`M <= 16`) Tile & Split-K Tuning for `BF16` Marlin `u4b8`
- **Locations:** `crates/w34-kernels/csrc/ops/op_marlin.cu`, `crates/w34-kernels/csrc/marlin/marlin_template.h`
- **Problem:** `w34_marlin_gemm_bf16_u4b8` (used by `Qwen3.5-27B-DFlash2-W4A16-HGPTQ` and `Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ` at `M=8` and `M=16`) skipped `enable_m_invariant_split_k` for `N <= 4096, K >= 1024` (leaving `{128, 64, 128}` on 1 slice for `N = 2560, K = 2048` `o_proj` and `choose_thread_config` checking `get_marlin_kernel_f16` instead of `get_marlin_kernel_bf16u4`).
- **Resolution:** Enabled `is_bf16` grouped (`group_blocks > 0`) `enable_m_invariant_split_k` for `((prob_n / 64) * 2 <= sms) && (prob_k >= 1024)` in `op_marlin.cu`, guarded `is_valid_config` so channelwise (`group_size == -1`, `prob_n >= 128`) projections require `thread_n >= 128`, and plumbed `is_bf16` through `choose_thread_config` / `cfg128` kernel table lookups.

---

## 2. Code Quality, Modularity, & Reliability Hotspots

### [ ] 2.1 Preemptive Modular Splits for Files Approaching 2,000 Lines
Keep all `.rs`, `.cu`, and `.cuh` files comfortably below the 2,000-line ceiling before adding new features:
- `crates/w34-host/tests/backbone_and_collectives.rs` (**1,987 lines**) — split GDN WY/conv parity tests into `tests/gdn_backbone.rs`.
- `crates/w34-kernels/csrc/ops/op_gdn_wy.cu` (**1,958 lines**) — extract WY chunk-prefill helpers into `csrc/ops/gdn_wy_inline.cuh`.
- `crates/w34-host/tests/tq4_vision_and_moe.rs` (**1,950 lines**) — split TQ4 paged-attention tests from Vision/MoE integration tests.
- `crates/w34-host/src/ple.rs` (**1,892 lines**) — split `PleMmapTable` / CPU hashing (`ple/mmap.rs`) from CUDA wrappers (`ple/gpu_ops.rs`).
- `crates/w34-kernels/csrc/ops/op_moe.cu` (**1,882 lines**) — move dense/unquantized `f16` MoE reference kernels or tiered unpack/pull kernels into a dedicated `.cu` translation unit.
- `crates/w34-host/src/model/rank.rs` (**1,878 lines**) — extract VRAM capacity & scratch sizing estimators into `model/capacity.rs`.
- `crates/w34-host/tests/qwen4_flash_next.rs` (**1,863 lines**) — split `test_qwen4_exp_reference_bench_suite` into `tests/qwen4_flash_next_bench.rs`.
- `crates/w34-kernels/csrc/ops/op_collectives.cu` (**1,845 lines**) — extract `TP=4` ring/P2P collectives into `csrc/ops/op_collectives_tp4.cu`.

### [ ] 2.2 Per-Kernel Error Attribution in `prepare_execution_plan()` Warmup
- **Locations:** `crates/w34-host/src/cuda/warmup.rs`, `crates/w34-host/src/cuda/warmup_vision_moe.rs`
- **Problem:** Cold-start warmup currently chains dozens of FFI launches via bitwise OR (`rc |= unsafe { ffi::w34_...(...) };`) before calling `check_rc(rc, "prepare_execution_plan ...")`. If a kernel fails during cold-start warmup, `rc |= ...` corrupts the `cudaError_t` integer (`1 | 2 = 3`) and obscures which FFI entrypoint failed.
- **Plan:** Replace `rc |= ...` accumulation with per-call `check_rc(unsafe { ... }, "<ffi_fn_name>")?` (via a concise macro) so any warmup failure immediately reports the exact kernel name and genuine CUDA error code.

### [ ] 2.3 Strict Shape & Missing-Tensor Checks in `dump-diff` Oracle Comparisons
- **Locations:** `crates/w34-host/src/cli/oracle.rs:647`, `crates/w34-host/src/cli/oracle.rs:701`
- **Problem:** In `run_dump_diff`:
  1. `--compare-tp2` uses `if let Some((shape2, vals2)) = dump_tp2.tensors_f32.get(name) && shape1 == shape2`, silently skipping any tensor whose shape differs between `TP=1` and `TP=2` (other than `logits`) or that is missing on `TP=2`.
  2. `--golden` silently skips tensors in `dump.tensors_f32` that are absent from the golden archive without reporting skipped tensor names.
- **Plan:**
  1. In `--compare-tp2`, return an explicit `Err` if a non-`logits` tensor in `dump.tensors_f32` is missing from `dump_tp2.tensors_f32` or has `shape1 != shape2`.
  2. In `--golden`, fail if a matched golden tensor has a length mismatch with `shape`, and log the count/names of any runtime-only diagnostic tensors not present in the golden archive.
