# Kernel-Connecting Code Performance Audit & Optimization Plan: `efe00ea`

## Scope and method

Full clean-room performance audit of the **Rust/FFI orchestration, host-device staging, CUDA Graph capture/replay, multi-GPU coordination, MoE/LoRA/PLE dispatch, speculative verification, sampler, and multimodal/decision seams** connecting the CUDA kernels across `crates/w34-host/src/` and `crates/w34-kernels/` at baseline commit `efe00ea`.

### Verified clean foundations (preserved invariants)
- **Zero-cast `FP16` target spine & 256-byte scratch alignment:** All phase buffers in `crates/w34-host/src/model/scratch.rs` (`ALIGN = 256`) and `crates/w34-host/src/cuda.rs` preserve 16-byte (`uint4` / `int4` / `vec8`) alignment across `GemmaRMSNorm`, `SiLU*Mul`, `RoPE`, `LoRA`, and `TP=2`/`TP=4` collectives without intermediate `FP32` activation casts. `DFlash2` drafters remain native `BF16`, and persistent GDN `ssm_state` remains `FP32`.
- **Fused producer-consumer spine:** `QKNorm+RoPE+Gate`, strided `KV` write without `V` materialization, `AttnGate+H64`, `GDNGatedNorm+H64`, `SiLU*Mul+H64`, `Residual+RMSNorm+Tap/H64`, `AllReduce+Residual+RMSNorm+H64`, `AllReduce+HCCombine+BF16Tap`, `PLEDequant+H64`, and `QSASparseAttn+H64` already connect directly without redundant materialization passes.
- **Zero-copy GDN & KV state management:** `MambaRadixCache` restore (`radix.rs:1012-1037`), donation (`radix.rs:1131-1190`), and speculative rollback (`cache.rs:423-442`) use integer `state_slot` indirection and fused in-kernel checkpoint writes rather than `72 MiB` host-driven D2D copies. `TQ4`/`Q8_0` codebooks and Rademacher signs reside in `__constant__` memory with fixed per-layer layouts (`cache.rs:64-84`).
- **Deliberate non-fusions & lock resets:** `const bool fuse_down_h64 = false` in `op_moe.cu:1494, 2060` (`88de5f4`) preserves fine-grained `n_interm/4` and `n_interm/16` `gate_up` CTA grids across `84-128` SMs followed by `~1.1 µs` L2-resident `w34_block_hadamard_inplace_f16`. Marlin `locks` workspaces are self-resetting with zero per-GEMM `cudaMemsetAsync`, and `cudaFuncSetAttribute` uses cached per-device warm masks.

---

## Batch A — Speculative & Sampler Seam Wins

### A1. Speculative GPU verification runs a wasted greedy kernel + D2H + `stream.synchronize()` before GPU rejection sampling
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/proposer/scheduled.rs`, `crates/w34-host/src/proposer.rs`, `crates/w34-host/src/model/batch.rs`, `crates/w34-host/src/model/graphs.rs`, `crates/w34-host/src/model/rank.rs`
- **Problem:** When `use_gpu_sampler == true` (stochastic sampling, repetition/frequency/presence penalties, logit bias, or grammar constraints), all three speculative verification callers invoke `step_batch_slices(...)` instead of `step_batch_slices_for_sampler(..., true)`, and `step_spec_verify_graph` ignores `skip_host_greedy` because `spec_verify_graph` hardcodes greedy token output. Every processed-sampling round launches an unused `248,320`-vocab greedy scan (+ `TP=2` top-1 collective), downloads unused greedy IDs, and **synchronizes all rank streams** (`3T` syncs/round instead of `2T`), blocking CPU grammar/penalty preparation from overlapping target execution.
- **Implemented:** Keyed the 4-entry LRU `spec_verify_graphs` cache by `(m_verify, long_ctx, DecodeGraphSampleMode, GraphLoraMode)` (`PlainGreedy` vs `LogitsOnly`), routed all three speculative verify callers (`proposer/scheduled.rs` single & batched MTP, `proposer.rs`) through `step_batch_slices_for_sampler(..., use_gpu_sampler)`, overlapped CPU sampler preparation with async target forward execution, and synchronized once on rejection output.

### A2. Constrained `MTP K=3` and `DFlash2 M=8/16` overflow 96 KiB pinned sampler staging and regenerate grammar masks 2×–3×
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/model/graph_io.rs`, `crates/w34-host/src/model/sampling.rs`, `crates/w34-host/src/proposer/scheduled.rs`, `crates/w34-host/src/proposer.rs`, `crates/w34-host/src/parser.rs`
- **Problem:** Each `248,320`-vocab bitmask is `31,040 B`. `MTP K=3` (`4` rows = `124,160 B`) and `DFlash2 M=8/16` (`8/16` rows = `248,320 / 496,640 B`) always exceed `PINNED_SAMPLER_BYTES = 96 KiB`, dropping every constrained speculative round (and `B >= 4` constrained decode batch) onto 3-4 pageable `cudaMemcpyAsync` transfers. In addition, `proposer/scheduled.rs` clones `sampler` and computes `K` `31 KiB` grammar bitmasks during draft pre-validation, discards them, and `model/sampling.rs` clones `sampler` again to recompute the exact same `K` masks.
- **Implemented:** Raised `PINNED_SAMPLER_BYTES` to `576 KiB` (`16 × 31,040 B = 496,640 B` masks + sparse penalties + row outputs fit in 1 pinned H2D/D2H), introduced `PreparedSpecSamplerRows::prepare_drafts` + `O(1)` `truncate_drafts` + `verify_speculative_gpu_prepared` / `verify_speculative_gpu_batch_prepared` so grammar bitmasks and sparse penalties are computed once per speculative round, and gated penalty map scans in `parser.rs` behind `any_penalties_active`.

### A3. `TP=2` constrained / penalized greedy sampling gathers `248 KiB/row` of peer logits before checking for `temperature == 0`
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/model/sampling.rs`, `crates/w34-kernels/csrc/ops/op_sampler.cu`, `crates/w34-kernels/csrc/w34_cuda.h`, `crates/w34-kernels/src/ffi_ext.rs`, `crates/w34-host/src/cuda/warmup.rs`
- **Problem:** `run_gpu_sampler_rows` unconditionally executes `comm.gather_vocab_f16` (`248,320 B/row` written to sysmem staging by Rank 1 and read over PCIe by Rank 0; `1.99 MiB` at `M=8`, `3.97 MiB` at `B=16`) whenever `tp_size > 1` before checking whether all rows are greedy (`temperature == 0.0`) without normalized logprobs.
- **Implemented:** Added `w34_sampler_shard_fast_greedy_f16` (`vocab_offset`-aware masked/penalized local `(max_logit, global_id)` reduction per shard, registered in `prepare_execution_plan()` and `check_cuda_sanitizer.py`), enabled up to `32` unbanned sparse penalty/bias entries in `is_row_fast_greedy_masked` (`allow_bans: false` so soft-ban grammar relaxation falls back to full `sampler_step_f16_kernel`), and reduced across ranks via `8 B/row` `tp2_argmax_exchange` (`31,040×` less cross-GPU traffic per row than `gather_vocab_f16`).

### A4. Intermediate speculative prefill chunks run discarded `lm_head` + greedy sampling
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/proposer/scheduled.rs`, `crates/w34-host/src/proposer.rs`
- **Problem:** Non-final chunks of multi-chunk speculative prefills call `step_batch_slices` with `emit_last_row = true`, projecting `lm_head` (`248,320` logits), running greedy sampling, and synchronizing, then discarding the token.
- **Implemented:** Switched speculative chunked prefills to `step_batch_slices_for_sampler_filtered(..., Some(&[is_final_chunk && max_tokens > 0]), use_gpu_sampler && is_final_chunk)`, skipping `lm_head` and greedy sampling on all non-final chunks.

### A5. Single-CTA greedy argmax on small-row plain target & MTP draft steps
- **Status:** `[x]`
- **Locations:** `crates/w34-kernels/csrc/ops/op_sampler.cu`, `crates/w34-host/src/cuda/warmup.rs`, `tools/check_cuda_sanitizer.py`
- **Problem:** Plain target greedy (`top5 == null`) and MTP $K$-step local greedy launch `w34_sampler_greedy_top5_f16<<<num_rows, 1024>>>`, putting only `1` CTA (`4` waves on `1` SM, leaving `83-127` SMs idle) on a `248,320`-element FP16 row at `M=1..4`.
- **Implemented:** Added 16-CTA split-K `sampler_greedy_splitk_stage1_f16_kernel` (`256` threads, `HalfVec8` vectorized loads) + `sampler_greedy_splitk_merge_kernel` in `w34_sampler_greedy_top5_f16` when `top5 == null && num_rows <= 8 && vocab_size >= 4096` using caller-supplied `splitk_scratch` (`8` bytes/split: `(val, id)`), warmed in `prepare_execution_plan()` and verified clean (`0` errors) under `memcheck`, `racecheck`, and `synccheck`.

---

## Batch B — Unified Pinned Metadata, Standalone CLI Parity, & Speculative Graph Coverage

### B1. `BatchedDecodeGraphIo` 160 KiB pinned staging cliff triggers on configured `max_blocks_per_req` pitch even when `bt_unchanged == true`
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/model/graph_io.rs`
- **Problem:** `last_bt_end` multiplies `(B - 1)` by `max_blocks_per_req * 4` (`64 KiB` pitch at `262k` context, `16 KiB` at `64k`) and gates pinned metadata eligibility even when `bt_unchanged == true`. Thus `B >= 4` at `262k` (or `B >= 11` at `64k`) silently loses pinned staging for the `816 B` `meta + masks` upload on *every* decode step, and changed `B=2..3` tables DMA `64 KiB` inter-row holes.
- **Implemented:** Decoupled the fixed `816 B` `meta + masks` pinned upload from `max_blocks_per_req` pitch and added `BatchedDecodeGraphIo::upload_block_tables_slot`, densely packing only dirty `(req_idx, used_blocks)` rows in the remaining `159 KiB` pinned metadata region (`0 B` when `bt_unchanged`) and coalescing contiguous dirty spans when `used_blocks == max_blocks`.

### B2. `step_tokens`, `decode_one_token`, and `generate_tokens` (`w34 generate` & `w34 bench --stage decode`) pay sync-allocations and 8–9 pageable H2Ds per token
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/model/graph_io.rs`, `crates/w34-host/src/model/execution.rs`, `crates/w34-host/src/model/generation.rs`, `crates/w34-host/src/model/forward.rs`
- **Problem:** `step_tokens` allocates and frees 10 temporary `DeviceBuffer`s per rank (10 `cudaMalloc` + zero-memset + `stream_synchronize(0)` + `cudaFree` cycles plus 4 output syncs). `decode_one_token` issues 8-9 separate pageable H2D calls per rank per token (re-uploading unchanged `block_table`s), synchronizes twice on output, and in `generation.rs:195-224` runs greedy decode + sync before overwriting with GPU sampling.
- **Implemented:** Packed `DecodeGraphIo` inputs (`token_id`, `position`, `write_slot`, `seq_len`, `state_in`, `state_out`) and outputs (`out_greedy_id`, `local_max_val`, `out_top5_ids`, `out_top5_logprobs`) into contiguous buffers with `cached_block_table` delta tracking and 1-H2D / 1-D2H pinned helpers, eliminated all 10 per-call `DeviceBuffer::alloc`s in `stage_step_tokens_meta_on_rank` (`batch_meta_arena`), preserved pre-existing `dump_out` tensors across `TP>=2` `step_tokens_with_positions` (`dump_multimodal_prompt`), and routed sampled `generate_tokens` through `step_batch_slices_for_sampler(..., true)` (`logits_only` graph path).

### B3. Batched MTP redundant post-D2D sync, multi-shape `spec_verify_graph` cache, & batched verify graph coverage
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/proposer/scheduled.rs`, `crates/w34-host/src/model/graph_io.rs`, `crates/w34-host/src/model/graphs.rs`, `crates/w34-host/src/model/batch.rs`, `crates/w34-host/src/model/rank.rs`
- **Problem:**
  1. `proposer/scheduled.rs:967-993` calls `rank.stream.synchronize()` across all ranks right after same-stream D2D accepted-feature copies.
  2. `spec_verify_graph` caches only a single `(m_verify, long_ctx)` entry, requires `speculative_logits` capacity `== q_len` instead of `>= q_len`, uploads pageable `SpecVerifyGraphIo` metadata + unchanged `block_table`s every round, and sets engine-wide `use_cuda_graph = false` if one shape fails capture.
  3. Multi-request batched MTP target verification (`B = 2..4`) misses CUDA graphs (`slices.len() == 1` gate) and serializes per-slot draft waits.
- **Implemented:** Removed the redundant post-D2D `rank.stream.synchronize()` calls in `proposer/scheduled.rs`, upgraded `SpecVerifyGraphIo` to packed pinned metadata + cached `block_table` delta upload with a 4-entry LRU `spec_verify_graphs` (`speculative_logits_cap >= q_len`) and per-key `failed_spec_verify_keys` isolation, and added `BatchedSpecVerifyGraphIo` + 8-entry LRU `batched_spec_verify_graphs` capturing uniform `(B in {2, 3, 4}, q_len >= 2, total_tokens <= 24)` batched speculative verify steps (`PlainGreedy` and `LogitsOnly`).

### B4. Pinned staging for MTP `PrefixMemo` hidden-state D2H & eager batch metadata
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/proposer.rs`
- **Problem:** MTP `PrefixMemo` copies `r × 2H` bytes (`10 KiB/tok` on 27B) directly into a pageable `Vec<u8>` on Rank 0's compute stream before proposal launch.
- **Implemented:** Added `256 KiB` `MappedHostBuffer` `memo_pinned_staging` on `NativeProposer` for `MTP`, staging Rank 0 hidden features asynchronously into pinned memory and copying into `memo.features` after the existing proposal stream sync.

---

## Batch C — Tiered-MoE, LoRA Graph Cliffs, & Multi-Rank Dispatch

### C1. Tiered-MoE mapped expert pull launches `5,120` empty CTAs on 0-miss steps, caps GPU-LRU at `M <= 8`, & syncs 6× per dirty layer
- **Status:** `[x]`
- **Locations:** `crates/w34-kernels/csrc/ops/op_moe_router.cu`, `crates/w34-host/src/moe_offload.rs`, `crates/w34-host/src/moe_offload/engine.rs`, `crates/w34-host/src/moe_offload/host_store.rs`, `crates/w34-host/src/moe_cpu_bridge.rs`, `crates/w34-host/src/model/forward.rs`
- **Problem:**
  1. `w34_moe_pull_mapped_experts_with_count` launches `dim3(64, max_misses)` (`640` CTAs at `M=1, K=10`; `5,120` CTAs at `M=8, K=10`) even when `*num_misses == 0`.
  2. `can_execute_device_lru` restricts pure mapped GPU-LRU (`!cache.has_cpu_bridge()`) to `TIERED_MOE_CPU_BRIDGE_MAX_DECODE_TOKENS = 8`, forcing 48 mid-forward `stream.synchronize()` calls + eager fallback at `B=16` (`M=9..16`) even when `M * top_k <= k_hot` and `miss_cap` fits.
  3. `sync_from_device_if_dirty` issues 4 `copy_to_host` calls + 2 explicit `stream.synchronize()` calls (`6` syncs/layer = `288` syncs across 48 layers) for `2.5 KiB` of metadata.
  4. `StreamedPrefill` waits until all GPU cold chunks are enqueued before starting CPU expert evaluation even though activation D2H completes much earlier.
- **Implemented:** Bounded `w34_moe_pull_mapped_experts_with_count` `grid.y` to `min(max_misses, max(1, (multi_processor_count * 2) / 64))` (`2` expert waves = `128` CTAs on 84-SM `sm_86`, `4` waves = `256` CTAs on 128-SM `sm_89`, `40×` fewer CTAs on 0-miss `M=8` steps) with a `gridDim.y` stride loop; added `TIERED_MOE_GPU_LRU_MAX_DECODE_TOKENS = 16` for pure GPU-LRU (`!cache.has_cpu_bridge()`) with `16 * top_k` `gpu_lru_ws` capacity; added `RankTieredMoe::sync_all_dirty_from_device` coalescing all 48 layers into async D2Hs + `1` stream sync (`288 -> 1` sync); and overlapped CPU expert Rayon `prefill_tasks` + pinned staging buffers (`d2h_x_f16`, `h2d_y_f16`) with GPU cold expert pulls in `StreamedPrefill`.

### C2. LoRA `BaseOnly` / `Uniform` / `Mixed` graph keys & dynamic `active_midloop_specs()` graph eligibility
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/model/graph_io.rs`, `crates/w34-host/src/model/graphs.rs`, `crates/w34-host/src/model/batch.rs`, `crates/w34-host/src/model/execution.rs`, `crates/w34-host/src/model/rank.rs`, `crates/w34-host/src/model/rank_runtime.rs`
- **Problem:** When any LoRA adapter is loaded, captured graphs unconditionally bind `d_ids_ptr != 0`, forcing `305-610` early-exiting LoRA kernel launches (and disabling `in_proj_ba` aux-stream overlap) on `lora_id == -1` base-model requests. Loading a `midloop` Decision LoRA (`glide-9b` / `glide-27b`) also disables CUDA graphs globally even for plain `CausalLm` requests where `active_midloop_specs()` is empty.
- **Implemented:** Added `GraphLoraMode::{BaseOnly, Uniform(i32), Mixed}` across `decode_graph` (8-entry LRU), `spec_verify_graphs`, `batched_spec_verify_graphs`, and `batched_decode_graphs` so `BaseOnly` captures 0 LoRA launches + `in_proj_ba` aux-stream overlap, `Uniform(id)` captures homogeneous single-adapter specialization, and `Mixed` binds dynamic `d_ids_ptr`; gated graph eligibility dynamically per `GraphLoraMode` (`supports_cuda_graph_for_tokens_with_lora`) so loading a `midloop` or `noncausal_full_attention` Decision LoRA preserves CUDA Graphs for `BaseOnly` and causal `Uniform(id)` requests.

### C3. Persistent multi-rank eager worker pool, PLE warm-cache `Vec` reuse, & `HostFailures` zero-alloc inherit
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/coordinated.rs`, `crates/w34-host/src/model/engine.rs`, `crates/w34-host/src/model/batch.rs`, `crates/w34-host/src/model/execution.rs`, `crates/w34-host/src/proposer.rs`, `crates/w34-host/src/proposer/scheduled.rs`, `crates/w34-host/src/model/rank.rs`, `crates/w34-host/src/model/rank_runtime.rs`, `crates/w34-host/src/cuda.rs`
- **Problem:** Eager `TP=2`/`TP=4` forward calls spawn fresh OS threads via `std::thread::scope` every step; batched PLE staging allocates 7 scratch `Vec`s per step; and `HostFailures::inherit` clones a `Vec<Arc<...>>` on every GDN aux-stream fork/join event when CPU-bridge slots exist.
- **Implemented:** Built futex-parked (`AtomicU32` + `libc::SYS_futex`) `TpWorkerPool` (`run_tp_closures` / `run_split`) on `Qwen3_5Engine::tp_workers` eliminating per-step OS thread creation across all 10 `TP > 1` target and proposer call sites, reused 7 persistent `ple_scratch_*` `Vec`s on `Qwen3_5ModelRank`, and gave `HostFailures` a lock-free `count == 0` fast path + `Arc<[HostFailureEntry]>` cached snapshot.

---

## Batch D — Multimodal Vision, Decision Serving, & Radix Cleanup

### D1. Persistent `VisionEncoder` workspace, cached GPU `pos_interp`, fused quantized bias, `TP=2` pinned splice, & mixed-decode vision budget
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/vision.rs`, `crates/w34-host/src/vision/linear.rs`, `crates/w34-host/src/scheduled_vision.rs`, `crates/w34-host/src/model/engine.rs`, `crates/w34-host/src/model/rank.rs`, `crates/w34-host/src/model/execution.rs`, `crates/w34-host/src/model/generation.rs`, `crates/w34-host/src/engine_worker/decode.rs`
- **Problem:**
  1. `VisionEncoder::encode` performs 14 `DeviceBuffer::alloc`s (`cudaMalloc` + zero-fill + `stream_synchronize(0)`) and 13 `cudaFree`s per call.
  2. Every encode recomputes `12.6M` CPU software FP64-emulated half-FMAs for `interpolate_pos_embed_and_rot_ids` and uploads `8 MiB` of pageable `pos_interp_f16`.
  3. Quantized `W4A16` vision projections launch standalone `w34_vision_bias_add_f16` (`96` extra launches across 24 blocks) instead of fusing bias into `qkv_rope` / `residual` / `gelu`.
  4. `TP=2` vision splice bounces `5 MiB` of merged embeddings through a pageable `Vec<u16>` + `DeviceBuffer::alloc` per image.
  5. `scheduled_vision.rs` runs discarded `lm_head` + greedy sampling on intermediate prefill chunks, `engine_worker/decode.rs` bypasses the mixed-decode token budget across active vision requests, and multi-question noncausal Decision waves re-encode identical images `Q` (`2Q`) times.
- **Implemented:** Added grow-only `VisionWorkspace` + 8-entry LRU `VisionPosGpuEntry` (`pos_cache` by `grid_thw` list) + pinned `d_patches` H2D and `replicate_to_peers_pinned` (`TP>=2`), added `forward_bias_maybe_defer` deferring quantized vision biases (`qkv`, `proj`, `fc1`, `fc2`, `merger.fc1`) into downstream fused `op_vision.cu` consumers (`97` fewer kernel launches per 24-block ViT encode at `0`-ULP parity; disabled when `trace.is_some()`), added reusable `vision_splice_idx` / `vision_splice_emb` buffers per rank and an 8-entry / `64 MiB` LRU `vision_embed_cache` on `Qwen3_5Engine` (sharing encoded visual embeddings across multi-question Decision waves), added `ScheduledVision::step_prefill_with_emit` suppressing `lm_head` on intermediate chunks and `HiddenPositions`-only requests, and enforced `effective_mixed_prefill_budget()` (with `mamba_chunk_size = 64` alignment) in `step_vision_prefills` when decode requests are active.

### D2. Single-slot GDN reservation for non-checkpointing Decision requests, cold `midloop` state snapshot skip, & batched candidate-logit D2H
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/scheduler.rs`, `crates/w34-host/src/engine_worker/runner.rs`, `crates/w34-host/src/engine_worker/decode.rs`, `crates/w34-host/src/model/forward.rs`, `crates/w34-host/src/model/execution.rs`, `crates/w34-host/src/model/decision.rs`, `crates/w34-host/src/radix.rs`
- **Problem:** Non-checkpointing (`prefix_bidir_len.is_some()`) Decision requests reserve 3 GDN state slots (`active + 2 ping-pong`) instead of 1 (wasting `146.8 MiB/rank` on 27B `TP=2` / `293.6 MiB` on `TP=1`), `midloop` snapshots/restores GDN state even on cold `has_initial_state == 0` inputs and tracks output/checkpoint slots in `last_batch_state_slots`, and `read_candidate_logits` issues one `2 B` pageable `cudaMemcpyAsync` per candidate option and synchronizes per request.
- **Implemented:** Added `SchedulerV1::admit_waiting_by_shareability` reserving `1` GDN slot (`ping_pong_slots = [-1, -1]`) for unshareable `prefix_bidir_len.is_some()` requests (`3×` concurrent Decision capacity per GDN pool), filtered `last_batch_state_slots` to `state_in_indices` with `has_initial_state != 0` so cold `midloop` prefill passes skip `72 MiB/rank` GDN snapshot/restore completely, and added `read_batch_candidate_logits` + pinned `read_norm_hidden_rows_f32` + `capture_batch_decision_taps` coalescing all `CandidateLogits` and `HiddenPositions` taps across a batch into `1` sync per rank.

### D3. `MambaRadixCache` evicted dead-node `Vec` release & non-contiguous `selected_rows` gather
- **Status:** `[x]`
- **Locations:** `crates/w34-host/src/radix.rs`, `crates/w34-host/src/model/forward.rs`
- **Problem:** Evicted radix nodes retain heap `key`/`value` `Vec<i32>` allocations and grow `self.nodes` monotonically until full reset, and non-contiguous `selected_rows` loops `1-2` `cudaMemcpyAsync(D2D)` calls per selected row.
- **Implemented:** Added `retire_dead_node` + `free_nodes` recycling (`alloc_node`) in `MambaRadixCache` so evicted dead nodes immediately drop `key` / `value` / `draft_pages` / `children` heap buffers and reuse arena indices (`nodes.len() <= 8` across `512` 2-checkpoint sessions), and coalesced contiguous `selected_rows` spans in `forward_selected_rows` (`0` D2D copies when single-span contiguous).

---

## Verification & Measured Results (`2× RTX 3090 Ti`, `sm_86`)

- **CUDA Binary & Sanitizer Gates:**
  - `mise run cuda-bin-check`: **`1,054` kernel entries (`sm_86 = 527 == sm_89 = 527`), `LOCAL = 0`, `static_shared <= 48 KiB`, `zero_stack = 910 / 1,054`** (all non-zero frames within the explicit call-frame allowlist).
  - `mise run cuda-sanitize`: **`114/114` `compute-sanitizer --error-exitcode=1` runs (`memcheck` + `racecheck` + `synccheck`) `0` errors** across all CUDA subsystems including new `w34_sampler_greedy_top5_f16` 16-CTA split-K and `w34_sampler_shard_fast_greedy_f16` tests (`split_k_greedy_top1_matches_single_cta`, `shard_fast_greedy_matches_full_sampler_across_edges`, and `tiered_moe_gpu_lru_cpu_hybrid_and_streamed_prefill_parity_tp1_and_ep2`).
- **End-to-End `w34 bench --stage all` & `dump-diff` (`TQ4`, `lossless_parity = true` on all Radix & Vision stages):**
  - **`nchapman/Qwen3.5-4B-W4A16-HGPTQ` (`TP=1`):** `Plain` **`208.33 tok/s`** (`1,171.79 tok/s` `B=8` 2-turn scheduler); `MTP-3` **`589.33 tok/s`** (`1.697 ms` ITL, up from `583.5 tok/s`); `DFlash2` (`M=16`) **`1,574.20 tok/s`** (`0.635 ms` ITL, `815.62 tok/s` `B=8` scheduler, up from `1,288–1,427 tok/s`); Vision `28.66 -> 11.96 ms` (`2.40×` warm speedup).
  - **`nchapman/Qwen3.8-27B-W4A16-HGPTQ` (`TP=2`):** `Plain` **`83.90 tok/s`** (`364.01 tok/s` `B=8` scheduler); `MTP-3` **`216.70 tok/s`** (`4.615 ms` ITL, `290.83 tok/s` `B=8` scheduler); `DFlash2` (`M=8`) **`215.15 tok/s`** (`316.70 tok/s` `B=8` scheduler); Vision `83.06 -> 30.01 ms` (`2.77×`).
  - **`nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ` (`TP=2`, 256-expert resident MoE):** `Plain` **`246.05 tok/s`** (`1,067.30 tok/s` `B=8` scheduler); `MTP-3` **`553.81 tok/s`** (`1.806 ms` ITL, up from `548.6 tok/s`, `698.10 tok/s` `B=8` scheduler); Vision `39.99 -> 16.02 ms` (`2.50×`); `dump-diff --compare-tp2` `match = true` (`draft_exact = 3/3`).
  - **`nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ` (`TP=2`, 512-expert tiered MoE + QSA + PLE + HC):** `Plain` **`50.52 tok/s`**; `MTP-3` **`105.73 tok/s`** (`91.61 tok/s` `B=8` scheduler); `DFlash2` (`M=8`) **`67.28 tok/s`** (up from `62.49 tok/s`, `+7.7%`, `95.24 tok/s` `B=8` scheduler); Vision `588.40 -> 35.21 ms` (`16.7×` warm speedup).
  - **`nchapman/Qwen3.5-9B-W4A16-HGPTQ` + `Heretic-LoRA` (`TP=1`):** `cuda_graph = true` (`GraphLoraMode::Uniform(0)`), **`126.07 tok/s`** (`7.932 ms` ITL).
  - **`Qwen/Qwen3.5-4B` Multimodal `dump-diff --image ... --compare-tp2`:** `287` tensors captured on both `TP=1` and `TP=2`, `target_max_rel_l2 = 0.014557`, `min_cosine = 0.999225`, `tp1_top1 = 357 == tp2_top1 = 357` (`match = true`).
