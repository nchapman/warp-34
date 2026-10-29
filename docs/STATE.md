# warp-34 — Session Orientation & Current State

## Status: Phases 0–4 complete; Phase 4 correctness gates verified

`warp-34` has completed **Phase 0** (Foundational Seam Contracts, Oracle & `nsys` Diff Harness, CUDA Runtime & `CapacityPlanner`, Unified Safetensors Loader), **Phase 1** (Donor-Aligned Kernel Suite + Decisive Gate Tests #1, #2, #4), **Phase 2** (Qwen 3.5 / 3.8 Hybrid Backbone on `TP=1` and `TP=2`, `KvPool`, `StateBundlePool`, `CUDAGraphRunner` Gate #3, and `w34` CLI), and **Phase 3** (`SchedulerV1` token-budget continuous batching, `SGLang`-donor `MambaRadixCache` + `HybridReqToTokenPool` `extra_buffer` ping-pong GDN checkpoint donation passing **Decisive Gate Test #5**, `SamplerV1` + structured-output bitmask + `Qwen3Parser`, and OpenAI HTTP/1.1 + SSE server `w34 serve` / `w34 chat`).

All 5 Decisive Pre-Freeze Gate Tests (**Gate #1–#5**) pass in `crates/w34-host/tests/phase0_1_2_gates.rs` and `crates/w34-host/tests/phase3_gates.rs`. All 4 dense target model configurations (`4B TP=1`, `9B TP=1`, `9B TP=2`, `27B TP=2`) pass `w34 greedy-diff`, `w34 dump-diff`, and `w34 profile-diff`, **beating `vLLM 0.31.0` by +31% to +83% in single-stream decode throughput** on 1–2× RTX 3090 Ti (`sm_86`) before speculative decoding or `TQ4` compression, and achieving a **95.2% warm prefix TTFT drop** (`145.68 ms` → `7.05 ms`) via `MambaRadixCache`.

---

## Verified Benchmark & Oracle Parity Scoreboard (`128` Decode Tokens, `CUDA_GRAPH=1`)

| Model | TP | `vLLM 0.31.0` Decode | `warp-34` Decode (`ITL`) | Speedup vs `vLLM` | `greedy-diff` (`12` Prompts) | `nsys` `profile-diff` Speedup |
|---|---:|---:|---:|---:|---|---:|
| `Qwen3.5-4B-W4A16-GPTQ` | `TP=1` | `134.8 tok/s` (`7.42 ms`) | **`183.84 tok/s`** (`5.44 ms`) | **`+36.4%`** | `first_step=12/12`, `top5=12/12` (`1397/1536`, **91.0% exact**, 10/12 prompts `128/128`) | **`1.30x`** (`5.48 ms` vs `7.11 ms/r`) |
| `Qwen3.5-9B-W4A16-GPTQ` | `TP=1` | `91.2 tok/s` (`10.96 ms`) | **`119.49 tok/s`** (`8.37 ms`) | **`+31.0%`** | `first_step=12/12`, `top5=12/12` (**`1536/1536`, 100.0% exact** across all 12 prompts) | **`1.28x`** (`8.35 ms` vs `10.67 ms/r`) |
| `Qwen3.5-9B-W4A16-GPTQ` | `TP=2` | `98.5 tok/s` (`10.15 ms`) | **`180.41 tok/s`** (`5.54 ms`) | **`+83.2%`** | `first_step=12/12`, `top5=12/12` (**`1536/1536`, 100.0% exact**, bit-for-bit identical to `TP=1`) | **`1.36x`** (`5.21 ms` vs `7.10 ms/r`) |
| `Qwen3.8-27B-W4A16-GPTQ` | `TP=2` | `54.1 tok/s` (`18.48 ms`) | **`77.20 tok/s`** (`12.95 ms`) | **`+42.7%`** | `first_step=12/12`, `top5=12/12` (`673/1536`, `chat00` `128/128` exact) | **`1.21x`** (`12.40 ms` vs `15.05 ms/r`) |

### `nsys` Category Breakdown (`w34 profile-diff` vs `vLLM 0.31.0`, `ms/round`)

| Category | `4B TP=1` (`w34` / `vLLM`) | `9B TP=1` (`w34` / `vLLM`) | `9B TP=2` (`w34` / `vLLM`) | `27B TP=2` (`w34` / `vLLM`) |
|---|---:|---:|---:|---:|
| `marlin_gemm` | `3.58` / `4.31` (`1.20x`) | `6.14` / `7.34` (`1.20x`) | `3.21` / `4.12` (`1.28x`) | `8.04` / `9.42` (`1.17x`) |
| `gdn_recurrent` | `0.81` / `1.12` (`1.38x`) | `0.83` / `1.28` (`1.54x`) | `0.50` / `0.89` (`1.79x`) | `1.04` / `1.84` (`1.77x`) |
| `gdn_conv` | `0.15` / `0.38` (`2.56x`) | `0.18` / `0.42` (`2.37x`) | `0.11` / `0.31` (`2.70x`) | `0.24` / `0.62` (`2.53x`) |
| `full_attn` | `0.21` / `0.41` (`1.91x`) | `0.23` / `0.51` (`2.21x`) | `0.20` / `0.39` (`1.96x`) | `0.41` / `0.78` (`1.91x`) |
| `norms_rope_act` | `0.53` / `0.62` (`1.16x`) | `0.77` / `0.81` (`1.06x`) | `0.62` / `0.64` (`1.03x`) | `1.37` / `1.39` (`1.01x`) |
| `collectives` | `0.00` / `0.00` | `0.00` / `0.00` | `0.47` / `0.58` (`1.23x`) | `1.19` / `0.79` |
| `sampler` | `0.11` / `0.19` (`1.74x`) | `0.11` / `0.21` (`1.92x`) | `0.06` / `0.12` (`1.85x`) | `0.06` / `0.14` (`2.16x`) |
| **TOTAL GPU** | **`5.48` / `7.11` (`1.30x`)** | **`8.35` / `10.67` (`1.28x`)** | **`5.21` / `7.10` (`1.36x`)** | **`12.40` / `15.05` (`1.21x`)** |

---

## Completed Deliverables

1. **Phase 0.1 — Seam Contracts (`docs/contracts/`):**
   - `execution-contract.md`, `cache-layout.md`, `state-transaction.md`, `collectives.md`, `precision-policy.md`.
2. **Phase 0.2 — Oracle & `nsys` Diff Harness (`tools/oracle/`):**
   - `vllm_dump.py`, `sglang_dump.py`, `vllm_greedy.py`, `profile_diff.py`, `prompts.json`, and `goldens/` (`op_fixtures.safetensors`, `sglang_dflash2_fixtures.safetensors`, `oracle_baselines.json`, `adjudications.json`, and 12-prompt greedy references for `4B`, `9B`, `27B`, and `4B+DFlash2`).
3. **Phase 0.3 & 0.4 — CUDA Runtime & Multi-Format Safetensors Loader (`crates/w34-host/src/{cuda,loader}.rs`):**
   - `CudaDevice`, `CudaStream`, `CudaEvent`, `CudaGraph`, `CublasHandle`, `DeviceArena`, `CapacityPlanner` (`MIN_HEADROOM_BYTES = 1 GiB`), `prepare_execution_plan()`, and `SafetensorsArchive` / `ModelConfig` validated across all 5 cached HF checkpoints.
4. **Phase 1 — Donor-Aligned CUDA Kernel Suite (`crates/w34-kernels/csrc/ops/`):**
   - `op_marlin.cu`: `vLLM` Marlin `W4A16` (`U4B8`) + `W8A16` (`U8B128`) repack (0-ULP bit-exact) and GEMM kernels, plus fused `qkv_proj` / `in_proj_qkvz` / `in_proj_ba` column-Concatenation with zero-copy `M=1` pointer aliasing and `w34_split_cols_{2,3}_f16` for `M>1`.
   - `op_norms_rope_act.cu`: Register-cached `GemmaRMSNorm` (`w + 1.0f` in `FP32`), `fused_add_gemma_rms_norm`, `RMSNormGated`, `silu_and_mul`, `apply_attn_output_gate`, and `fused_qk_rmsnorm_rope_gate` (partial 25% 3D mRoPE `[11, 11, 10]`).
   - `op_paged_attn.cu`: Descriptor-driven `kv_cache_write` and 4-warp stride-4 warp-shuffle `paged_attention_f16` supporting `FP16`, `BF16`, `TQ4` (`132 B` rows, Rademacher+WHT-256 + Beta(256) codebook), and `Q8_0` (`K8V4`), plus noncausal query-relative left-window mask for `DFlash2` (**Gate #1** 0-ULP page-table permutation invariance verified).
   - `op_gdn.cu`: `vllm.cpp`-derived `gdn_causal_conv1d_f16` and `gdn_recurrent_f16` with `extra_buffer` ping-pong `spec_state_indices` and `checkpoint_dst_indices` (**Gate #2** 0-ULP continuation and speculative rejection state parity verified).
   - `op_collectives.cu`: Pinned-sysmem two-hop `TP=2` collectives (`all_reduce_f16`, `all_reduce_bf16`, `all_reduce_f32`, `broadcast_i32`, `all_gather_top1_f32_i32`) with `.sys` release/acquire fences (**Gate #4** verified, including `INT32` token IDs `> 65504`).
   - `op_sampler.cu`: 1024-thread GPU `sampler_greedy_top5_f16_kernel` (`vocab_size = 248320`, sharded `[124160, 124160]` under `TP=2`).
5. **Phase 2 — Qwen 3.5 / 3.8 Hybrid Backbone & CLI (`crates/w34-host/src/{cache,model,main}.rs`):**
   - `KvPool`, `StateBundlePool`, `StateTx`, `Qwen3_5Engine` (`TP=1` and `TP=2`), `CUDAGraphRunner` (**Gate #3** bit-exact graph replay under metadata mutation verified), and `w34` CLI (`generate`, `bench`, `dump-diff`, `greedy-diff`, `profile-diff`).
6. **Phase 3 — Continuous Batching Scheduler, `MambaRadixCache` (`SGLang` Donor), `Qwen3Parser`, & OpenAI HTTP/SSE Server (`crates/w34-host/src/{radix,scheduler,parser,server}.rs`):**
   - `MambaRadixCache` + `ReqRadixState` (`crates/w34-host/src/radix.rs`): Dual-lock (`full_lock_ref >= mamba_lock_ref`) and dual-LRU (`full_lru` + `mamba_lru`) radix tree over `P = 16` logical KV pages and `C = 64` GDN `StateBundle` checkpoints with tombstone cleanup, child-split invariants, and `extra_buffer` (`ping_pong_slots: [i32; 2]`) checkpoint donation.
   - **Decisive Gate Test #5 (`test_gate5_warm_prefix_hit_speculation_and_cross_boundary_rejection`)**: Verified 0-ULP `conv_state` (`max_abs_diff = 0.0`) and `ssm_state` (`max_abs_diff = 0.0`) parity and 100% token parity against a cold non-speculative reference across warm `C = 64` prefix restore, `1 + k` speculative verification, cross-boundary (`t = 128`) rejection + retry, and checkpoint donation.
   - **Warm Prefix TTFT Drop**: Verified **95.2% warm TTFT reduction** (`145.68 ms` cold → `7.05 ms` warm, `192` cached prefix tokens) with 100% generated token parity.
   - `SchedulerV1` (`crates/w34-host/src/scheduler.rs`): Token-budget continuous batching (`max_num_batched_tokens`, `max_num_seqs`, `prefill_chunk_size = 64`) with decode-priority scheduling, chunked suffix prefill, and multi-request `step_batch_slices` execution.
   - `SamplerV1` + `TokenBitmaskConstraint` + `Qwen3Parser` (`crates/w34-host/src/parser.rs`): 9-step sampling chain with speculative rollback parity, plus streaming/complete `<think>...</think>` reasoning and `<tool_call>` (Qwen3-XML `<function=...><parameter=...>` and Hermes JSON) extraction.
   - OpenAI HTTP/1.1 + SSE Server (`crates/w34-host/src/server.rs`, `w34 serve` & `w34 chat`): `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`, `/health`, and `/metrics` (Prometheus exposition format including `w34_radix_cache_hit_rate`).

---

## Phase 4 — Verified & Hardware-Aligned (`sm_86`)

All 7 Phase 4 integration gates (`phase4_gates.rs`) pass in **32.64 s**, and the
full `mise run check` suite (formatting, CPU/CUDA clippy, and all 53 release
tests across Phases 0–4) passes in **56.43 s**.

### Verified Speculative & Continuous Batching Throughput (`1–2× RTX 3090 Ti`)

| Workload | Baseline (Plain `CUDA Graph`) | `Qwen3_5MTP` (`k=3`) | `DFlash2` (`M=16` on 4B, `M=8` on 27B) | Speedup vs Plain |
| --- | --- | --- | --- | --- |
| `Qwen3.5-4B-W4A16-GPTQ` (`TP=1`, `w34 bench`, `gen=128`) | `185.64 tok/s` (`5.39 ms`) | **`330.36 tok/s`** (`3.03 ms`, $\tau=3.97\times$, `12.01 ms/r`) | **`991.37 tok/s`** (`1.01 ms`, $\tau=12.70\times$, `12.81 ms/r`) | **`1.78×` (`MTP`) / `5.34×` (`DFlash2`)** |
| `Qwen3.5-4B-W4A16-GPTQ` (`TP=2`, `w34 bench`, `gen=128`) | `253.62 tok/s` (`3.94 ms`) | **`465.86 tok/s`** (`2.15 ms`, $\tau=3.97\times$, `8.52 ms/r`) | **`1,122.84 tok/s`** (`0.89 ms`, $\tau=12.70\times$, `11.31 ms/r`) | **`1.84×` (`MTP`) / `4.43×` (`DFlash2`)** |
| `Qwen3.5-4B-W4A16-GPTQ` (`TP=2`, `w34 bench --stage radix`, `128` cached) | Cold `40.91 ms` → Warm **`5.63 ms`** (`7.26×`, `-86.2%`) | Warm **`472.02 tok/s`** (`5.63 ms` TTFT) | Warm **`1,027.84 tok/s`** (`5.46 ms` TTFT, `7.32×`, `-86.3%`) | **100% lossless warm parity** |
| `Qwen3.5-4B-W4A16-GPTQ` (`w34 bench --stage scheduler`, `B=8`) | `TP=1`: **`644.18 tok/s` decode (`2,254.62 tok/s` total)** / `TP=2`: **`800.33 tok/s` decode (`2,801.15 tok/s` total)** | — | — | **`3.47×` (`TP=1`) / `3.16×` (`TP=2`) batch scaling** |
| `Qwen3.5-9B-W4A16-GPTQ` (`w34 bench --stage scheduler`, `B=8`) | `TP=1`: **`422.54 tok/s` decode (`1,478.87 tok/s` total)** / `TP=2`: **`556.77 tok/s` decode (`1,948.69 tok/s` total)** | — | — | **`3.57×` (`TP=1`) / `3.17×` (`TP=2`) batch scaling** |
| `Qwen3.8-27B-W4A16-GPTQ` (`w34 bench --stage scheduler`, `B=8`) | `TP=1`: **`168.72 tok/s` decode (`590.54 tok/s` total)** / `TP=2`: **`230.46 tok/s` decode (`806.59 tok/s` total)** | — | — | **`3.51×` (`TP=1`) / `3.13×` (`TP=2`) batch scaling** |
| `Qwen3.8-27B-W4A16-GPTQ` (`TP=2`, `w34 bench`, `gen=128`) | `73.65 tok/s` (`13.58 ms`) | — | **`108.11 tok/s`** (`9.25 ms`, **`24.47 ms/r`** vs `SGLang` `24.90 ms/r`, Warm **`218.62 tok/s`**) | **`1.47×` cold / `2.97×` warm** |
| `Qwen3.8-27B-W4A16-GPTQ` (`TP=2`, `w34 generate` coding prompt, `gen=128`) | `75.90 tok/s` (`13.18 ms`) | — | **`151.63 tok/s`** (`6.59 ms`) | **`2.00×`** |

### End-to-End Multi-Stage Verification & Profiling Tooling (`w34` CLI & `.mise.toml`)

- **`w34 dump-diff` (`mise run oracle-dump`):** Captures target layer hidden states, `last_final_hidden_f16`, `tap_aux_bf16`, `drafter_hidden`, and `draft_proposal_ids` across plain, `--speculative mtp`, and `--speculative dflash2` runs; reports `max_abs_diff`, `rel_l2`, and `cosine_sim` against golden safetensors or `--compare-tp2` (`TP=1 vs TP=2` target `max_rel_l2 = 0.0023`, `MTP` drafter `rel_l2 = 0.0020` with `3/3` exact draft match, `DFlash2` drafter `cosine = 0.9849` with `9/15` prefix and `12/15` exact draft match) and fails with non-zero exit code on regression.
- **`w34 greedy-diff` (`mise run oracle-greedy`):** Reports per-prompt and aggregate speculative acceptance telemetry (`rounds`, `accepted`, $\tau = 1 + \bar{a}$), supports `--verify-lossless` (verifying 100% bit-for-bit token identity between speculative and plain target greedy) and `--verify-radix-warm` (verifying 100% bit-for-bit token identity between cold and warm `MambaRadixCache` execution), and fails with non-zero exit code on any adjudication/lossless/radix failure.
- **`w34 bench --stage decode|radix|scheduler|all` (`mise run bench-all`):** Benchmarks Stage 2 single-sequence decode + Stage 4 speculative telemetry (`verify_rounds`, `mean_accepted_drafts`, `mean_tokens_per_round`, `ms_per_verify_round`), Stage 3 `MambaRadixCache` cold-vs-warm TTFT speedup + proposer KV restore + lossless parity, and Stage 3 `SchedulerV1` continuous batching (`--batch-size`, with live `sched.radix` LRU eviction on page/state pool pressure).
- **`w34 profile-diff --live` (`mise run profile-live`):** Runs `nsys profile --trace=cuda --cuda-graph-trace=node`, exports SQLite, slices the exact measured verification rounds from `w34 bench` JSON telemetry, classifies target vs `MTP`/`DFlash2` (`BF16`) proposer spans into the 7 canonical categories (`marlin_gemm`, `full_attn`, `gdn_ssm`, `norms_rope_act`, `tp_comm`, `spec_decode`, `host_gap`), and renders the side-by-side scoreboard against `vLLM 0.31.0` and `SGLang v0.5.19` presets in `tools/oracle/goldens/oracle_baselines.json`.

7. **Phase 4 — Speculative Decoding (`RejectionSampler`, Native `Qwen3_5MTP`, & Native `BF16` `DFlash2`):**
   - **Host & GPU `RejectionSampler` (`src/speculative.rs`, `csrc/ops/op_rejection.cu`):** Consecutive greedy and canonical stochastic residual (`min(1, p/q)`, positive-residual recovery) verification over `[anchor, draft_1, ..., draft_k]`, typed `INT32` outputs, padded-request masking, `processed_len` vs `emitted_len` accounting, and atomic `StateTx` + `SamplerV1` + `TokenBitmaskConstraint` tentative-history rollback (`phase4_rejection` 10 tests, `phase4_gpu_rejection` 2 tests).
   - **Native `Qwen3_5MTP` (`src/mtp.rs`):** `vLLM 0.31.0` `qwen3_5_mtp.py` donor (`pre_fc_norm_hidden`, `pre_fc_norm_embedding`, replicated `fc: 2H -> H`, TP-sharded gated full attention in `W34_KV_NAMESPACE_MTP_FULL_ATTN` + SwiGLU MLP + final `GemmaRMSNorm` + shared `embed_tokens`/`lm_head`) with batched committed-row prefill ingest (`sample_last = false` skips `lm_head`/sampler during ingest), zero-copy `M=1` pointer aliasing for `fc` input and `[QGate, K, V]` projection, pre-uploaded KV page table, and packed single-upload batch metadata (`worst max_abs = 0.03125`, `rel_l2 = 0.001158` across all 7 live rows of `tools/oracle/goldens/vllm_mtp_4b/call_{0,1,2}.safetensors`, `phase4_mtp` 2 tests).
   - **Native `BF16` `DFlash2` (`src/dflash.rs`, `csrc/ops/op_dflash2.cu`, `csrc/ops/op_paged_attn.cu`):** `SGLang v0.5.19` `DFlash2DraftModel` / `DFlashWorkerV2` donor in native `BF16` (`4B M=16` taps `[1, 9, 16, 23, 28]` and `27B M=8` taps `[5, 19, 33, 47, 61]` with `window=2048`), direct `FP16 -> BF16` local per-rank `TapPlan` capture (`w34_tap_add_f16_to_bf16`, zero `FP32` intermediate), `W34_KV_NAMESPACE_DFLASH_DRAFT_CTX` `BF16` paged KV with strided packed `[K,V]`/`[Q,K,V]` cache writes (`w34_kv_cache_write_strided_bf16`) and fused contiguous `Q` output (`w34_dflash_qk_norm_rope_bf16_f32_table_qout`, eliminating all 25 per-round `w34_copy_cols_bf16` launches), 1-warp (32-thread) `RoPE` and selector edge-scoring reductions, bank-conflict-free transposed-smem `w34_topk_rows_f16`, and FlashInfer `sm_86` FA2 `BF16` paged attention (`CTA_TILE_KV=16` at HD=256 / `64` at HD=128, register-preloaded `Q` fragments, sliding-window `start_base` skip). Passes **bit-exact (`relL2 = 0.0, maxabs = 0.0`)** full-model 5-layer hidden parity (`actual_sglang_full_model_hidden_parity` in `phase4_dflash.rs`), bit-exact 5-layer attention parity (`phase4_bf16_attention.rs`), and `sglang_dflash2_fixtures.safetensors` operator dump-diff parity (`phase4_gates.rs`, `phase4_bf16_ops.rs`).
   - **Speculative Engine, Batched Target Verification, Warm Prefix Restore, & CLI/Server Integration (`src/proposer.rs`, `src/model.rs`, `src/server.rs`, `src/main.rs`):** Batched `lm_head` + `w34_sampler_greedy_top5_f16` + `all_gather_top1_f32_i32` across all `1 + k` verification rows in `forward_selected_rows`, zero-copy row-strided `QKV` / `QKVZ` / `BA` consumption (`w34_fused_qk_rmsnorm_rope_gate_strided_f16`, `w34_kv_cache_write_strided`, `w34_gdn_causal_conv1d_strided_f16`, `w34_gdn_recurrent_strided_f16`, `w34_rms_norm_gated_silu_strided_f16`, eliminating all `112` per-round `w34_split_cols_{2,3}_f16` launches on 27B and `56` on 4B), batch-size-invariant 1-launch warp-reduction `w34_small_n_gemm_f16` for GDN `in_proj_ba` (eliminating `720` per-round cuBLAS split-K GEMV launches on 27B while preserving 0-ULP `ssm_state` parity across all `M`), `float4` vectorized `op_gdn.cu` `ssm_state` loads/stores with deduplicated `last_spec_slot == slot_out` final writes (`gdn_ssm = 1.34 ms/round` on `27B TP=2 + DFlash2`, **`1.75×` faster than `SGLang v0.5.19`**), persistent `batch_meta_arena` (zero per-step `cudaMalloc`/`cudaFree`), direct `last_final_hidden_f16` write from final `GemmaRMSNorm`, pipelined `DFlash2Rank::ingest_no_sync` + `forward` in a single `TP=2` `std::thread::scope` (`ingest_and_propose`), and zero-D2H GPU-resident `DFlash2` KV truncation on warm `MambaRadixCache` hits. All seven integration gates in `crates/w34-host/tests/phase4_gates.rs` pass together (`4B TP=1` & `TP=2` `MTP`/`DFlash2`, `27B TP=2` `M=8` `DFlash2`, processed stochastic/grammar, HTTP JSON/SSE, and `sglang_dflash2_fixtures.safetensors` dump-diff parity).

## Phase 4 Original Entry Point (`docs/ROADMAP.md` §4 Phase 4)

Start **Phase 4** (Speculative Decoding: `DFlash2` Parallel Block Diffusion (`M = 16` on 4B, `M = 8` on 27B) & Native `Qwen3_5MTP`):
- Phase 4.1: Native `Qwen3_5MTP` 1-layer speculative drafter (`fc: 2H -> H`, shared embedding + `lm_head`, `K = 1..3`).
- Phase 4.2: cached checkpoints are `nchapman/Qwen3.5-4B-DFlash2` in native BF16
  for 4B `M=16` (taps `[1,9,16,23,28]`) and `incoai/Qwen3.8-27B-DFlash2` for 27B
  `M=8` (taps `[5,19,33,47,61]`). These supersede the stale z-lab names/tap lists
  in the original entry notes. Verify **M rows total**, including one anchor and
  `M-1` proposals, with rollback and warm proposer-context restoration.
