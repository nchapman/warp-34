# warp-34 — Session Orientation & Current State

## Status: Phase 0, Phase 1, & Phase 2 Complete (`main`)

`warp-34` has completed **Phase 0** (Foundational Seam Contracts, Oracle & `nsys` Diff Harness, CUDA Runtime & `CapacityPlanner`, Unified Safetensors Loader), **Phase 1** (Donor-Aligned Kernel Suite + Decisive Gate Tests #1, #2, #4), and **Phase 2** (Qwen 3.5 / 3.8 Hybrid Backbone on `TP=1` and `TP=2`, `KvPool`, `StateBundlePool`, `CUDAGraphRunner` Gate #3, and `w34` CLI).

All 4 dense target model configurations (`4B TP=1`, `9B TP=1`, `9B TP=2`, `27B TP=2`) pass `w34 greedy-diff`, `w34 dump-diff`, and `w34 profile-diff`, **beating `vLLM 0.31.0` by +31% to +83% in single-stream decode throughput** on 1–2× RTX 3090 Ti (`sm_86`) before speculative decoding or `TQ4` compression.

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
   - `op_paged_attn.cu`: Descriptor-driven `kv_cache_write` and 4-warp stride-4 warp-shuffle `paged_attention_f16` supporting `FP16`, `BF16`, `TQ4` (`132 B` rows, Rademacher+WHT-256 + Beta(256) codebook), and `Q8_0` (`K8V4`), plus symmetric sliding-window mask for `DFlash2` (**Gate #1** 0-ULP page-table permutation invariance verified).
   - `op_gdn.cu`: `vllm.cpp`-derived `gdn_causal_conv1d_f16` and `gdn_recurrent_f16` with `extra_buffer` ping-pong `spec_state_indices` and `checkpoint_dst_indices` (**Gate #2** 0-ULP continuation and speculative rejection state parity verified).
   - `op_collectives.cu`: Pinned-sysmem two-hop `TP=2` collectives (`all_reduce_f16`, `all_reduce_bf16`, `all_reduce_f32`, `broadcast_i32`, `all_gather_top1_f32_i32`) with `.sys` release/acquire fences (**Gate #4** verified, including `INT32` token IDs `> 65504`).
   - `op_sampler.cu`: 1024-thread GPU `sampler_greedy_top5_f16_kernel` (`vocab_size = 248320`, sharded `[124160, 124160]` under `TP=2`).
5. **Phase 2 — Qwen 3.5 / 3.8 Hybrid Backbone & CLI (`crates/w34-host/src/{cache,model,main}.rs`):**
   - `KvPool`, `StateBundlePool`, `StateTx`, `Qwen3_5Engine` (`TP=1` and `TP=2`), `CUDAGraphRunner` (**Gate #3** bit-exact graph replay under metadata mutation verified), and `w34` CLI (`generate`, `bench`, `dump-diff`, `greedy-diff`, `profile-diff`).

---

## Next Phase Entry Point: Phase 3 (`docs/ROADMAP.md` §4)

Start **Phase 3** (Paged `TQ4` / `FirstLast2` Long-Context Validation & Continuous Batching Scheduler + `MambaRadixCache`):
- Validate `FirstLast2` (`--kv-format first-last2`) and `TQ4` (`--kv-format tq4`) long-context (`8k–128k`) decode throughput and perplexity / greedy parity against `FP16` KV.
- Implement `Scheduler` + `MambaRadixCache` (`SGLang` `extra_buffer` ping-pong GDN prefix checkpointing) and `W4A8` dense MLP evaluation (`docs/ROADMAP.md` §4 Phase 3.3).
