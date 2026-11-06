# HGPTQ integration (`feat/hgptq`)

## Status

**Complete and verified end-to-end** against `vLLM 0.31.0` across all four flagship HGPTQ target models and the quantized `DFlash2` drafter:
- **`nchapman/Qwen3.5-2B-W4A16-HGPTQ`** (`0fbb9a38f631ea63678f4b51a491e7ecb26c681c`)
- **`nchapman/Qwen3.5-9B-W4A16-HGPTQ`** (`7b2e48b7e7e8ee38925f5337c66feed22d3b031b`)
- **`nchapman/Qwen3.8-27B-W4A16-HGPTQ`** (`1de743eedefdb7422e6b171174c7080204f5b005` / `ae51987e3339a2a70070702548571e10fe8bcbc2`)
- **`nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ`** (`46b87ea125d24924affa8da9c3e981a937aab216`, `2.16 GiB` vs `4.81 GiB` for `incoai/Qwen3.8-27B-DFlash2`)
- **`nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ`** (`8aeeacaf9ba943fb45b5e27de6655a633de58767`, `19.4 GiB`, 256-expert MoE + Vision + MTP)

Verified across Text (`dump-diff` & `greedy-diff`), `MTP` (`k=1..3` lossless), `DFlash2` (`M=8` lossless, both `BF16` and `W4A16-HGPTQ`), Vision (`coco_sample.png`, including 9B/27B/35B-A3B `b=16` channelwise `K/N=4304`), CUDA Graphs, and `TP=1 == TP=2`.

## Pinned checkpoint inventory

1. **`nchapman/Qwen3.5-2B-W4A16-HGPTQ`** (`0fbb9a38f631ea63678f4b51a491e7ecb26c681c`):
   - `quant_method=compressed-tensors`, `format=pack-quantized`, symmetric INT4 (`U4B8`), `group_size=64`, `actorder=static`.
   - **150 text + 98 vision + 7 MTP** packed linears (`head_dim=64` block-Hadamard across text, vision, and MTP).
2. **`nchapman/Qwen3.5-9B-W4A16-HGPTQ`** (`7b2e48b7e7e8ee38925f5337c66feed22d3b031b`):
   - **32 text layers + 27 vision blocks + 1 MTP layer** (`K/N=4304` vision MLP with `group_0` `g=64, b=64` and `group_1` channelwise `strategy="channel", b=16` on `blocks.*.mlp.linear_fc2`).
3. **`nchapman/Qwen3.8-27B-W4A16-HGPTQ`** (`1de743eedefdb7422e6b171174c7080204f5b005` / `ae51987e3339a2a70070702548571e10fe8bcbc2`):
   - **64 text layers + 27 vision blocks + 1 MTP layer** (`K/N=4304` vision MLP with `group_1` channelwise `strategy="channel", b=16` on `blocks.*.mlp.linear_fc2`).
4. **`nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ`** (`46b87ea125d24924affa8da9c3e981a937aab216`):
   - **5 DFlash2 layers + `fc` (`[5120, 25600]`)** quantized with symmetric INT4 (`U4B8`), `group_size=64`, and online `H_64` block-Hadamard rotation (`2.16 GiB` total vs `4.81 GiB` unquantized `BF16`, saving `2.65 GiB` VRAM while preserving native `BF16` activations, `BF16` norms, `BF16` grouped convolutions, and `BF16` selector heads).
5. **`nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ`** (`8aeeacaf9ba943fb45b5e27de6655a633de58767`):
   - **40 MoE layers (`30` GDN + `10` full attention, `256` routed experts, `top_k=8`, `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`) + 27 vision blocks + 1 MoE MTP layer** (`19.4 GiB`).
   - Attention (`self_attn`, `linear_attn`, `mtp.layers.0.self_attn`) and Vision (`visual.blocks.*.attn`, `mlp.linear_fc1`, `merger`) use `U4B8` `group_size=64` with `H_64`; `visual.blocks.*.mlp.linear_fc2` (`K/N=4304`) uses channelwise `U4B8` (`strategy="channel"`) with `H_16`; all `256` routed experts per layer (`mlp.experts.*` and `mtp.layers.0.mlp.experts.*`) use `U4B8` `group_size=64` (`hadamard=None`); and `mlp.shared_expert.*`, `mlp.gate`, `in_proj_a/b`, `mtp.fc`, and `lm_head` remain unquantized `FP16`.
   - Unquantized `DenseF16` MoE `shared_expert` projections (`size_n <= 2048, size_k <= 2048, m <= 16`) route through the batch-size-invariant `w34_small_n_gemm_f16` warp-reduction kernel (`crates/w34-host/src/model/weights.rs`), guaranteeing `0-ULP` parity between `M = 1` plain decode and `M <= 16` `MTP` verification across both `TP=1` and `TP=2`.
- Target rules use negative lookbehind (`re:.*(?<!mtp\.)model\.layers\.\d+\.mlp\.(gate|up|down)_proj$`) and explicit `ignore` lists (`lm_head`, `patch_embed.proj`, `mtp.fc`, `in_proj_a`, `in_proj_b`), resolved via bounded `fancy-regex` in `crates/w34-host/src/loader/config_rules.rs`.

## Architecture, kernel contracts & bug fixes

1. **Vectorized Block-Hadamard (`w34_block_hadamard_f16`, `crates/w34-kernels/csrc/ops/op_hadamard.cu`):**
   - Each 4-lane (`B=16`) or 16-lane (`B=64`) subgroup loads 4 contiguous `__half` values per lane via coalesced 8-byte `uint2` loads, executes stages `h=1, 2` via an in-register `H_4` butterfly in `FP32`, and executes stages `h=4..B/2` via `__shfl_xor_sync` before normalizing (`1/4` or `1/8`) and writing 8-byte `uint2` stores.
   - Fuses zero-padding of `padded_k - k` tail columns (`4304 -> 4352` for 9B/27B vision `fc2`).
   - Strictly out-of-place into `ex.scratch_ptr` (rejecting overlapping ranges) so shared unrotated consumers (e.g., GDN `in_proj_a`/`in_proj_b` on `aux_stream`) never observe mutated activations.
   - Because `qkv_proj`, `in_proj_qkvz`, and `gate_up_proj` are concatenated at load time via `RawLinearHost::concat_rows`, `LinearProj::Hadamard` automatically shares a single input Hadamard rotation across fused Q+K+V, QKV+Z, and Gate+Up projections.
2. **Channelwise W4 Marlin & Pre-Cast `FP32` Scale Fix (`marlin_template.h`, `op_marlin.cu`, `op_hadamard.cu`, `hgptq.rs`):**
   - Instantiates `MARLIN_CASES_15(marlin::kU4B8, -1)` in `crates/w34-kernels/csrc/ops/op_marlin.cu` so `w34_marlin_gemm_f16u4` supports `group_size = -1` (channelwise) alongside `group_size in {32, 64, 128}`.
   - **Pre-`FP16`-cast `FP32` channelwise scale multiplication (`crates/w34-kernels/csrc/marlin/marlin_template.h`):** In 9B/27B Vision (`blocks.26.mlp.linear_fc2`, `group_blocks == -1`, `K = 4304`), the unscaled `FP32` accumulator `sum(x * (q - 8))` reaches `67,342.7 > 65,504.0` (`FP16` max) while the true scaled output (`* 0.131836 = 8,878.19`) fits well within `FP16`. Extended `marlin_template.h` so `4-bit` channelwise (`b_type.size_bits() == 4`, `group_blocks == -1`) applies `scale_float<c_type_id>` to `frag_c` in `FP32` prior to `global_reduce_fp16` and `write` (instead of converting unscaled `frag_c` to `FP16` first and multiplying by `frag_s` in `FP16`), preventing `FP16` `+Inf` overflow. Covered by unit regression test `channelwise_hadamard_k4304_does_not_overflow_fp16_before_scale`.
   - Pads non-tile-aligned `K` (to multiple of `128`) and `N` (to multiple of `64`) at load time using symmetric `U4B8` zero nibbles (`0x8888_8888`, `(8 - 8) * scale = 0.0`) and scale `1.0`.
   - `w34_linear_epilogue_f16` crops physical `N` (`4352`) back to logical `N` (`4304`) and applies optional `FP16` bias (`VisionEncoder` linears).
3. **27B Quantized MTP Repack Staging Buffer (`crates/w34-host/src/mtp.rs`):**
   - Scaled `staging_bytes` in `Qwen3_5MtpRank::from_safetensors_with_tp` dynamically to `(64 * 1024 * 1024).max(((2 * (i / tp).div_ceil(64) * 64) * (h.div_ceil(128) * 128) * c.quant.bits()) / 8)` (`85.0 MiB` on 27B `TP=1` quantized MTP `gate_up_proj`).
4. **Upstream `vLLM 0.31.0` Oracle Patches (`tools/oracle/hgptq_oracle.py`):**
   - Patched `HadamardTransform.forward` to pad non-multiple-of-32 block counts (e.g., `1200 * (4304 / 16) = 322,800` blocks in 9B/27B Vision `linear_fc2`, where `322,800 % 32 = 16`) to a multiple of 32 before `ops.hadacore_transform`, avoiding `hadacore`'s upstream tail-tile corruption on the final 16 blocks.
   - Patched `MarlinLinearKernel.apply_weights` for `group_size == -1` with an exact `2^-3` / `2^+3` (`0.125` / `8.0`) exponent shift so `vLLM`'s unscaled `FP16` cast does not overflow to `NaN` on 9B Vision `blocks.26.mlp.linear_fc2`.
5. **Producer-Fused In-Register `H_64` Rotation (`hadamard_inline.cuh`, `op_norms_rope_act.cu`, `op_collectives.cu`, `layers.rs`, `moe.rs`, `forward.rs`, `mtp/propose.rs`):**
   - Eliminates **100% of standalone `w34_block_hadamard_f16` kernel launches** (`96` per 2B step, `128` per 9B step, `256` per 27B step, plus `4` per MTP draft step) in the 2B, 9B, and 27B target backbone and MTP drafter by fusing `H_64` in registers into every producer kernel:
     - `w34_gemma_rms_norm_hadamard_f16` (`gemma_rms_norm_vec8_f16_kernel<R, true>`),
     - `w34_fused_add_gemma_rms_norm_tap_hadamard_f16` (`fused_add_gemma_rms_norm_vec8_f16_kernel<R, true>`),
     - `w34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16` (`tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<BlockThreads, R>`),
     - `w34_rms_norm_gated_{silu,sigmoid}_strided_hadamard_f16` (`rms_norm_gated_d128_warp_f16_kernel<IsSigmoid, true>`),
     - `w34_apply_attn_output_gate_hadamard_f16` (`apply_attn_output_gate_hadamard_f16_kernel`),
     - `w34_silu_and_mul_hadamard_f16` (`silu_and_mul_hadamard_f16_kernel`).
   - **0-ULP bit-for-bit parity:** `w34::hadamard64_vec8_f16` (8-lane subgroup, 8 halves/lane, 3 `__shfl_xor_sync` stages) and `w34::hadamard64_vec4_f16` (16-lane subgroup, 4 halves/lane) round the producer output to `__half` before executing the bit-order `FP32` butterfly and `* 0.125f` RN rounding, matching standalone `w34_block_hadamard_f16(..., 64)` with **0 ULP** (`producer_fused_hadamard64_matches_standalone_0_ulp` & `tp2_all_reduce_residual_gemma_rms_norm_hadamard64_matches_standalone_0_ulp`).
   - **Dual-output norm for GDN shared input:** On GDN layers where `in_proj_ba` (`DenseF16`) needs unrotated `s.normed` and `in_proj_qkvz` (`Hadamard` `B=64`) needs `H_64(s.normed)`, `LayerBlockWeights::pre_attn_norm_targets` returns `[s.normed, ex.scratch_ptr]` so the norm kernel writes both unrotated `s.normed` and rotated `ex.scratch_ptr` in a single pass; on FullAttention (`qkv_proj`) and Dense FFN (`gate_up_proj`), it returns `[0, s.normed]` so only the rotated buffer is written.
   - **Verified with `nsys`:** Across 2B, 9B, and 27B (`TP=1` & `TP=2`, `MTP-3`), `block_hadamard_f16<64>` executes only once per rank during `prepare_execution_plan()` warmup and **0 times** during prefill, decode, or MTP drafting.
6. **Native `BF16` Marlin W4 + `BF16` Block-Hadamard for `DFlash2-W4A16-HGPTQ` (`op_marlin.cu`, `op_hadamard.cu`, `dflash.rs`, `config_rules.rs`, `hub/files.rs`):**
   - Instantiated `MARLIN_BF16_CASES_15(marlin::kU4B8, 2/4/8/-1)` (`w34_marlin_gemm_bf16u4`) in `crates/w34-kernels/csrc/ops/op_marlin.cu` and `w34_block_hadamard_bf16` (`B=16, 64`) in `crates/w34-kernels/csrc/ops/op_hadamard.cu` so `DFlash2Rank` executes quantized `U4B8` projections with native `BF16` activations and `BF16` scales inside its captured CUDA Graphs (`forward_graph` and `select_greedy_graph`).
   - Added `Linear::MarlinBf16U4B8` to `crates/w34-host/src/dflash.rs`, fusing `q_proj + k_proj + v_proj` (`qkv`) and `gate_proj + up_proj` (`gate_up`) at load time while repacking a separate `kv` (`k_proj + v_proj`) projection for `commit_context` (since Marlin `[k/16, n*2]` tiles cannot be pointer-offset sliced across `N`).
7. **Cold-start TP warmup (`crates/w34-host/src/cuda/warmup.rs`):**
   - Pre-warms `w34_block_hadamard_f16` & `w34_block_hadamard_bf16` (`B=16` and `B=64`), all 7 producer-fused `H_64` entrypoints across all `hidden_size` template regimes (`<=2560, <=4096, <=5120, <=6144`), `w34_linear_epilogue_f16`, channelwise `w34_marlin_gemm_f16u4(..., -1)`, and `w34_marlin_gemm_bf16u4` (`gs=-1, 32, 64, 128`) in `prepare_execution_plan()`.

## Verification & benchmarks on 1–2× RTX 3090 Ti (`sm_86`)

### 1. Vectorized `w34_block_hadamard_f16` Microbenchmarks (`tests/hgptq_hadamard.rs` & `hgptq_hadamard_probe.py`)

| Block / shape | `M=1` (µs) | `M=8` (µs) | `M=16` (µs) | `M=512` (µs) | `M=4096` (µs) | vs Hadacore (`rel_l2`) |
|---|---:|---:|---:|---:|---:|---:|
| `H64`, `K=2048` (2B) | 1.297 | 1.315 | 1.329 | 3.760 | 39.106 | 3.72e-4 |
| `H64`, `K=4096` (9B) | 1.297 | 1.329 | 1.376 | 8.907 | 75.850 | 3.72e-4 |
| `H64`, `K=5120` (27B) | 1.295 | 1.329 | 1.399 | 13.076 | 94.594 | 3.72e-4 |
| `H64`, `K=6144` (2B/27B) | 1.198 | 1.243 | 1.404 | 15.472 | 112.736 | 3.72e-4 |
| `H64`, `K=12288` (9B MLP) | 1.190 | 1.393 | 1.614 | 29.785 | 223.557 | 3.72e-4 |
| `H64`, `K=17408` (27B MLP) | 1.195 | 1.482 | 1.776 | 41.167 | 316.406 (`901 GB/s`) | 3.72e-4 |
| `H16`, `K=4304 -> 4352` (9B/27B Vision) | 1.138 | 1.167 | 1.208 | 8.869 | 79.755 | 2.69e-4 |

### 2. End-to-End `vLLM 0.31.0` & `TP=1 == TP=2` Parity (`w34 dump-diff` & `w34 greedy-diff`)

| Model | Workload | `dump-diff` vs `vLLM` (`max_rel_l2` / `min_cos`) | `TP=1` vs `TP=2` (`max_rel_l2`) | `greedy-diff` Token Match | Speculative Lossless Verification |
|---|---|---:|---:|---|---|
| **2B HGPTQ** | Text (`25` tensors) | `0.002022` / `0.999998` | `0.002856` (`match=true`) | **256/256 (100.0%)** | `MTP k=1..3` **4/4 lossless** (`tau = 1.92x / 2.80x / 3.41x`, `draft_exact = 3/3`) |
| **2B HGPTQ** | Vision (`245` tensors) | `0.022055` / `0.999871` | `0.003225` (`match=true`) | **32/32 (100.0%)** | `TP=1 == TP=2 == vLLM` exact |
| **9B HGPTQ** | Text (`33` tensors) | `0.001863` / `0.999998` | `0.004095` (`match=true`) | **128/128 (100.0%)** | `MTP k=1..3` **4/4 lossless** (`tau = 1.88x / 2.58x / 3.10x`, `draft_exact = 3/3`) |
| **9B HGPTQ** | Vision (`280` tensors) | `0.008970` / `0.999960` | `0.023136` (`match=true`) | **10/10 (100.0%)** | `TP=1 == TP=2 == vLLM` exact |
| **27B HGPTQ** | Text (`65` tensors) | `0.001938` / `0.999998` | `0.003472` (`match=true`) | **128/128 (100.0%)** | `MTP k=1..3` (`tau = 1.91x / 2.58x / 3.18x`) & `DFlash2` (`tau = 3.88x`, `draft_exact = 7/7`) **4/4 lossless** |
| **27B HGPTQ** | Vision (`312` tensors) | `0.011096` / `0.999938` | `0.013295` (`match=true`) | **32/32 (100.0%)** | `TP=1 == TP=2 == vLLM` exact |
| **35B-A3B HGPTQ** | Text (`41` tensors) | `0.048726` / `0.994578` (`7.25e-4` L0..10) | `0.028963` (`match=true`) | **127/128 (99.2%)** (`4/4` adjudicated, `gap=0.0156`) | `MTP k=1..3` (`tau = 1.82x / 2.48x / 2.88x`, `draft_exact = 3/3`) **4/4 lossless & radix-warm** |
| **35B-A3B HGPTQ** | Vision (`288` tensors) | `0.007034` ViT / `0.2729` L39 (`0.999975` / `0.9553`) | `0.157357` (`match=true`) | **14/14 (100.0%)** | `TP=1 == TP=2 == vLLM` exact |

### 3. Comprehensive Benchmark Suite (`1–2× RTX 3090 Ti`, `TQ4` KV, CUDA Graphs Enabled)

| Model | TP | Mode | `128:32` Prefill / Decode (`tok/s`) | `512:128` Prefill / Decode (`tok/s`) | `2048:32` Prefill / Decode (`tok/s`) | Vision `308:32` TTFT / Decode | vs `vLLM 0.31.0` Graph (`128:32`) | vs Standard W4A16 GPTQ (`128:32`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **2B HGPTQ** | `TP=1` | Plain | `12,213` / **343.2** | — | — | `79.6 ms` / **301.7 tok/s** | **`1.25×`** (vs `274.7 tok/s`) | — |
| **2B HGPTQ** | `TP=1` | `MTP-3` | `13,010` / **851.6** (`3.88x`) | `17,381` / **848.4** (`3.85x`) | `16,100` / **716.4** (`3.44x`) | — | **`3.10×`** | — |
| **2B HGPTQ** | `TP=2` | Plain | `12,088` / **478.8** | — | — | `81.2 ms` / **390.7 tok/s** | **`1.74×`** | — |
| **2B HGPTQ** | `TP=2` | `MTP-3` | `12,049` / **1,131.3** (`3.88x`) | `15,764` / **1,119.6** (`3.85x`) | `14,602` / **931.8** (`3.44x`) | — | **`4.12×`** | — |
| **9B HGPTQ** | `TP=1` | Plain | `4,160` / **133.2** | — | — | `161.7 ms` / **124.6 tok/s** | **`1.18×`** (vs `112.7 tok/s`) | `97.5%` of GPTQ (`136.6`) |
| **9B HGPTQ** | `TP=1` | `MTP-3` | `3,948` / **356.9** (`3.88x`) | `4,404` / **356.4** (`3.85x`) | `4,505` / **344.6** (`3.88x`) | — | **`3.17×`** | **`+6.9%`** vs GPTQ (`333.8`) |
| **9B HGPTQ** | `TP=2` | Plain | `4,455` / **213.3** | — | — | `152.6 ms` / **185.9 tok/s** | **`1.89×`** | `97.7%` of GPTQ (`218.4`) |
| **9B HGPTQ** | `TP=2` | `MTP-3` | `4,401` / **540.1** (`3.88x`) | `5,231` / **545.9** (`3.85x`) | `5,060` / **507.0** (`3.88x`) | — | **`4.79×`** | **`+6.5%`** vs GPTQ (`507.2`) |
| **27B HGPTQ** | `TP=1` | Plain | `1,331` / **51.8** | — | — | `344.4 ms` / **49.3 tok/s** | **`1.19×`** (vs `43.5 tok/s`) | `97.6%` of GPTQ (`53.1`) |
| **27B HGPTQ** | `TP=1` | `MTP-3` | `1,314` / **108.7** (`2.58x`) | `1,450` / **150.1** (`3.63x`) | `1,418` / **139.3** (`3.44x`) | — | **`3.45×`** (`512:128`) | `162.9 tok/s` on `greedy-diff` |
| **27B HGPTQ** | `TP=1` | `DFlash2-W4` | `1,293` / **107.1** (`2.82x`) | `1,472` / **247.9** (`6.68x`) | — | — | **`5.70×`** (`512:128`) | **`+10.6%`** vs `BF16` DFlash2 (`224.1`), `-2.65 GiB` VRAM |
| **27B HGPTQ** | `TP=2` | Plain | `1,644` / **85.4** | — | — | `300.8 ms` / **77.9 tok/s** | **`1.96×`** | `96.8%` of GPTQ (`88.2`) |
| **27B HGPTQ** | `TP=2` | `MTP-3` | `1,628` / **252.1** (`3.88x`) | `1,848` / **235.6** (`3.63x`) | `1,822` / **213.2** (`3.44x`) | — | **`5.80×`** | **`+3.8%`** vs GPTQ (`242.9`) |
| **27B HGPTQ** | `TP=2` | `DFlash2-W4` | `1,545` / **351.0** (`6.20x`) | `1,826` / **365.1** (`6.68x`) | — | — | **`8.39×`** (`512:128`) | **`+8.3%`** vs `BF16` DFlash2 (`337.1`), `17.66 ms/round` |
| **35B-A3B HGPTQ** | `TP=1` | Plain | `2,554` / **206.9** | `5,729` / **204.2** | — | `304.2 ms` / **190.4 tok/s** | `vLLM` OOMs on 1× 24 GB (`TP=1`) | **`+10.9%`** vs GPTQ (`186.5`) |
| **35B-A3B HGPTQ** | `TP=1` | `MTP-3` | `2,589` / **432.2** (`3.88x`) | `4,627` / **404.5** (`3.63x`) | `4,698` / **415.2** (`3.88x`) | — | Fits 1× 24 GB with `MTP-3` + Vision | **`2.32×`** vs GPTQ `TP=1` (`186.5`) |
| **35B-A3B HGPTQ** | `TP=2` | Plain | `3,229` / **267.6** | `5,623` / **269.5** | — | `226.3 ms` / **235.9 tok/s** | **`1.40×`** (vs `191.4 tok/s`) | **`+6.7%`** vs GPTQ (`250.7`) |
| **35B-A3B HGPTQ** | `TP=2` | `MTP-3` | `3,467` / **562.2** (`3.88x`) | `5,300` / **460.5** (`3.10x`) | `5,433` / **509.3** (`3.88x`) | — | **`2.94×`** (vs `191.4 tok/s`) | **`+44.1%`–`+75.9%`** vs GPTQ `MTP` (`319.6`) |

## Reproducing checks

```sh
export PATH="/opt/cuda/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --lib hgptq
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --test hgptq_hadamard -- --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --test quant_formats -- --test-threads=1
CUDA_VISIBLE_DEVICES=0,1 cargo test --release --features cuda --test hgptq_checkpoint -- --ignored --test-threads=1
```
