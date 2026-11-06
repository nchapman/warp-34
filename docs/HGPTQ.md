# HGPTQ integration (`feat/hgptq`)

## Status

**Complete and verified end-to-end** against `vLLM 0.31.0` across all three flagship HGPTQ models:
- **`nchapman/Qwen3.5-2B-W4A16-HGPTQ-g64`** (`0fbb9a38f631ea63678f4b51a491e7ecb26c681c`)
- **`nchapman/Qwen3.5-9B-W4A16-HGPTQ-g64`** (`7b2e48b7e7e8ee38925f5337c66feed22d3b031b`)
- **`nchapman/Qwen3.8-27B-W4A16-HGPTQ-g64`** (`1de743eedefdb7422e6b171174c7080204f5b005`, plus `incoai/Qwen3.8-27B-DFlash2`)

Verified across Text (`dump-diff` & `greedy-diff`), `MTP` (`k=1..3` lossless), `DFlash2` (`M=8` lossless), Vision (`coco_sample.png`, including 9B/27B `b=16` channelwise `K/N=4304`), CUDA Graphs, and `TP=1 == TP=2`.

## Pinned checkpoint inventory

1. **`nchapman/Qwen3.5-2B-W4A16-HGPTQ-g64`** (`0fbb9a38f631ea63678f4b51a491e7ecb26c681c`):
   - `quant_method=compressed-tensors`, `format=pack-quantized`, symmetric INT4 (`U4B8`), `group_size=64`, `actorder=static`.
   - **150 text + 98 vision + 7 MTP** packed linears (`head_dim=64` block-Hadamard across text, vision, and MTP).
2. **`nchapman/Qwen3.5-9B-W4A16-HGPTQ-g64`** (`7b2e48b7e7e8ee38925f5337c66feed22d3b031b`):
   - **32 text layers + 27 vision blocks + 1 MTP layer** (`K/N=4304` vision MLP with `group_0` `g=64, b=64` and `group_1` channelwise `strategy="channel", b=16` on `blocks.*.mlp.linear_fc2`).
3. **`nchapman/Qwen3.8-27B-W4A16-HGPTQ-g64`** (`1de743eedefdb7422e6b171174c7080204f5b005`):
   - **64 text layers + 27 vision blocks + 1 MTP layer** (`K/N=4304` vision MLP with `group_1` channelwise `strategy="channel", b=16` on `blocks.*.mlp.linear_fc2`).
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
5. **Cold-start TP warmup (`crates/w34-host/src/cuda/warmup.rs`):**
   - Pre-warms `w34_block_hadamard_f16` (`B=16` and `B=64`), `w34_linear_epilogue_f16`, and channelwise `w34_marlin_gemm_f16u4(..., -1)` in `prepare_execution_plan()`.

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

### 3. Comprehensive Benchmark Suite (`1–2× RTX 3090 Ti`, `TQ4` KV, CUDA Graphs Enabled)

| Model | TP | Mode | `128:32` Prefill / Decode (`tok/s`) | `512:128` Prefill / Decode (`tok/s`) | `2048:32` Prefill / Decode (`tok/s`) | Vision `308:32` TTFT / Decode | vs `vLLM 0.31.0` Graph (`128:32`) | vs Standard W4A16 GPTQ (`128:32`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **2B HGPTQ** | `TP=1` | Plain | `12,763` / **339.9** | — | — | `79.6 ms` / **301.7 tok/s** | **`1.24×`** (vs `274.7 tok/s`) | — |
| **2B HGPTQ** | `TP=1` | `MTP-3` | `12,749` / **842.1** (`3.88x`) | `16,968` / **824.6** (`3.85x`) | `15,856` / **713.5** (`3.44x`) | — | **`3.07×`** | — |
| **2B HGPTQ** | `TP=2` | Plain | `11,809` / **459.1** | — | — | `81.2 ms` / **390.7 tok/s** | **`1.67×`** | — |
| **2B HGPTQ** | `TP=2` | `MTP-3` | `11,780` / **1,087.5** (`3.88x`) | `15,375` / **1,080.3** (`3.85x`) | `14,325` / **884.9** (`3.44x`) | — | **`3.96×`** | — |
| **9B HGPTQ** | `TP=1` | Plain | `3,978` / **131.1** | — | — | `161.7 ms` / **124.6 tok/s** | **`1.16×`** (vs `112.7 tok/s`) | `96.0%` of GPTQ (`136.6`) |
| **9B HGPTQ** | `TP=1` | `MTP-3` | `3,737` / **356.2** (`3.88x`) | `4,384` / **354.9** (`3.85x`) | `4,521` / **339.7** (`3.88x`) | — | **`3.16×`** | **`+6.7%`** vs GPTQ (`333.8`) |
| **9B HGPTQ** | `TP=2` | Plain | `4,366` / **207.6** | — | — | `152.6 ms` / **185.9 tok/s** | **`1.84×`** | `95.1%` of GPTQ (`218.4`) |
| **9B HGPTQ** | `TP=2` | `MTP-3` | `4,311` / **531.3** (`3.88x`) | `5,043` / **530.7** (`3.85x`) | `4,935` / **493.8** (`3.88x`) | — | **`4.71×`** | **`+4.8%`** vs GPTQ (`507.2`) |
| **27B HGPTQ** | `TP=1` | Plain | `1,319` / **51.3** | — | — | `344.4 ms` / **49.3 tok/s** | **`1.18×`** (vs `43.5 tok/s`) | `96.6%` of GPTQ (`53.1`) |
| **27B HGPTQ** | `TP=1` | `MTP-3` | `1,313` / **107.9** (`2.58x`) | `1,437` / **149.3** (`3.63x`) | `1,409` / **137.8** (`3.44x`) | — | **`3.43×`** (`512:128`) | `162.0 tok/s` on `greedy-diff` |
| **27B HGPTQ** | `TP=1` | `DFlash2` | `1,314` / **96.2** (`2.82x`) | `1,436` / **223.6** (`6.68x`) | — | — | **`5.14×`** (`512:128`) | `153.9 tok/s` on `greedy-diff` |
| **27B HGPTQ** | `TP=2` | Plain | `1,619` / **84.4** | — | — | `300.8 ms` / **77.9 tok/s** | **`1.94×`** | `95.7%` of GPTQ (`88.2`) |
| **27B HGPTQ** | `TP=2` | `MTP-3` | `1,594` / **250.7** (`3.88x`) | `1,811` / **233.6** (`3.63x`) | `1,799` / **210.5** (`3.44x`) | — | **`5.76×`** | **`+3.2%`** vs GPTQ (`242.9`) |
| **27B HGPTQ** | `TP=2` | `DFlash2` | `1,548` / **320.3** (`6.20x`) | `1,826` / **334.2** (`6.68x`) | — | — | **`7.68×`** (`512:128`) | `334.2 tok/s` (`6.68x` tau) |

## Reproducing checks

```sh
export PATH="/opt/cuda/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --lib hgptq
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --test hgptq_hadamard -- --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --test quant_formats -- --test-threads=1
CUDA_VISIBLE_DEVICES=0,1 cargo test --release --features cuda --test hgptq_checkpoint -- --ignored --test-threads=1
```
