# HGPTQ integration (`feat/hgptq`)

## Status

**Complete and verified end-to-end** against `vLLM 0.31.0` on `nchapman/Qwen3.5-2B-W4A16-HGPTQ-g64` (`0fbb9a38f631ea63678f4b51a491e7ecb26c681c`), covering text, MTP (`k=1..3`), Vision (`coco_sample.png`), CUDA Graphs, and `TP=1 == TP=2`, plus synthetic unit/GPU coverage for the 9B/27B `b=16` channelwise vision `K=4304 / N=4304` geometry.

## Pinned checkpoint inventory

`nchapman/Qwen3.5-2B-W4A16-HGPTQ-g64` at `0fbb9a38f631ea63678f4b51a491e7ecb26c681c`:

- `quant_method=compressed-tensors`, `format=pack-quantized`, symmetric INT4 (`U4B8`), `group_size=64`, `actorder=static`.
- `model.safetensors`: 248 packed linears = **150 text + 98 vision** (`blocks.0..23.{qkv,proj,mlp.fc1,mlp.fc2}` + `merger.linear_fc1`, `merger.linear_fc2`).
- `model_mtp.safetensors`: **7 packed MTP linears** (`mtp.layers.0.*`), referenced directly from `model.safetensors.index.json`.
- Neither file stores `g_idx` or explicit Hadamard transform tensors.
- `transform_config.config_groups.v`: `type=hadamard`, `head_dim=64`, `randomize=false`, `precision=torch.float32`, with paired offline `weight_input` (inverse) and online `input` (forward) transform applications.
- Target rules use negative lookbehind (`re:.*(?<!mtp\.)model\.layers\.\d+\.mlp\.(gate|up|down)_proj$`) and explicit `ignore` lists (`lm_head`, `patch_embed.proj`, `mtp.fc`, `in_proj_a`, `in_proj_b`), resolved via bounded `fancy-regex` in `crates/w34-host/src/loader/config_rules.rs`.

## Architecture & kernel contracts

1. **Vectorized Block-Hadamard (`w34_block_hadamard_f16`, `crates/w34-kernels/csrc/ops/op_hadamard.cu`):**
   - Each 4-lane (`B=16`) or 16-lane (`B=64`) subgroup loads 4 contiguous `__half` values per lane via coalesced 8-byte `uint2` loads, executes stages `h=1, 2` via an in-register `H_4` butterfly in `FP32`, and executes stages `h=4..B/2` via `__shfl_xor_sync` before normalizing (`1/4` or `1/8`) and writing 8-byte `uint2` stores.
   - Fuses zero-padding of `padded_k - k` tail columns (`4304 -> 4352` for 9B/27B vision `fc2`).
   - Strictly out-of-place into `ex.scratch_ptr` (rejecting overlapping ranges) so shared unrotated consumers (e.g., GDN `in_proj_a`/`in_proj_b` on `aux_stream`) never observe mutated activations.
   - Because `qkv_proj`, `in_proj_qkvz`, and `gate_up_proj` are concatenated at load time via `RawLinearHost::concat_rows`, `LinearProj::Hadamard` automatically shares a single input Hadamard rotation across fused Q+K+V, QKV+Z, and Gate+Up projections.
2. **Channelwise W4 Marlin & Logical/Physical Epilogue (`op_marlin.cu`, `op_hadamard.cu`, `hgptq.rs`):**
   - Instantiates `MARLIN_CASES_15(marlin::kU4B8, -1)` in `crates/w34-kernels/csrc/ops/op_marlin.cu` so `w34_marlin_gemm_f16u4` supports `group_size = -1` (channelwise) alongside `group_size in {32, 64, 128}`.
   - Pads non-tile-aligned `K` (to multiple of `128`) and `N` (to multiple of `64`) at load time using symmetric `U4B8` zero nibbles (`0x8888_8888`, `(8 - 8) * scale = 0.0`) and scale `1.0`.
   - `w34_linear_epilogue_f16` crops physical `N` (`4352`) back to logical `N` (`4304`) and applies optional `FP16` bias (`VisionEncoder` linears). Unquantized `LinearProj::DenseF16` in `vision/linear.rs` continues to dispatch through `cublas.gemm_f16_bias` to preserve exact cuBLASLt fused-bias rounding on dense checkpoints.
3. **Cold-start TP warmup (`crates/w34-host/src/cuda/warmup.rs`):**
   - Pre-warms `w34_block_hadamard_f16` (`B=16` and `B=64`), `w34_linear_epilogue_f16`, and channelwise `w34_marlin_gemm_f16u4(..., -1)` in `prepare_execution_plan()`.

## Verification & benchmarks on RTX 3090 Ti (`sm_86`)

### 1. Vectorized `w34_block_hadamard_f16` vs `vLLM` Hadacore (`tools/oracle/hgptq_hadamard_probe.py`)

| Block / shape | Native µs | Hadacore µs | Relative L2 error |
|---|---:|---:|---:|
| H64, M=1, K=2048 | 1.305 | 1.143 | 3.59e-4 |
| H64, M=16, K=2048 | 1.345 | 1.204 | 3.72e-4 |
| H64, M=512, K=2048 | 3.766 | 3.662 | 3.72e-4 |
| H64, M=4096, K=6144 | 112.759 | 112.304 | 3.72e-4 |
| H16, M=1, K=4304 → 4352 | 1.129 | 3.128 | 2.58e-4 |
| H16, M=512, K=4304 → 4352 | 8.750 | 8.899 | 2.69e-4 |
| H16, M=4096, K=4304 → 4352 | 79.696 | 79.060 | 2.69e-4 |

### 2. End-to-end `vLLM 0.31.0` oracle parity (`tools/oracle/hgptq_oracle.py`)

- **Text `dump-diff` (`9` tokens, `TP=1` vs `vLLM` and `TP=1` vs `TP=2`):**
  - `final_norm`: `max_abs = 0.250000`, `rel_l2 = 0.002022`, `cosine = 0.999998`, `top1_match = true`, `top5_Set = 5/5` — **PASS**.
- **Text `greedy-diff` (`4` prompts × `32` tokens = `128` tokens):**
  - Plain decode (`k=0`): **128/128 (100.0%) exact token match**, `mean_top1_abs_logprob_diff = 0.003835`, `343.9 tok/s` — **PASS**.
  - Speculative `MTP (k=3)`: **4/4 prompts lossless exact match**, `mean_accepted_drafts = 2.35`, `mean_tokens_per_round = 3.35x`, `790.9 tok/s` — **PASS**.
- **Vision multimodal parity (`coco_sample.png`, `306` prompt tokens, `32` generated tokens):**
  - `vision_trace.patch_embed` / `blocks.0.input` / `blocks.0.norm1`: `rel_l2 = 0.0` (exact).
  - `vision_trace.blocks.0..22.output`: `rel_l2 = 0.000604 .. 0.002715`.
  - `visual_embeddings` (`[300, 2048]`): `rel_l2 = 0.016080` (`<= 0.05` limit).
  - `mrope_exact = true`, `step0_top1_exact = true` (`logprob_abs_diff = 0.007548`), **`greedy_exact = true` (32/32 tokens match `vLLM 0.31.0` and `TP=1 == TP=2`)** — **PASS**.

### 3. Matched 1× RTX 3090 Ti throughput (`prompt_len=128`, `gen_len=32`, identical token IDs)

| Engine / Mode | TTFT (ms) | Decode (ms) | Total (ms) | Decode tok/s | End-to-End tok/s | Token Match |
|---|---:|---:|---:|---:|---:|---|
| `vLLM 0.31.0` (CUDA Graph) | — | — | 116.45 | — | 274.8 | reference |
| `vLLM 0.31.0` (eager) | — | — | 109.70 | — | 291.7 | 32/32 exact |
| `w34` plain (`CUDA Graph`, `TQ4`) | 10.13 | 91.33 | 101.46 | **339.4** | **315.4** (`1.15×` vs vLLM graph) | 32/32 exact |
| `w34` `MTP-3` (`k=3`, `CUDA Graph`, `TQ4`) | 10.06 | 36.87 | **46.93** | **840.7** | **681.9** (`2.48×` vs vLLM graph) | 32/32 exact |

## Reproducing checks

```sh
export PATH="/opt/cuda/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --test hgptq_hadamard -- --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --test quant_formats -- --test-threads=1
CUDA_VISIBLE_DEVICES=0,1 cargo test --release --features cuda --test hgptq_checkpoint -- --ignored --test-threads=1
```
