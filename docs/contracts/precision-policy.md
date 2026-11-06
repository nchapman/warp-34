# Precision Policy Contract — `FP16` Target Spine, `BF16` Drafter Preservation, & `FP32` State/Reductions

## 1. Summary Matrix

| Subsystem / Tensor | Storage Dtype | Compute / Accumulator Dtype | Rationale |
|---|---|---|---|
| **Target Model Activation Spine** (`hidden_states`, residual stream, `qkv`, `attn_out`, `mlp_out`, `TapPlan` captures) | `FP16` (`__half`) | `FP32` (`float`) internal | Matches `vLLM 0.31.0` GPTQ/Marlin spine; eliminates ~891 `f32 <-> f16` glue casts/round from `warp3090`. |
| **Target Quantized Linear Weights** (`W4A16` / `W8A16` Marlin tiles, scales) | `INT4`/`INT8` packed (`uint32_t`) + `FP16` scales | `FP16` MMA w/ `FP32` acc | Single weight residency in Marlin tile layout (`gptq_marlin_repack` / `awq_marlin_repack`). |
| **Target Norms** (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `final_norm`, `GemmaRMSNorm` `1.0 + w`) | `FP16` or `FP32` weights; `FP16` activation I/O | `FP32` variance & `1.0f + float(w)` | Prevents variance underflow/overflow and matches `vLLM` `GemmaRMSNorm` (`eps = 1e-6`). |
| **GDN Persistent Recurrent State (`ssm_state`)** | **`FP32` (`float`) in global memory** `[slots, H_v, 128, 128]` | `FP32` (`float`) | `vLLM` (`qwen_gdn_linear_attn.py:483`) & `vllm.cpp` (`cuda_gdn.cu`) mandate `FP32` `ssm_state`; `FP16`/`BF16` state drifts on long context. |
| **GDN Conv State (`conv_state`) & Conv Weights** | `FP16` (`__half`) `[slots, conv_dim, 3]` | `FP32` dot product, `FP16` store | Matches `vLLM` `conv_state` dtype (`torch.float16`). |
| **GDN Decay & Bias Parameters (`A_log`, `dt_bias`)** | **`FP32` (`float`)** | `FP32` (`-expf(A_log) * softplus(a + dt_bias)`) | Exponential decay sensitivity requires `FP32` weights and math. |
| **GDN Output Norm (`RMSNormGated`)** | `FP16` I/O, `FP16`/`FP32` weight | `FP32` RMS + `silu(z)` gate | `norm_before_gate = True`, `activation = "silu"`. |
| **Full-Attention KV Cache (`KvLayerLayout`)** | `FP16` (`512 B/row`), `BF16` (`512 B/row`), `TQ4` (`132 B/row`), or `Q8_0` (`272 B/row`) | `FP32` online softmax (`m_i`, `l_i`, `acc_o`) | Online softmax and log-sum-exp accumulators are strictly `FP32`. |
| **Per-Layer Embeddings (`PLE`, `qwen4_exp`)** | `F8_E4M3` n-gram shards + `BF16` per-row scale on disk/pinned staging; `FP16` `ple_conv_state` | `FP32` signed-sqrt sigmoid gate & dilated conv dot product -> `FP16` output | Zero-copy mmap gathers `2,560 B/tok`, dequantizes on GPU into `FP16` activation spine. |
| **`MTP` Draft `lm_head` (`draft_lm_head`)** | Online GPU `MarlinW4A16` (`U4B8`, `group_size = 64`) quantized from `DenseF16` `target.lm_head` (or shared `MarlinW4A16` when target is already quantized) | `FP16` MMA w/ `FP32` acc | Cuts draft `lm_head` HBM reads by `3.56x` (`vLLM` PR #59973 `quantized_lm_head_copy`) while target verification uses full `target.lm_head`. |
| **`DFlash2` Drafter Spine (4B vs 27B)** | **Native checkpoint dtype preserved** (`BF16` on 4B drafter; `FP16` or `BF16` per `config.json` on 27B drafter) | `FP32` acc | 4B `DFlash2` drafter activations exceed `FP16` max (`65504`) and overflow to `NaN` if downcast to `FP16`. `sm_86`/`sm_89` support native `BF16` MMA and elementwise kernels. |
| **LM Head Logits & GPU Sampler (`SamplerV1`)** | `FP16` GEMM output -> **`FP32` sampler logits** | `FP32` penalties, temperature, softmax, top-k/top-p, logprobs | Prevents ties/saturation during temperature scaling, Gumbel-max sampling, and logprob computation. |
| **`TP=2` / `TP=4` Collectives** | Typed per payload (`FP16`, `BF16`, `FP32`, `INT32`) | `FP32` sum for `FP16`/`BF16`/`FP32`; exact `INT32` for token IDs | Never cast token IDs (`vocab_size = 248,320 > 65,504`) or counters through `FP16`. |

## 2. Loader Dtype Enforcement Rules (`w34-host::loader`)

When loading any Safetensors checkpoint:
1. **Target Model (`qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen4_exp`, `qwen4_exp_text`):**
   - `A_log` and `dt_bias` tensors are converted to and stored as **`FP32`** on GPU regardless of whether the file stores them in `F32`, `BF16`, or `F16`.
   - Unquantized embedding tables (`embed_tokens.weight`), norm weights (`*.weight`), `conv1d.weight`, `in_proj_ba.weight` / `in_proj_b.weight` / `in_proj_a.weight` (when unquantized), and `lm_head.weight` (when unquantized) are stored in **`FP16`** on the target spine ( converting `BF16 -> FP16` at load time with finite-range assertion).
   - Quantized weights (`qweight` / `weight_packed`, `scales` / `weight_scale`, `qzeros` / `weight_zero_point`, `g_idx`) are repacked once at load time into the unified **Marlin `W4A16` / `W8A16` tile layout** with `FP16` scales.
2. **Drafter Model (`dflash2`):**
   - Inspects `torch_dtype` / tensor dtype in the checkpoint header.
   - If the drafter checkpoint is `BF16` and max weight/scale dynamic range or `config.json` specifies `bfloat16` (mandatory for `Qwen3.5-4B-DFlash2`), weights and execution buffers remain native **`BF16` (`W34_DTYPE_BF16`)**. Target hidden-state taps (`FP16`) are converted `FP16 -> BF16` inside the fused target-tap projection kernel.
