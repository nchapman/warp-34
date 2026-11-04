# Qwen3.8-Flash-Next (`qwen4_exp`) — Architecture & Implementation Plan

> **Posture:** Native, production-grade support for **`Qwen3.8-Flash-Next` (`Qwen4ExpForConditionalGeneration`, `model_type: "qwen4_exp"`)** on **both 1× and 2× RTX 3090 Ti (`sm_86`)** in `warp-34`, built on the `CudaDeviceProps` / `w34_device_props` multi-arch (`sm_86,sm_89`) foundation so a subsequent RTX 4090 (`sm_89`) tuning pass requires no structural rework.
>
> **Strategic Scope:** `Qwen3.8-Flash-Next` is the first open-weight release of the **Qwen4 architecture family** (`qwen4_exp` / `qwen4_exp_text`). Every new primitive introduced here — **HyperConnection (`GatedResidual`)**, **Qwen Sparse Attention (`QSA`)**, **N-Gram Per-Layer Embeddings (`PLE`)**, **Tiered 512-Expert MoE Offloading**, and **Multi-Stream `Qwen4` `MTP`** — is designed as a first-class, reusable `warp-34` subsystem that composes cleanly with our existing `CudaDeviceProps` device abstraction, `KvPool`, `StateBundlePool`, `MambaRadixCache`, `CUDAGraphRunner`, and `w34-oracle` verification contracts.
>
> **Non-Goals:** No runtime abliteration / weight-modification hacks, no `llama.cpp` / GGUF side-quests, and no single-sequence-only shortcuts that break `warp-34`'s transactional scheduler or prefix cache.

---

## 1. Model Architecture & Delta vs. `Qwen3.5` / `Qwen3.6`

`Qwen/Qwen3.8-Flash-Next` (`180B` total parameters: `125B` backbone with `6B` activated per token + `51.2B` PLE n-gram table + `4B` MTP drafter) shares the 3:1 hybrid recurrence schedule and multimodal stem already proven in `warp-34` Phases 0–6, while introducing five Qwen4-generation architectural primitives.

### 1.1 Structural Overview (`config.json`)

| Parameter | Value | `warp-34` Status / Subsystem |
|---|---|---|
| `architectures` / `model_type` | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` (`qwen4_exp_text`) | **New** family registration in `loader.rs` & `model/` |
| `num_hidden_layers` | `48` (`12 × [3 × GDN → MoE, 1 × QSA → MoE]`) | Reuses hybrid 3:1 layer schedule |
| `hidden_size` / `vocab_size` | `2560` / `248,320` (`bos` / `eos` = `248044`) | Same vocab size & tokenizer family as Qwen 3.5 / 3.8 |
| `max_position_embeddings` | `262,144` (extensible to `1,000,000` via YaRN) | Supported by existing capacity & RoPE infrastructure |
| **Gated DeltaNet (`linear_attention`, 36 layers)** | `H_v = 48`, `H_qk = 16`, `head_dim = 128`, `conv_kernel = 4`, `FP32` `ssm_state` | **Already implemented** (`csrc/gdn/`, `StateBundlePool`) |
| **Vision (`Qwen3_VisionTransformer`) & 3D `mRoPE`** | `mrope_section = [11, 11, 10]`, `partial_rotary_factor = 0.25`, `theta = 1e7` | **Already implemented** (Phase 5 `vision.rs`) |
| **HyperConnection (`GatedResidual`)** | `hc_count = 4` streams (`[T, 4, 2560]`), `hc_lowrank = 320`, per-branch `GemmaRMSNorm` | **New** (`csrc/hc.cu`, `model/hc.rs`) |
| **Qwen Sparse Attention (`QSA`, 12 layers)** | Main GQA `24 Q / 2 KV` (`d = 256`, output sigmoid gate) + MQA Indexer (`4 Q / 1 KV`, `d = 128`, `compress_ratio = 4`, `budget = 2048` tokens = `512` blocks) | **New** (`csrc/qsa.cu`, `KvNamespace` extensions) |
| **N-Gram Per-Layer Embedding (`PLE`)** | `ple_layer_ids = [2]` (0-based layer `1`), `ngram_size = 3`, `heads_per_ngram = 8` (`16` heads × `160` dim = `2560`), `~20M` rows/head (`51.2B` params across `128` shards), dilated short-conv (`kernel = 4`, `dilation = 3`, `state_len = 9`) | **New** (`csrc/ple.cu`, host `PleMmapTable`, `StateBundlePool` extension) |
| **512-Expert Sparse MoE (`48` layers)** | `num_experts = 512`, `top_k = 10` routed + `1` shared expert, `moe_intermediate_size = 640`, `shared_expert_intermediate_size = 640` (`~62.8 GB` at `W4A16 g128`) | **New** tieredVRAM/RAM/CPU-PCIe execution engine extending `csrc/moe/` |
| **Qwen4 Multi-Token Predictor (`MTP`)** | `1` layer (`full_attention` QSA + `512`-expert `INT4 g32` MoE), `residual_linear_shared` (`fc_embedding` + `fc_hidden`), dual-stream output (`[T, 2560]` + `[T, 10240]`), `K = 3` (`M_verify = 4`) | **New** (`mtp.rs` Qwen4 variant) |

### 1.2 Canonical Checkpoint Target

Our primary reference checkpoint is [`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) (pinned revision `ef554143369a706525336f6b42a09094835dc077`), alongside compatibility with upstream `Qwen/Qwen3.8-Flash-Next` and standard AutoRound / GPTQ / Compressed-Tensors `W4A16` checkpoints:
- **Target Backbone (`~72.6 GiB`):**
  - Routed MoE experts (`48 × 512` experts, `gate_proj`/`up_proj`/`down_proj`): Intel AutoRound symmetric `W4A16` (`bits = 4`, `group_size = 128`, `~62.8 GiB` total = `~2.56 MiB` per expert, `~1.31 GiB` per layer).
  - Dense weights (GDN projections, QSA projections & indexer, shared experts, `GatedResidual` norms & projections, token embeddings, `lm_head`): `~9.8 GiB` total (`BF16` / `W4A16` per checkpoint `extra_config`).
- **PLE N-Gram Table (`~51.3 GiB`):** `128` shards (`ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`) in serialized `F8_E4M3` with per-tensor `BF16` `weight_scale`.
- **MTP Draft Module (`~3.85 GiB`):** 1-layer QSA + 512-expert MoE in symmetric `INT4` (`group_size = 32`), sharing `embed_tokens` and `lm_head` with the target model.

---

## 2. Hardware Budget & Unified Single-Card / Dual-Card Offloading Strategy

### 2.1 Host Hardware Envelope (`docs/HARDWARE.md`)

- **Primary Target GPUs:** **1× and 2× NVIDIA GeForce RTX 3090 Ti (`sm_86`, 84 SMs, 6 MiB L2, 24 GiB GDDR6X @ 1008 GB/s, 80 TF FP16-mma FP32-acc)** on PCIe Gen4 x8 (`topo PHB`, P2P disabled; cross-GPU communication uses `warp-34`'s `~18 µs` pinned-sysmem two-hop `TP=2` collectives).
  - All new `qwen4_exp` kernels (`hc.cu`, `ple.cu`, `qsa.cu`, `moe_offload.cu`) use `w34_get_current_device_props` / `CudaDeviceProps` (`multi_processor_count`, `l2_cache_bytes`) rather than hardcoding 84-SM or 6-MiB-L2 constants, keeping the codebase clean for a dedicated RTX 4090 (`sm_89`) optimization pass once 3090 Ti qualification is complete.
- **CPU:** **AMD Ryzen 9 7950X3D (16C / 32T, Zen 4)** with native **AVX-512** (`avx512f`, `avx512bw`, `avx512vnni`, `avx512_bf16`) + AVX2/FMA.
- **System Memory & Storage:** **124 GiB DDR5 RAM** + **124 GiB NVMe swap** + **3.7 TB PCIe Gen4 NVMe** (`2.5 TB` free).

### 2.2 Why a Unified 3-Tier Residency Design Works on Both `1×` and `2×` RTX 3090 Ti

A naive "load everything into RAM" approach (`58 GiB` pinned experts + `51.3 GiB` anonymous PLE RAM = `109.3 GiB` before OS/processes) risks swap thrashing on a 124 GiB host and fails outright on 64 GiB machines. Conversely, keeping only a static subset of experts in VRAM without dynamic adaptation or CPU/PCIe co-execution bottlenecks decode on PCIe transfers.

Synthesizing the best mechanisms from `DominikBucko/qwen38-flash-next-{3090,2x3090}` and `Niko1221/Strata` with `warp-34`'s native advantages (`TQ4`/`Q8_0` KV compression, pinned-sysmem `TP=2` collectives, and Zen 4 AVX-512) gives us a single unified 3-tier architecture:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: GPU VRAM (24 GiB on TP=1  |  2 × 24 GiB = 48 GiB on TP=2)                │
│  • Dense backbone (GDN, QSA, Shared Experts, HC, Embed/LM Head) + MTP            │
│  • Paged KV Cache (Main QSA KV in TQ4/Q8_0/FP16 + QSA Indexer Raw/Compressed KV) │
│  • StateBundlePool (36× GDN FP32 ssm_state + conv_state + 1× PLE conv_state)     │
│  • Per-Layer Hot Expert Cache:                                                   │
│      - TP=1 (1× 3090 Ti): K_hot = 32..48 experts/layer (~4.0..6.0 GiB)           │
│      - TP=2 (2× 3090 Ti): K_hot = 88..112 experts/layer/rank (~10.2..13.0 GiB/rk)│
│  • Double-buffered Prefill Expert Staging Arena (2 × G_stage experts)            │
└───────────────────────────────────────▲──────────────────────────────────────────┘
                                        │ PCIe Gen4 DMA + Pinned Bridge
┌───────────────────────────────────────▼──────────────────────────────────────────┐
│ TIER 2: Pinned System RAM (`HostExpertStore`, 38–58 GiB Hugepage / Mapped)       │
│  • Holds cold routed experts in dual-compatible layout (GPU Marlin tiles +       │
│    CPU-friendly contiguous blocks).                                              │
│  • Decode Miss Handling (unified across 1× and 2× 3090 Ti):                      │
│      1. GPU LRU + PCIe Pull: GPU pulls missed experts from pinned RAM over PCIe  │
│         and updates per-layer LRU slots on-device (`lru_map`).                   │
│      2. Hybrid CPU AVX-512 + GPU PCIe Share: Pinned CPU thread pool computes     │
│         `(1 - gpu_share_frac)` of cold routed experts in place while GPU runs    │
│         hot experts + `gpu_share_frac` over PCIe; partial sums merge in-graph.   │
│  • Prefill Handling (`total_tokens >= prefill_stream_threshold`):                │
│      Async H2D stream DMAs cold experts in groups of `80..128` two groups ahead  │
│      of GPU Grouped-MMA execution, overlapping PCIe with QSA/GDN compute.        │
└───────────────────────────────────────▲──────────────────────────────────────────┘
                                        │ Zero-Copy `mmap` + `madvise(MADV_WILLNEED)`
┌───────────────────────────────────────▼──────────────────────────────────────────┐
│ TIER 3: NVMe + Linux Page Cache (`PleMmapTable`, 51.3 GiB on Disk, 0 B Anon RAM) │
│  • 128 FP8 (`F8_E4M3`) PLE shards are memory-mapped read-only from HF cache.     │
│  • Each token needs only 16 rows × 160 B = 2,560 B.                              │
│  • Batched `MADV_WILLNEED` prefetches pages in parallel; Linux automatically     │
│    caches hot n-gram pages in unused system RAM (30–40+ GiB on our 124 GiB host) │
│    with zero swap pressure and zero startup copy time.                           │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Per-Card VRAM Budgets (`1× 3090 Ti` vs. `2× 3090 Ti`)

Because `Qwen3.8-Flash-Next` has only **12 full-attention (QSA) layers** with **2 KV heads** (`head_dim = 256`), its KV cache footprint is already `4×` smaller than `Qwen3.8-27B` (`16` layers × `4` KV heads):
- **Main QSA KV per token (all 12 layers):**
  - `FP16`/`BF16`: `12 × 2 × 2 × 256 × 2 B = 24,576 B/tok` (`~6.0 GiB` total at `256K` tokens; `3.0 GiB/rank` on `TP=2`).
  - `Q8_0` (`272 B/row`): `12 × 2 × 2 × 272 B = 13,056 B/tok` (`~3.19 GiB` total at `256K` tokens; `1.60 GiB/rank` on `TP=2`).
  - `TQ4` (`132 B/row` interior): **`~1.95 GiB` total at `256K` tokens** (`~0.98 GiB/rank` on `TP=2`).
- **QSA Indexer Side-Cache per token (all 12 layers, `1` KV head, `head_dim = 128`):**
  - Raw key cache (`128 × 2 B = 256 B`) + 4:1 compressed key cache (`256 B / 4 = 64 B`) = `320 B/tok/layer` → **`3,840 B/tok` across 12 layers** (`~0.96 GiB` at `256K` tokens).
- **GDN + PLE `StateBundle` per slot:**
  - `36` GDN layers (`H_v = 48`, `128×128` `FP32` `ssm_state` + `conv_state`) + `1` PLE short-conv (`10240 × 9` `FP16`): **`110.3 MiB / slot` on `TP=1`** (`55.2 MiB / rank / slot` on `TP=2`).

| Component | `TP=1` (1× RTX 3090 Ti, 24 GiB) | `TP=2` / `EP=2` (2× RTX 3090 Ti, 24 GiB/rank) |
|---|---:|---:|
| Dense backbone (`GDN`, `QSA`, Shared Experts, `HC`, `Embed`/`LM Head`) | `~9.6 GiB` | `~5.1 GiB / rank` |
| Native `MTP` draft weights (dense + hot MTP experts) | `~1.6 GiB` | `~1.2 GiB / rank` |
| Main QSA KV (`Q8_0` / `TQ4`) + QSA Indexer Cache (`128K–256K` context) | `~2.5–3.1 GiB` | `~1.9–2.6 GiB / rank` |
| `StateBundlePool` (active + spec + ping-pong + radix checkpoints) | `~0.9 GiB` | `~0.6 GiB / rank` |
| Prefill Staging + Workspace + CUDA Graphs | `~3.2 GiB` | `~2.8 GiB / rank` |
| **Hot Routed Expert Cache (`K_hot` per layer per rank)** | **`40` experts/layer (`~4.9 GiB`)** | **`96–104` experts/layer/rank (`~11.8–12.8 GiB/rank`)** |
| **Total Peak VRAM per GPU** | **`~22.7 GiB / 24.0 GiB`** | **`~23.4 GiB / 24.0 GiB`** |

> **Impact of `warp-34` KV Compression on Expert Hit Rate:** On `TP=2`, compressing main QSA KV from `BF16` (`4.13 GiB/rank`) to `Q8_0`/`TQ4` frees `~1.5–2.2 GiB/rank`, allowing `12–18` additional hot experts per layer per GPU (`192–208` total hot experts cached per layer across both GPUs out of `512`).

---

## 3. Detailed Subsystem Specifications

### 3.1 Foundational Seam Extensions (`docs/contracts/`)

1. **`PrecisionPolicy` (`docs/contracts/precision-policy.md`):**
   - `Qwen3.8-Flash-Next` checkpoints store `HyperConnection` norms/projections, QSA indexer weights, and routers in `BF16`.
   - Before freezing the `qwen4_exp` kernel ABI, Phase 1 `dump-diff` will verify whether `HyperConnection` 4-stream states and `PLE` gated activations remain strictly within `FP16` finite bounds (`< 65,504`) or require a native `BF16` activation spine (reusing `warp-34`'s existing `sm_86` native `BF16` MMA and norm templates built for `DFlash2`).
   - Persistent GDN `ssm_state` (`[36, H_v, 128, 128]`), `A_log`, `dt_bias`, router softmax, and `SamplerV1` logits remain strictly **`FP32`**.
2. **`KvPool` & `KvNamespace` (`docs/contracts/cache-layout.md`):**
   - Add `W34_KV_NAMESPACE_QSA_INDEXER_RAW` (`head_dim = 128`, `1` head, `B = 16`) and `W34_KV_NAMESPACE_QSA_INDEXER_COMPRESSED` (`head_dim = 128`, `1` head, 1 compressed row per `compress_ratio = 4` tokens) alongside `TargetFullAttn` and `MtpFullAttn`.
   - Logical page allocation remains unified: when a request allocates or shares a logical `page_id` in `MambaRadixCache`, the corresponding QSA indexer raw/compressed slices are tracked by the same logical page lifecycle and copy-on-write rules.
3. **`StateBundlePool` (`docs/contracts/state-transaction.md`):**
   - Extend `StateBundle` for models with `ple_layer_ids != []` to include:
     - `ple_conv_state`: `[num_ple_layers, hc_count * hidden_size, conv_state_len]` (`[1, 10240, 9]` in `FP16`/`BF16`, `184,320 B` per slot) plus speculative verify staging (`+ max_verify_tokens` columns).
     - `ple_ngram_context`: `[ngram_size - 1]` (`2 × int32` token IDs recording the last 2 committed tokens prior to the current step, reset to `eos_token_id` across segment boundaries).
   - Because `ple_conv_state` and `ple_ngram_context` live inside `StateBundle`, `MambaRadixCache` checkpoint donation, warm prefix restore, and speculative rejection rollback (`processed_len` vs `emitted_len`) work identically to GDN state with zero special-case scheduler logic.

---

### 3.2 HyperConnection (`GatedResidual`, `csrc/hc.cu`)

Each of the 48 decoder layers wraps its token mixer (`GDN` or `QSA`) and its `MoE` block in a 4-branch `GatedResidual` (`hc_count = 4`, `hidden_size = 2560`, `hyper_hidden_size = 10240`, `hc_lowrank = 320`):
- **State Layout:** Inter-layer hidden state has checkpoint-native shape `[T, 4, 2560]` (`HC` outer, `H` inner).
- **Checkpoint Weight Mapping:** At load time, `loader.rs` concatenates `hyper_connection.input_mix_weight_down.weight` (`[320, 10240]`) and `hyper_connection.block_inject_weight.weight` (`[4, 10240]`) into a single 16-row-aligned projection `[336, 10240]` (zero-padded in rows `324..336` for Tensor Core alignment).
- **Fused Kernel Pipeline (`combine_and_mix`):**
  1. **`w34_hc_combine_and_grouped_gemma_rmsnorm`:** For each token `t` and branch `b ∈ 0..4`:
     - If a pending `(prev_block_output: [T, 2560], prev_injection: [T, 4])` exists from the preceding sub-block, fuses `hidden_states[t, b, :] += (2.0 * sigmoid(prev_injection[t, b] / 4.0)) * prev_block_output[t, :]` (rounding to activation dtype before norm reduction).
     - Computes independent `FP32` RMS variance per branch `b` over `2560` elements and writes normalized output `xn[t, b, i] = hidden_states[t, b, i] * rrms * (1.0f + weight[b, i])`.
  2. **Low-Rank Down + Inject GEMM:** Skinny GEMM (`xn: [T, 10240] × W_down_inj^T: [10240, 336] → [T, 336]`), splitting `down: [T, 320]` and raw `inject_logits: [T, 4]`.
  3. **`w34_hc_silu` + `w34_hc_gate_mix`:** Computes `lora = SiLU(down / 4.0)`, multiplies by `input_mix_weight_up^T` (`[320, 10240]`), applies `sigmoid(gate) / 4.0` element-wise read gates to the normalized streams `xn: [T, 4, 2560]`, and sums over the 4 branches into `block_input: [T, 2560]` while returning `inject_logits: [T, 4]`.
- **Final Layer Mixer:** Before `lm_head` (no separate `model.language_model.norm` exists in `qwen4_exp`), `model.language_model.hyper_connection_mixer` (`use_combine = false`) runs `combine_and_mix` to materialize the last layer's combine into `multi_hidden: [T, 10240]` (retained as a `TapPlan` output for `Qwen4` `MTP`) and collapse it into `sample_hidden_states: [T, 2560]`.

---

### 3.3 N-Gram Per-Layer Embedding (`PLE`, `csrc/ple.cu` & `model/ple.rs`)

At layer index `1` (`ple_layer_ids = [2]` in 1-based config indexing), before `attn_hyper_connection.mix`:
1. **Materialize Pending HC Combine:** Any pending `(prev_block_output, prev_injection)` from Layer 0's MoE is combined into `hidden_states: [T, 10240]` first.
2. **Deterministic N-Gram Hashing (`ngram_size = 3`, `heads_per_ngram = 8` → `16` heads):**
   - Uses exact 64-bit `splitmix64` multipliers seeded from `seed + 10007 * ple_dense_layer_id` and per-head prime moduli `ngram_heads_vocab_sizes[0..16]` (`_nth_prime_after(19_999_999, head + 1)`).
   - Prepends each request's 2-token `ple_ngram_context` to its scheduled `input_ids`, resets shifts across `eos_token_id` (`248044`) boundaries, and computes `[T, 16]` global row IDs into the `~320M`-row table (`head_dim = 160`).
3. **Zero-Copy `PleMmapTable` Gather + Async H2D:**
   - Groups the `16 * T` row IDs by shard (`0..128`), issues coalesced `madvise(MADV_WILLNEED)` across unique 4 KiB OS pages, gathers `16 × 160 B` (`2,560 B/tok`) of `F8_E4M3` bytes into a pinned host staging buffer using the CPU thread pool, and copies to GPU (`2.56 KB` for `M=1` decode, `10.24 KB` for `M=4` verify, `20 MiB` for an `8192`-token prefill chunk).
   - For CUDA Graph capture/replay (`M=1` decode and `M=4` verify), the host prepares the n-gram IDs from the known input token IDs (or runs a lightweight host-launch callback / pre-step gather for known anchor tokens, and in-graph pinned gather for draft tokens) into fixed-address device buffers.
4. **GPU PLE Gating & Dilated Short-Conv (`csrc/ple.cu`):**
   - Dequantizes `F8_E4M3` embeddings via `weight_scale` into `[T, 2560]`.
   - Computes `key = key_proj(embeddings)` (`[T, 4, 2560]`) and `value = value_proj(embeddings)` (`[T, 2560]`).
   - Applies grouped `GemmaRMSNorm(group_size = 2560)` to `key` and `query = hidden_states.view(T, 4, 2560)`.
   - Computes branch-wise scalar gate:
     $$g = \frac{\langle \text{norm\_key}, \text{norm\_query} \rangle}{\sqrt{2560}}, \quad \text{gate} = \sigma\!\left(\text{sign}(g)\sqrt{\max(|g|, 10^{-6})}\right)$$
   - Broadcast-multiplies `gated_value = gate * value.unsqueeze(1)` (`[T, 4, 2560]`), normalizes with `norm_conv`, and runs the **stateful dilated depthwise `conv1d` (`kernel = 4`, `dilation = 3`, `conv_state_len = 9`, `SiLU` activation)** updating `StateBundle.ple_conv_state`.
   - Adds `gated_value + conv_out` into `hidden_states: [T, 10240]`.

---

### 3.4 Qwen Sparse Attention (`QSA`, `csrc/qsa.cu`)

On the 12 full-attention layers (`layer_idx ∈ {3, 7, 11, ..., 47}`):
1. **Fused Pre-Indexer (`w34_qsa_pre_indexer`):**
   - Projects `index_qk_proj(block_input)` (`[T, 2560] → [T, (4 + 1) × 128]`).
   - Applies `q_layernorm` (`GemmaRMSNorm(128)`) and `k_layernorm` (`GemmaRMSNorm(128)`), followed by partial interleaved 3D `mRoPE` (`rotary_dim = 64`, `mrope_section = [11, 11, 10]`, `theta = 1e7`).
   - Writes raw indexer keys into `QsaRawKeyCache` (`[B=16, 128]`) and incrementally updates `QsaCompressedKeyCache` (`4:1` mean-pooling over completed 4-token micro-blocks).
2. **Block Top-K Selection (`w34_qsa_select_blocks`, `budget = 2048` tokens = `512` blocks):**
   - For sequences where `seq_len <= 2048`, bypasses scoring and selects all valid causal tokens directly.
   - For `seq_len > 2048`, computes MQA dot-product scores between the `4` indexer query heads and the `QsaCompressedKeyCache` micro-blocks (`1` key per `4` tokens), aggregates across the 4 query heads, forces inclusion of the local causal tail/sink blocks, selects the top `512` blocks (`2048` token indices), and writes sorted/valid logical token positions `topk_indices: [T, 2051]`.
   - **Bounded Workspace:** Scores are computed in bounded tiles (`<= 64 MiB` scratch) so `256K` prefill chunks never spike VRAM.
   - **MTP Index Reuse:** During `MTP` drafting (`K = 3`), draft step 0 computes `topk_indices` and steps 1..2 reuse the step-0 selected block indices while appending the new draft positions to the QSA side-cache.
3. **Sparse Paged GQA (`w34_qsa_sparse_paged_attn`):**
   - Projects main `qkv_proj` (`24 Q` heads with packed sigmoid gate + `2 KV` heads, `head_dim = 256`; on `TP=2`, `12 Q` heads and `1 KV` head per rank).
   - Runs `fused_qk_rmsnorm_rope_gate` and appends main `K, V` to the `TargetFullAttn` paged cache (`FP16`/`BF16`, `Q8_0`, or `TQ4`).
   - Gathers only the `<= 2051` selected logical token positions per query via `block_tables[req, logical_pos / 16]` and computes GQA `12:1` attention with online `FP32` softmax, `out *= sigmoid(gate)`, and `o_proj` (+ `TP=2` AllReduce).

---

### 3.5 Tiered 512-Expert MoE Engine (`csrc/moe/` & `model/moe_offload.rs`)

Each of the 48 layers routes every token to `top_k = 10` of `512` experts (`gate_up: [2560, 1280]`, `down: [640, 2560]`, `W4A16 g128` = `2,621,440 B` packed weights + `40,960 B` scales = **`2.54 MiB` per expert**) plus `1` GPU-resident shared expert (`shared_expert` + `shared_expert_gate`).

#### A. Multi-GPU Sharding (`TP=1` vs. `TP=2 / EP=2`)
- **Single Card (`TP=1`):** Owns all `512` experts per layer (`K_hot = 32..48` in VRAM; remaining `464..480` in `HostExpertStore`).
- **Dual Card (`TP=2` with `EP=2` on Routed Experts):**
  - Rank 0 owns experts `0..256`; Rank 1 owns experts `256..512`.
  - Each rank caches `K_hot = 88..112` of its `256` local experts in VRAM (`176..224` hot experts cached per layer across the two cards!) and stores only its remaining `144..168` cold experts in its NUMA/CCD-local `HostExpertStore` (`~16.6–19.4 GiB` per rank = **`~33–39 GiB` total pinned RAM** in hot-only ownership mode, or `~58 GiB` in full dynamic-LRU mode).
  - The post-MoE `TP=2` pinned-sysmem `AllReduce` sums both ranks' routed + shared expert outputs in a single `~18 µs` exchange (no extra collective needed for `EP=2`).

#### B. Decode & Verify (`M <= 16` tokens): Adaptive Dual-Path Execution
We support two complementary cold-expert decode mechanisms controlled by `CapacityPlanner` (and CLI flags `--moe-offload-mode {auto,gpu-lru,cpu-hybrid}`):
1. **Dynamic GPU LRU + PCIe Pull (`gpu-lru`, default on `2× 3090 Ti` with `>= 96 GiB` RAM):**
   - Each layer maintains on-device `cache_map[512]`, `slot_global_ids[K_hot]`, and `slot_ages[K_hot]` initialized from a static popularity profile (`static_hot_cache_rankings.json`).
   - An integer-only GPU kernel (`w34_moe_lru_update`) identifies unique local expert misses in the current step (`M = 1` or `M = 4` MTP verify), selects non-protected LRU victim slots, issues async PCIe reads from pinned `HostExpertStore` into the victim VRAM slots (overlapping down-proj transfers with gate/up compute), and runs the GPU `w34_moe_w4a16` kernel over all active experts.
2. **CUDA-Graph-Safe Hybrid CPU AVX-512 + GPU PCIe Co-Execution (`cpu-hybrid`, default on `1× 3090 Ti` and `< 96 GiB` RAM):**
   - For each layer, the router kernel partitions selected `(token, expert)` pairs into:
     - **Hot GPU set:** Experts resident in the GPU's `K_hot` cache.
     - **GPU PCIe share (`gpu_share_frac`, default `0.22–0.30`):** Top-ranked cold experts read directly over PCIe by a zero-copy GPU kernel while the CPU works.
     - **CPU cold set:** Written into a fixed-address pinned request ring buffer (`bridge_cuda.cu`), signaling a persistent **pinned Zen 4 AVX-512 / AVX2 worker pool** (`1` thread per physical core) that computes the `W4A16` expert GEMVs in `FP32` straight from `HostExpertStore` and writes the partial sum back to a pinned response slot.
   - Because the GPU bridge uses device-side atomic/volatile polling on mapped pinned memory, **the entire 48-layer decode / `M=4` verify step remains inside a single captured CUDA Graph**.

#### C. Chunked Prefill (`total_tokens >= 512`, e.g., `4096` or `8192` tokens): Double-Buffered GPU Streaming
- Large prefill chunks activate nearly all 512 experts per layer, making CPU GEMM uncompetitive and making per-chunk PCIe streaming optimal (`1.31 GiB` per layer on `TP=1`, `670 MiB` per layer per rank on `TP=2`).
- For each MoE layer during prefill:
  1. Hot experts in VRAM execute immediately via Grouped-MMA `w34_moe_w4a16`, accumulating into an `FP32` output buffer `[T, 2560]`.
  2. Simultaneously, a dedicated H2D DMA stream copies cold experts from `HostExpertStore` into a 2-slot GPU staging buffer in groups of `G_stage = 80..128` experts (`~200–320 MiB`), **pipelined 2 groups ahead** across sub-block and layer boundaries.
  3. Because `HostExpertStore` already stores weights in `warp-34`'s repacked Marlin tile layout, **no on-GPU weight repack kernel is needed after DMA** — the compute stream immediately launches Grouped-MMA `w34_moe_w4a16` on the staged group and accumulates into the layer's `FP32` output buffer.

---

### 3.6 Native `Qwen4` `MTP` Speculative Decoding (`mtp.rs`)

`Qwen3.8-Flash-Next` includes a native 1-layer Multi-Token Predictor trained for multi-step drafting (`K = 3` draft tokens → `M_verify = 4` target verification rows):
- **Dual-Stream Input Contract:**
  - Unlike `Qwen3.5` MTP (which concatenates `[target_hidden, embed(token)]` into `eh_proj: Linear(2H, H)`), `Qwen4` MTP receives the **pre-final-mixer 4-stream HyperConnection state `[T, 4, 2560]`** from the target model alongside `embed_tokens(token_id): [T, 2560]`.
  - Applies `residual_linear_shared` (`fc_embedding` + per-branch `fc_hidden`), runs 1 `Qwen4ExpDecoderLayer` (`QSA` full attention + `512`-expert `INT4 g32` MoE with its own hot cache / offload path), and outputs both:
    - Collapsed `[T, 2560]` (via `final_hyper_connection` + `norm`) → `lm_head` to sample the next draft token.
    - Uncollapsed `[T, 4, 2560]` fed recurrently into the next draft step (`step = 1..K`).
- **Draft Vocabulary Filtering (Optional Fast Path):**
  - Supports restricting draft `lm_head` evaluation to the active token range `[[0, 65536], [248044, 248320]]` (`~26%` of full vocab) while the target model verifies all `K` proposals against the full `248,320` vocabulary, preserving 100% lossless target distribution parity.

---

## 4. Phased Implementation & Verification Plan

Every phase follows `warp-34`'s engineering rules: no phase graduates without passing `w34 dump-diff` (tensor parity), `w34 greedy-diff` (token parity), and `w34 profile-diff` (kernel timing & throughput verification).

### Phase QF-0: Checkpoint Acquisition, Golden Oracle Dumps, & Config/Loader Plumbing
1. **Checkpoint & Golden Dumps:**
   - Acquire `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` (`ef554143369a706525336f6b42a09094835dc077`) into the local HF cache (`/home/nchapman/.cache/huggingface/hub/`).
   - Add `tools/oracle/qwen4_exp_dump.py` to dump reference per-op tensors for:
     - Layer 0 (`HyperConnection` + `GDN` + `512-expert MoE`)
     - Layer 1 (`PLE` n-gram hash/lookup/gate/dilated-conv + `HyperConnection` + `GDN` + `MoE`)
     - Layer 3 (`HyperConnection` + `QSA` indexer/top-k/sparse-attn + `MoE`)
     - `MTP` draft step 0..2 and `M_verify = 4` target verification.
   - Verify activation dynamic ranges across all 48 layers to lock the `FP16` vs. `BF16` spine precision rule for `qwen4_exp`.
2. **Config & Three-Tier Loader (`crates/w34-host/src/loader.rs`):**
   - Parse `model_type: "qwen4_exp"` / `"qwen4_exp_text"` and `Qwen4ExpForConditionalGeneration`.
   - Implement shard-streaming `HostExpertStore` (repacking `W4A16` experts shard-by-shard into pinned memory without exceeding physical RAM) and zero-copy `PleMmapTable` (`mmap` + `MADV_RANDOM` + `MADV_WILLNEED`).
   - Update `CapacityPlanner` to size `K_hot`, `HostExpertStore`, `QsaRawKeyCache`, `QsaCompressedKeyCache`, and `StateBundle` (`+ ple_conv_state`).
- **Exit Gate:** Loader unit tests and `CapacityPlanner` validate both `TP=1` (24 GiB) and `TP=2` (2× 24 GiB) memory plans against `Qwen3.8-Flash-Next-W4A16-FP8PLE` with zero swap thrashing.

### Phase QF-1: `HyperConnection`, `PLE`, & `QSA` Kernels + Per-Op Parity (`dump-diff`)
1. **`csrc/hc.cu`:** Implement `w34_hc_combine_and_grouped_gemma_rmsnorm`, merged low-rank projection, and `w34_hc_silu_up_sigmoid_mix`.
2. **`csrc/ple.cu` & `model/ple.rs`:** Implement 64-bit `splitmix64` prime-modulo n-gram hashing, `PleMmapTable` parallel gather, FP8 dequantization, signed-sqrt sigmoid gating, and stateful dilated short-conv (`kernel=4, dilation=3, state_len=9`) across prefill, decode, and speculative rollback.
3. **`csrc/qsa.cu`:** Implement `w34_qsa_pre_indexer` (4Q/1K norm + partial mRoPE + raw/4:1 compressed cache write), bounded-memory `w34_qsa_select_blocks` (`budget = 512` blocks), and `w34_qsa_sparse_paged_attn` (`FP16`/`BF16`, `Q8_0`, and `TQ4`).
- **Exit Gate:** 100% of `HC`, `PLE`, and `QSA` kernels pass `w34 dump-diff` against reference dumps (`max_abs_err` / `rel_l2` / cosine similarity) and pass `cuobjdump --dump-resource-usage` (`STACK:0 LOCAL:0`).

### Phase QF-2: Tiered 512-Expert MoE Engine (`TP=1` & `TP=2 / EP=2`)
1. **Pre-Repacked Pinned `HostExpertStore` & Double-Buffered Prefill Streaming:**
   - Implement 2-group-ahead async H2D DMA streaming + Grouped-MMA `w34_moe_w4a16` accumulation into `FP32` without on-GPU repack overhead.
2. **Decode Offload Paths:**
   - Implement on-GPU `w34_moe_lru_update` + async PCIe expert pull (`gpu-lru` mode).
   - Implement pinned CPU Zen 4 AVX-512 / AVX2 expert worker pool + CUDA-graph-safe device bridge (`bridge.cu`) + concurrent GPU PCIe cold-expert share (`cpu-hybrid` mode).
- **Exit Gate:** Single-layer and multi-layer 512-expert MoE microbenchmarks pass `dump-diff` across `gpu-lru`, `cpu-hybrid`, and `stream-prefill` modes on both `TP=1` and `TP=2`.

### Phase QF-3: End-to-End `Qwen4Exp` Model Spine, CUDA Graphs, & `MambaRadixCache`
1. Assemble the 48-layer `Qwen4Exp` forward pass in `crates/w34-host/src/model/` for `TP=1` and `TP=2`.
2. Wire `StateBundlePool` (`GDN` + `PLE` state) and `KvPool` (`TargetFullAttn` + `QsaRawKeyCache` + `QsaCompressedKeyCache`) into `MambaRadixCache` and `SchedulerV1`.
3. Capture composite CUDA Graphs (`M=1` decode and `M_verify = 4` verify) and register all new kernels in `prepare_execution_plan()` for deadlock-free cold-start `TP=2` initialization.
- **Exit Gate:**
  - `w34 greedy-diff` passes on canonical prompts across `TP=1` and `TP=2`.
  - Multi-turn warm prefix hits via `MambaRadixCache` restore `GDN`, `PLE`, and `QSA` caches cleanly and pass **Decisive Gate Test #5** (synthetic speculative rejection across chunk boundaries).

### Phase QF-4: Native `Qwen4` `MTP` Speculative Decoding & Long-Context (`128K–256K`) Qualification
1. Implement the `Qwen4` `MTP` drafter (`K = 3`, `M_verify = 4`) with dual-stream HC state passing, QSA step-0 top-k index reuse, and optional draft-vocab range filtering.
2. Qualify `128K` and `256K` long-context prefill, needle retrieval, and sustained decode on both **1× RTX 3090 Ti** and **2× RTX 3090 Ti**, recording `nsys` `profile-diff` and end-to-end throughput in `docs/STATE.md`:
   - **1× RTX 3090 Ti (`TP=1`) Target:** `> 2,200 tok/s` prefill, `> 50 tok/s` decode at `128K` context (`256K` supported with `Q8_0`/`TQ4` KV).
   - **2× RTX 3090 Ti (`TP=2 / EP=2`) Target:** `> 3,800–4,200 tok/s` prefill, `> 115–135 tok/s` decode across `128K–256K` context without requiring CUDA P2P.

---

## 5. Measured Performance vs. Reference Headline Numbers (`QF-5`)

### 5.1 Measured Results on `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` vs. Reference READMEs

> **Hardware note:** `warp-34` is measured on **2× RTX 3090 Ti over PCIe 4.0 x8 (`~12.5 GB/s` per card, `topo PHB`, P2P disabled)**, whereas the reference `DominikBucko/qwen38-flash-next-*` benchmarks were recorded on a **Threadripper PRO 5975WX workstation with 2× RTX 3090 over full PCIe 4.0 x16 (`~25 GB/s` per card, 2× the per-card PCIe cold-expert bandwidth)**.

| Workload & Protocol | `warp-34` Measured (`TP=2 / EP=2` `hot112` + `MTP-3` `hot128`, **PCIe 4.0 x8**) | Reference `vLLM` Overlay (**PCIe 4.0 x16**, Threadripper PRO 5975WX) | Comparison & Notes |
| :--- | :--- | :--- | :--- |
| **`benchmark_serving.py` `short-chat` (`128` in + `256` out)** | **`91.28 tok/s`** (`0.42 s` TTFT, `2.23/3` accepted = `3.23 tok/round`) | **`78.73 tok/s`** (`hot100`) / **`80.08 tok/s`** (`hot104`) *(hillclimb `128:256`)* | **`+14.0%` faster than reference `hot104`** on the identical `128:256` shape. |
| **`benchmark_serving.py` `short-chat` (`128` in + `512` out)** | **`92.10 tok/s` cold** (`0.55 s` TTFT)<br>**`88.50 tok/s` warm** (`0.39 s` TTFT) | **`77.29 tok/s`** (`74.75–79.71 tok/s`, `0.90 s` TTFT) *(`2026-09-05` `short-chat`)* | **`+14.5%–19.2%` faster decode** and **`2×` faster TTFT** (`0.39–0.55 s` vs `0.90 s`). |
| **5-Prompt Diverse `chat-512` Suite (`512` out each, cold per task)** | **Mean `81.07 tok/s`** (`72.45 – 92.76 tok/s`):<br>• `creative_writing`: **`74.04 tok/s`** (`1.44/3`)<br>• `factual_essay`: **`72.45 tok/s`** (`1.54/3`)<br>• `technical_explain`: **`80.11 tok/s`** (`1.78/3`)<br>• `code_generation`: **`85.96 tok/s`** (`1.97/3`)<br>• `reasoning_math`: **`92.76 tok/s`** (`2.50/3`) | **Mean `91.2 tok/s`** (`64.8 – 106.7 tok/s` on PCIe x16):<br>• `creative_writing`: `64.8 tok/s` (`1.66/3`)<br>• `factual_essay`: `75.9 tok/s` (`1.80/3`)<br>• `technical_explain`: `97.2 tok/s` (`2.16/3`)<br>• `code_generation`: `106.7 tok/s` (`2.46/3`)<br>• `reasoning_math`: `104.2 tok/s` (`2.44/3`) | Beats reference on `creative_writing` (`+14%`); within `~11%` overall mean despite half the PCIe link width (`x8` vs `x16`) during cold-expert pulls. |
| **`scripts/bench.py` Code Review (`4096:256` & `8192:1024`)** | • `4096:256`: **`1,050.9 tok/s` prefill** (`3.93 s` TTFT), **`65.61 tok/s` decode** (`2.23/3`)<br>• `8192:1024`: **`1,120.0 tok/s` prefill** (`7.34 s` TTFT), **`65.61 tok/s` decode** (`2.23/3`) | • `4096:256`: `1,432.0 tok/s` prefill (`2.88 s` TTFT), `108.47 tok/s` decode (`2.20/3`)<br>• `8192:1024`: `2,506.5 tok/s` prefill (`3.28 s` TTFT), `121.80 tok/s` decode (`2.12/3`)<br>• `64 GB hot88`: `78.2–79.7 tok/s` decode | Long-code review touches a broader cold-expert set per step, making decode and streamed prefill directly proportional to PCIe link bandwidth (`x8` vs `x16`). |
| **Short Completion Probe (`5` in + `64` out, `TP=2 / EP=2`)** | **Target-only (`M=1`):** `46.83 tok/s` cold / **`65.41 tok/s` warm**<br>**With `MTP-3`:** **`108.24 tok/s` cold** / **`132.47 tok/s` warm** (`2.86 tok/round`) | **Target-only (`M=1`):** `61.32 tok/s` (warmed)<br>**With `MTP-3`:** `119.9 – 135.2 tok/s` (warmed) | Matches the reference warmed peak (`132.47 tok/s` vs `119.9–135.2 tok/s`) with 100% lossless 64-token greedy parity. |
| **1× GPU (`TP=1`) Decode (`hot48`, Mapped Zero-Copy PCIe)** | **`24.35 tok/s`** (cold 16-tok `M=1`)<br>**`29.34 tok/s`** (warm 64-tok `M=1`)<br>**`34.05 tok/s`** (warm 64-tok `MTP-3`, `3.00 tok/round`) | **`34 – 36 tok/s`** (8-core 1-CCD)<br>**`43 – 51 tok/s`** (32-core Threadripper AVX2 CPU cold-expert pool) | Matches the 1-CCD reference (`34.05 tok/s` vs `34–36 tok/s`) using pure GPU zero-copy PCIe LRU. |
| **Warm `MambaRadixCache` Prefix Hit (`64` tokens)** | **`21.65 ms` TTFT (`TP=2`)**<br>**`84.45 ms` TTFT (`TP=1`)** | **Unsupported in `2x3090` base** (or single-sequence only) | `warp-34` snapshots both GDN (`ssm_state`/`conv_state`) and PLE (`ple_conv_state`/`ple_history_tokens`) inside `StateBundlePool` plus QSA indexer caches in `KvPool`. |
| **Cold Checkpoint Load Time** | **`32.3 s` (`TP=1`) / `39.4 s` (`TP=2`) + `2.1 s` (`MTP-3`)** | **`~240 s`** (`~4 min` container startup) | `warp-34` loads and transposes `[K/8, N] -> [N, K/8]` expert shards in parallel across 16 worker threads. |

### 5.2 Landed `QF-5` Optimizations & Remaining Optional Enhancements

1. **[Landed] Auto-Discover & Load `runtime/mtp-int4-g32` on Real Checkpoints (`crates/w34-host/src/mtp.rs`, `crates/w34-host/src/moe_offload.rs`):**
   - `Qwen3_5MtpRank::load` automatically falls back to `<model_dir>/runtime/mtp-int4-g32` when `mtp.*` tensors are absent from the root index, and `HostExpertStore::load` natively supports `compressed-tensors` `pack-quantized` (`weight_packed: [N, K/8]`, `weight_scale: [N, K/32]`, symmetric `INT4 group_size=32`) alongside `GPTQ`/`AutoRound` (`qweight`/`scales`), loading all 512 MTP experts in `1.70 s`.
2. **[Landed] Pre-Seed `HotExpertCache` from `static_hot_cache_rankings.json` (`crates/w34-host/src/moe_offload.rs`):**
   - `HotExpertRankings::from_model_dir` automatically probes `static_hot_cache_rankings.json` and `runtime/static_hot_cache_rankings.json`, filters and deduplicates global expert IDs per `EP` rank (`hot48` on `TP=1`, `hot88` on `TP=2 / EP=2`, `hot64` on `MTP-3`), and pre-warms `HotExpertCache` at startup.
   - `HotExpertCache::sync_from_device_if_dirty` synchronizes device LRU metadata before host-fallback prefill execution so CUDA Graph replays (`decode_graph` / `spec_verify_graph`) and eager prefill chunks remain 100% coherent.
3. **[Landed] CUDA-Graph-Safe 16-Core AVX-512F/F16C Hybrid Cold-Expert Bridge (`crates/w34-kernels/csrc/ops/op_moe_cpu_bridge.cu`, `crates/w34-host/src/moe_offload.rs`):**
   - **In-Graph System-Atomic Mailbox (`w34_moe_hybrid_submit_and_pull` + `w34_moe_hybrid_wait_and_add`):** For single-GPU (`TP=1`, `k_hot = 80`, `k_static = 78`), `moe_hybrid_plan_and_submit_kernel` partitions routed experts inside a captured CUDA Graph into:
     1. **Static hot hits (`slots[0..78]`):** Executed on the GPU.
     2. **Transient PCIe cold share (`slots[78..80]`, `gpu_share_frac = 0.14`):** Pulled directly over zero-copy PCIe (`moe_hybrid_pull_mapped_experts_kernel`) without evicting any static hot expert.
     3. **CPU DDR5 cold share:** Published via `st.release.sys.global.b64` to a persistent 16-core Zen 4 worker pool (`CpuWorkerPool` with `futex` idle sleep and 3.5 ms spin window) while the GPU concurrently runs `launch_moe_w4a16`, followed by `moe_hybrid_wait_and_add_kernel` (`ld.acquire.sys.global.b64` + fused residual add).
   - **512-Bit `__m512` `u4b8` Nibbles-at-Once Kernels & 3D V-Cache Packing:**
     - `prepare_x_perm` permutes each 128-element activation group into `dst[b * 16 + l]` so each of the 8 nibble shifts (`B = 0..7`) unpacks all 16 `uint32_t` words of a 64-byte cache line in a single `__m512i` register (`_mm512_loadu_si512` $\rightarrow$ `_mm512_srli_epi32` + `_mm512_and_si512` + `_mm512_cvtepi32_ps` + `_mm512_fmadd_ps`).
     - `eval_four_rows_u4b8<1>` evaluates 2 gate + 2 up rows (or 4 down rows) simultaneously in **18 of 32 ZMM registers** with zero register spills, and `eval_four_rows_tile_u4b8` evaluates up to 8 routed tokens per cold expert in a single weight pass during prefill.
     - `compute_prefill_cpu` packs per-job outputs contiguously by `job_row_off[j]` (`~71 MB` instead of `240 MB`), keeping the buffer resident in the Ryzen 9 7950X3D's **96 MB 3D V-Cache L3**, and parallelizes F16C (`_mm256_cvtph_ps` / `_mm256_cvtps_ph`) activation conversion and reduction across all 16 cores.
   - **Zero-Extra-PCIe Prompt-Frequency Hot-Slot Promotion in `StreamedPrefill`:**
     - On long prompts (`num_tokens >= 1024`), `StreamedPrefill` ranks cold misses by their frequency in the current prompt, places up to 16 cold experts that are $\ge 3\times$ more frequent than the lowest-frequency slots in `slots[58..78]` at the start of `gpu_misses` (sorted by expert ID so host DDR5 reads remain sequential), and unpacks them directly from `ws.d_stage_packs[0]` into `cache.slots[58..78]` inside GDDR6X (`1,008 GB/s`, `~30 µs/layer`, **zero extra PCIe traffic**).
4. **[Landed] Parallel QSA Block Selection + GQA-6 Shared-Tile & Split-K `TQ4` Attention (`crates/w34-kernels/csrc/ops/op_qsa.cu`):**
   - Replaced serial `threadIdx.x == 0` selection in `qsa_select_blocks_f16_kernel` with 128-thread warp-collaborative scoring and parallel radix/threshold top-`k` selection.
   - Added `qsa_sparse_paged_attn_gqa_f16_kernel` so all **6 query heads sharing a KV head** run in one CTA and dequantize each selected 4-token `TQ4` (`WHT-256`) KV block **once into shared memory** (`6×` fewer `TQ4` dequantizations), plus `qsa_sparse_paged_attn_splitk_merge_f16_kernel` for Split-K decode/MTP verify (`M <= 4`).
5. **[Landed] Single-Pass 6,144-Token Prefill, Double-Buffered H2D DMA, Parallel PLE Stream Overlap, & Prefix-Primed `MTP` Memoization (`crates/w34-host/src/{moe_offload.rs,ple.rs,mtp.rs,proposer.rs,model/rank.rs}`):**
   - Expanded the prefill scratch arena to `1056 MiB` (`<= 8192` KV pages) with `hc_extra` accounting in `ordinary_prefill_token_capacity`, enabling single-pass `6,144`-token prefill (`stage_capacity = 96`) and double-buffered async H2D DMA on `ws.h2d_stream`.
   - Vectorized `PleMmapTable` host gathers across a persistent worker pool on a dedicated `ple_stream`, raised single-layer `MTP` `k_hot` on `TP=1` from `64` to `192` with single-chunk MTP prefill, and added a 16-entry `PrefixMemo` cache (`256 MiB` budget) with branching-prefix checkpoint donation (`mamba_branching_seqlen`).

### 5.3 Standalone HTTP Serving Campaign Sweep (`benchmarks/perf/campaigns/flash-next-serving.toml`)

With the optimizations in §5.2, `warp-34` sweeps **5/5** serving workloads on **both** 1× RTX 3090 Ti (`warp34-flash-tp1-mtp3` vs `qfn-3090-mtp3`) and 2× RTX 3090 Ti (`warp34-flash-tp2-mtp3` vs `qfn-2x3090-mtp3`), plus `cold-read-60k` (`57,611` prompt tokens):

| Workload | `warp34-flash-tp1-mtp3` (1× 3090 Ti) | `qfn-3090-mtp3` (1× 3090 Ti) | `warp34-flash-tp2-mtp3` (2× 3090 Ti) | `qfn-2x3090-mtp3` (2× 3090 Ti) |
|---|---:|---:|---:|---:|
| `short-chat` | **32.41–33.56 tok/s** (`+5.2%–8.9%`) | 30.82 tok/s | **69.25 tok/s** (`+23.2%`) | 56.20 tok/s |
| `prefill-context-r16` | **18.37–18.54 tok/s** (`+79.6%`) | 10.23 tok/s | **27.09 tok/s** (`+27.4%`) | 21.27 tok/s |
| `code-context-r64` (~6K prompt) | **21.51–21.90 tok/s** (`+1.3%–3.2%`) | 21.23 tok/s | **28.81 tok/s** (`+2.2%`) | 28.20 tok/s |
| `prefix-primed-r16` (warm prefix) | **28.67 tok/s** (`+181.9%`, `2.82×`) | 10.17 tok/s | **45.90 tok/s** (`+123.7%`, `2.24×`) | 20.52 tok/s |
| `coding-agent-replay` (strict replay) | **27.25–29.78 tok/s** (`8/8` passed) | Failed (`0/8` HTTP 400) | **55.49 tok/s** (`8/8` passed) | Failed (`0/8` HTTP 400) |
| `cold-read-60k` (`57,611` prompt) | **6/6** retrieval through ~252K | Not supported | **68.2 s** (`0.938 tok/s`, `1.471` steady) | 72.3 s (`0.885 tok/s`) |

---

## 6. Upstream `vLLM` (`origin/main @ 190df0b90`) Cross-Check: Intrinsic `Qwen4` Architecture vs. `1–2× RTX 3090 Ti` Hardware Optimizations

To keep `warp-34`'s model spine clean, composable, and maintainable across `Qwen3.5`, `Qwen3.6`, and `Qwen3.8` (`qwen4_exp`), we explicitly separate **intrinsic `qwen4_exp` architectural semantics** (defined by the checkpoint and cross-checked against upstream `vLLM` `vllm/models/qwen4_exp/{common,nvidia}/**` at `190df0b90`) from **consumer-GPU (`sm_86`, 24 GiB VRAM) memory and compute adaptations**.

### 6.1 Composable Architectural Capability Axes (`crates/w34-host/src/model/`)

Instead of branching on `if is_qwen4_exp()` across `forward.rs` and `mtp.rs`, `warp-34` represents the `Qwen3.5 / 3.6 / 3.8` family as five orthogonal capability types driven by `ModelConfig`:

| Capability Axis | `Qwen3.5` / `Qwen3.6` Variant | `Qwen3.8-Flash-Next` (`qwen4_exp`) Variant | Upstream `vLLM` (`190df0b90`) Equivalent |
| :--- | :--- | :--- | :--- |
| **Inter-Layer Residual (`LayerResidual`)** | `SingleStream { input_layernorm_ptr, post_attention_layernorm_ptr }` (`[T, H]`) | `HyperConnection { attn_hc, mlp_hc }` (`[T, hc_count, H]`) | `Qwen4ExpDecoderLayer.{attn,mlp}_hyper_connection` (`GatedResidual`) |
| **Final Stack Output (`FinalResidual`)** | `SingleStream { norm_ptr }` (`GemmaRMSNorm`) | `HyperConnection { mixer }` (`mix_only` collapse `[T, hc_count, H] -> [T, H]` + pre-mixer `[T, hc_count * H]` export for MTP) | `Qwen4ExpModel.hyper_connection_mixer` (`use_combine=False`) + `_mtp_hidden_buffer` |
| **Per-Layer Embedding (`PleRankRuntime` / `PleLayerWeights`)** | `None` | `Some(PleLayerWeights)` on `ple_layer_ids` (`splitmix64` prime-modulo n-gram hash, `F8_E4M3` dequant, signed-sqrt sigmoid gate, dilated `Conv1d(k=4, d=3, span=9)`) | `Qwen4ExpPLELayer` + `Qwen4ExpNGramEmbedding` (`ple_gate`, `ple_conv`) |
| **Full-Attention Indexing (`FullAttentionLayerWeights::qsa_indexer`)** | `None` (dense causal/sliding-window GQA) | `Some(QsaIndexerWeights)` (`index_qk_proj`, `q/k_layernorm`, partial `mRoPE`, `4:1` compressed key cache, ReLU-sum MQA block selector, sparse paged GQA + sigmoid output gate) | `Qwen4ExpQSAAttention` + `QSAIndexer` (`qsa_prepare`, `qsa_select_paged_*`, `qsa_sparse_paged_attention`) |
| **MTP Input Fusion (`MtpInputFusion`)** | `ConcatLinear { fc }` (`concat(norm(embed), norm(hidden)) -> Linear(2H, H)`) | `DualStreamBroadcast { fc_embedding, fc_hidden }` (`fc_embed(norm_e(embed))` broadcast-added across `hc_count` streams of `fc_hidden(norm_h(multi_hidden))`) | `Qwen4ExpMultiTokenPredictor` (`residual_linear_shared`: `pre_fc_norm_embedding`, `pre_fc_norm_hidden(hc_count * H)`, `fc_embedding`, `fc_hidden`) |

### 6.2 Intrinsic `qwen4_exp` Mathematical Parity Checklist vs. Upstream `vLLM` (`190df0b90`)

1. **HyperConnection (`GatedResidual`, `op_hc.cu` vs. `nvidia/ops/hc.py`):**
   - Stream initialization: replicates `embed_tokens(input_ids)` (`[T, H]`) across `hc_count` streams (`[T, hc_count, H]`).
   - Per-stream `GemmaRMSNorm`: normalizes each stream `s in 0..hc_count` independently over `H` (`eps = 1e-6`, affine `1.0 + w`), supporting both per-branch `[hc_count * H]` and shared `[H]` weights.
   - Low-rank down activation & stream injection scale: divides by `hc_count` (`SiLU(down / hc_count)` over `[T, hc_lowrank]` and `2.0 * sigmoid(inj / hc_count)` over `[T, hc_count]`).
   - Stream mixing: `out_mixed[t, d] = sum_{s=0..hc_count-1} (sigmoid(up[t, s, d]) / hc_count) * normed[t, s, d]`.
   - Stream combine: `residual[t, s, d] += inject_w[t, s] * block_out[t, d]`.
2. **Per-Layer Embedding (`PLE`, `ple.rs` & `op_ple.cu` vs. `common/ops/ple.py` & `nvidia/ple_layer.py`):**
   - N-gram hash: 64-bit `splitmix64` multipliers seeded by `ple_seed + 10007 * ple_dense_layer_id` and deterministic 64-bit Miller-Rabin prime moduli `nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)`, with `eos_token_id` boundary clamping (`crossed |= candidate == eos_token_id`) and Euclidean remainder (`rem_euclid`).
   - Signed-sqrt sigmoid gate: `d = dot(norm_query, norm_key) / sqrt(H)`, `gate = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))`, `gated = gate * value`.
   - Stateful dilated short-conv: `norm_conv(gated)` -> causal depthwise `Conv1d(kernel_size=4, dilation=3, span=9)` + `SiLU`, tracked in `StateBundlePool` for prefix caching and speculative rollback.
3. **Query-Sparse Attention (`QSA`, `op_qsa.cu` vs. `nvidia/qsa.py` & `nvidia/qsa_indexer.py`):**
   - Main attention: `24 Q` heads (with packed sigmoid output gate) and `2 KV` heads (`head_dim = 256`, `rotary_dim = 64`, `mrope_section = [11, 11, 10]`, `rope_theta = 1e7`).
   - Indexer: `4 Q` heads and `1 KV` head (`head_dim = 128`, sharing `rotary_dim = 64` and `mrope_section = [11, 11, 10]` with main attention), `4:1` compressed key cache, ReLU-sum MQA scoring (`sum_h max(dot(q_h, k_c), 0) / sqrt(128)`), and `skip_topk = true` reuse on MTP draft steps `> 0`.
4. **Fine-Grained MoE (`op_moe.cu` vs. `Qwen4ExpSparseMoeBlock`):**
   - `512` routed experts (`top_k = 10`, `moe_intermediate_size = 640`, `norm_topk_prob = true`) + `1` shared expert (`shared_expert_intermediate_size = 640`) with sigmoid `shared_expert_gate`.
   - Power-of-two sub-warp reduction (`sparse_vec4_lpr(k_dim)` in `{4, 8, 16, 32}`) in `moe_sparse_vec4_kernel` ensures exact `TP=1` vs. `TP=2 / EP=2` parity for non-power-of-two intermediate dimensions such as `K = 640` (`k_u4_per_row = 20`).

### 6.3 Delineation: Intrinsic `Qwen4` Compatibility vs. `1–2× RTX 3090 Ti` (`sm_86`) Optimizations

| Subsystem | Upstream `vLLM` (`origin/main @ 190df0b90`, Datacenter `sm_90 / sm_100` Target) | `warp-34` (`1–2× RTX 3090 Ti`, `sm_86`, 24 GiB VRAM Target) | Classification |
| :--- | :--- | :--- | :--- |
| **Activation Spine & Kernels** | `BF16`-only (`raise NotImplementedError("Qwen4Exp QSA currently requires BF16")`); Triton + CuTe-DSL (`sm_90+`) kernels | End-to-end `FP16` (`__half`) spine with `FP32` reductions/accumulators and native `sm_86` CUDA C++ kernels (`op_hc.cu`, `op_ple.cu`, `op_qsa.cu`, `op_moe.cu`) | **3090 Ti Hardware Optimization** (`sm_86` tensor core & bandwidth roofline) |
| **51.3 GiB PLE Table Residency** | GPU-resident `VocabParallelEmbedding` (`~51.3 GiB` VRAM) or `OffloadManager` full-layer weight swaps | Zero-copy read-only `mmap` (`PleMmapTable`) + `MADV_WILLNEED` + `2.56 KB/tok` pinned async H2D staging (`0 B` persistent VRAM, `0 B` anonymous RAM) | **3090 Ti Hardware Optimization** (fits 51.3 GiB PLE in 0 MiB VRAM) |
| **62.8 GiB Routed MoE Residency** | 100% GPU-resident `FusedMoE` (`> 62 GiB` VRAM required across TP/EP ranks) | `FfnBlockWeights::TieredMoe`: per-layer VRAM `HotExpertCache` (`K_hot` slots) + pinned `HostExpertStore` with on-GPU LRU + async PCIe pull (`gpu-lru`), hybrid CPU/GPU execution (`cpu-hybrid`), and 2-group-ahead prefill DMA streaming | **3090 Ti Hardware Optimization** (fits 62.8 GiB MoE on 1× or 2× 24 GiB GPUs) |
| **QSA Main KV Cache Formats** | `BF16` or `FP8-e4m3` only (`raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")`) | Native `FP16`, `Q8_0` (`272 B/row`), and **TurboQuant 4-bit (`TQ4`, `132 B/row`, Seed-42 Rademacher + WHT-256 + Beta(256))** decoded directly inside `qsa_sparse_paged_attn_f16_kernel` | **3090 Ti Hardware Optimization** (`3.07×–3.88×` KV compression freeing VRAM for hot experts) |
| **Multi-GPU Collectives (`TP=2 / EP=2`)** | NCCL P2P / NVLink `AllReduce` and `allgather_reducescatter` sequence parallelism (`use_hc_sequence_parallel`) | Pinned-sysmem two-hop `TP=2` `AllReduce` (`op_collectives.cu`) engineered for PCIe Gen4 x8 (`topo PHB`, no NVLink, P2P disabled) | **3090 Ti Hardware Optimization** (sub-20 µs latency without NVLink/P2P) |
| **Hybrid State & Prefix Caching** | `MambaHybridModelState` + `PleShortConvAttentionBackend` | Unified `StateBundlePool` (`36× GDN` + `PLE conv_state` + `ple_history_tokens`) + `KvPool` (`QsaRawKeyCache` + `QsaCompressedKeyCache`) integrated with `MambaRadixCache` and speculative rejection rollback | **Architectural + Runtime Integration** |


