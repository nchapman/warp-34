# warp-34 — Engineering Roadmap & `vLLM` Architecture Map

> **Posture:** `warp-34` is a clean-room Rust + CUDA inference engine for **Qwen 3.5 / 3.6 / 3.8 hybrid models** on **1–2× RTX 3090 Ti (`sm_86`)**.
>
> **Source-of-Truth Rubric:** Every subsystem has **one explicit reference donor** chosen by where the mechanism is actually proven on our hardware:
> 1. **`vLLM 0.31.0` (`8352b2427`)** — Primary donor for the **`FP16` model spine**, **Marlin GEMMs** (`W4A16` / `W8A16` / `W4A8` / `moe_wna16`), **layer fusions** (`GemmaRMSNorm`, `fused_qk_rmsnorm_rope_gate`, GDN `chunk` + `fused_recurrent`), **native `MTP`**, **9-step GPU sampler**, and **Qwen3 reasoning/tool/vision parsers**.
> 2. **`SGLang v0.5.19` (`0bcd822377` / `lmsysorg/sglang:v0.5.19`)** — Primary donor for **Hybrid Prefix Caching (`MambaRadixCache` + `HybridReqToTokenPool` `extra_buffer` strategy)** and **`DFlash2` (`DFlashWorkerV2` `M=8` speculative verify + GDN state management)**, where `vLLM 0.31.0` is either inert on `TP=2` (`DFlash2` `tau=1.000` bug) or lacks radix prefix reuse under hybrid speculation.
> 3. **`warp3090` (in-house proven `sm_86` IP)** — Primary donor for **TurboQuant 4-bit KV (`TQ4`)** (`block_tq4_0` 4.125 bpw, Rademacher+WHT-256 rotation, Beta(256) Lloyd-Max codebook, `FirstLast2` `FP16` boundary layers, fused `fattn_mma_tq` + GQA-8 `fattn_vec_tq`) and **Pinned-Sysmem PCIe Gen4 x8 `TP=2` AllReduce** (`~18 µs` graph-capturable two-hop exchange).
>
> **Core Rule:** Nothing lands without **both** a correctness gate and a performance measurement against `vLLM` / `SGLang` on the exact same checkpoint, prompt, and batch shape. Phase 0 builds the oracle and profiling harness *before* any model kernel is written.

---

## 1. Architectural Non-Negotiables (Lessons from `warp3090`)

1. **`vLLM` is the single source of truth for both correctness and performance.**
   - No `llama.cpp` / `ggml` kernels, no `f32` intermediate activation spine, no GGUF loader, no `llama-bench` side-quests.
   - Any departure from `vLLM` (specifically **Paged `TQ4` 4-bit KV**, **`W4A8` INT8-activation Marlin**, and **Pinned-Sysmem PCIe `TP=2` AllReduce**) must plug cleanly into `vLLM V1` interfaces and prove an end-to-end wall-time or capacity win against `vLLM`.
2. **End-to-end `FP16` (`__half`) activation spine.**
   - Every hidden state, residual stream, norm input/output, RoPE tensor, attention Q/K/V/O, GDN input/output, and Marlin GEMM input/output is **`FP16`**.
   - Internal FP32 accumulators stay strictly inside registers/shared memory (e.g., RMSNorm variance, softmax, GDN recurrent `ssm_state`), eliminating the ~891 `f32 ↔ f16` cast kernels per round that bottlenecked `warp3090`.
3. **One unified V1 token-budget forward pass.**
   - Following `vLLM V1` (`vllm/v1/worker/gpu_model_runner.py`), every step flattens scheduled tokens into `[total_tokens, hidden_size]` accompanied by `query_start_loc` (`cu_seqlens`) and `positions: [3, total_tokens]`.
   - Prefill chunks, `M=1` decode, batched decode, `MTP` `1+k` verify, and `DFlash2` `M=8` verify all execute through the **same** model spine.
4. **Paged `BlockPool` + `MambaStatePool` from Day 1 (No Fixed Seats).**
   - Full-attention KV uses paged blocks (`block_size = 16` or `32`) from a single global GPU pool; GDN layers use a slot-indexed state pool (`conv_state` in `FP16`, `ssm_state` in `FP32`).
5. **Single weight residency per matrix.**
   - Never stage duplicate quantized representations of the same weight matrix in VRAM.

---

## 2. Cross-Checked `vLLM 0.31.0` Reference Map (`file:line`)

Verified against the local `vLLM 0.31.0` runtime (`~/.local/share/mise/installs/python/3.12.12/lib/python3.12/site-packages/vllm/`) and `vendor/vllm/`:

| Subsystem | `vLLM 0.31.0` Source (`file:line`) | Key Implementation Details Uncovered by Audit |
|---|---|---|
| **Dense Hybrid Model (`0.8B` / `4B` / `9B` / `27B`)** | `model_executor/models/qwen3_5.py:475` (`Qwen3_5ForConditionalGeneration`) & `qwen3_next.py:307-455` | Both `Qwen3.5-*` and `Qwen3.8-27B` checkpoints use `model_type="qwen3_5"` / `qwen3_5_text`. Interleaves 3× `"linear_attention"` (`QwenGatedDeltaNetAttention`) + 1× `"full_attention"` (`Qwen3NextAttention`). Norms use **`GemmaRMSNorm` (`1.0 + weight` unit-offset, `qwen3_5.py:42`)**. |
| **MoE Hybrid Model (`Qwen3.6-35B-A3B`)** | `model_executor/models/qwen3_5.py:704` (`Qwen3_5MoeForConditionalGeneration`) & `qwen3_next.py:143-259` | Uses `Qwen3NextSparseMoeBlock`: routed `top_k` experts + `shared_expert` + `shared_expert_gate` (`ReplicatedLinear`), with `maybe_fuse_shared_experts` (`qwen3_5.py:262-288`) and `moe_wna16` (`layers/quantization/moe_wna16.py:103`) Marlin kernel. |
| **Full Attention + Fused QK-Norm/RoPE/Gate** | `model_executor/models/qwen3_next.py:307-420` & `layers/fused_qk_norm_rope.py` | `q_proj` outputs packed `q \|\| gate` (`2 * num_heads * head_dim`, `head_dim=256`). `vLLM` fuses Q/K split + per-head `q_norm`/`k_norm` + partial interleaved 3D `mRoPE` (`partial_rotary_factor=0.25`, `mrope_section=[11,11,10]`, `theta=1e7`) into **`fused_qk_rmsnorm_rope_gate`**, followed by FlashAttention and `out *= sigmoid(gate)`. |
| **Gated DeltaNet (GDN) Linear Attention** | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:242-354, 483-495, 1332-1670` & `mamba_utils.py:122-143` | State dtypes: **`conv_state` = `FP16`**, **`ssm_state` = `FP32`** (`tp_replicated=False`, sharded across TP ranks). Output norm is **`RMSNormGated(norm_before_gate=True, activation="silu")`**. Prefill runs `causal_conv1d_fn` + `chunk_gated_delta_rule`; decode/verify runs `causal_conv1d_update` + `fused_sigmoid_gating_delta_rule_update` (with `spec_state_indices_tensor` & `num_accepted_tokens`). |
| **Quantization & Marlin Repack** | `model_executor/layers/quantization/utils/marlin_utils.py:247,752`, `compressed_tensors_wNa16.py`, `auto_gptq.py:225`, `auto_awq.py:101-156`, `inc/inc.py:38` | `compressed-tensors` (`pack-quantized` W4A16/W8A16/W4A8, static/group actorder, ignore list for `lm_head`, `mtp.*`, `visual.*`), standard `gptq`, `awq`, and `auto-round` (`auto_round:auto_gptq` / `auto_awq`) all repack at load time into the unified Marlin tile layout (`gptq_marlin_repack` / `awq_marlin_repack`). |
| **Hybrid Radix Prefix Cache & Pool (`SGLang` Donor)** | `sglang/srt/mem_cache/mamba_radix_cache.py:73-140` (`MambaRadixCache`, `TreeNode`) & `memory_pool.py:1-130` (`HybridReqToTokenPool`, `MambaPool`) | **Why `SGLang` over `vLLM`:** `vLLM` cannot share prefixes across hybrid Mamba/GDN requests when draft models (`DFlash2`/spec) are attached. `SGLang`'s `TreeNode` maintains **dual lock refs & dual LRU lists**: `full_lock_ref` (path-locked full-attn KV pages) and `mamba_lock_ref` (node-local GDN state snapshot, `full_lock_ref >= mamba_lock_ref`), so intermediate GDN states can be evicted (`mamba_evicted`) without evicting KV pages. `HybridReqToTokenPool` uses `enable_mamba_extra_buffer` ping-pong track buffers (`req_index_to_mamba_ping_pong_track_buffer_mapping`) to donate chunk-aligned GDN snapshots into the radix tree, `spec_state_size` scratch slots for speculative verify, and `clone_with_new_mamba` to isolate draft-model (`MTP`) state. |
| **CUDA Graphs & Async Output** | `v1/worker/gpu_model_runner.py:280-345, 761-777, 2282-2360` & `v1/cudagraph_dispatcher.py:36-80` | Captures batch-size buckets (`cudagraph_capture_sizes`) keyed by `uniform_decode_query_len = 1 + num_speculative_tokens`; pads unused batch rows with `NULL_BLOCK_ID`. `AsyncGPUModelRunnerOutput` copies sampled token IDs on `async_output_copy_stream` to overlap host bookkeeping with GPU execution. |
| **Native `MTP` (`vLLM`) & `DFlash2` (`SGLang` Donor)** | `vllm/model_executor/models/qwen3_5_mtp.py:1-60`; `sglang/srt/speculative/dflash_worker_v2.py:458, 1485, 1530, 2087-2096` | **Why `SGLang` for `DFlash2`:** `vLLM 0.31.0`'s attached `DFlash2` is inert on `TP=2` (`tau=1.000`, rank-local drafter logits bug), whereas `SGLang v0.5.19` runs `0df81ea` 27B + `incoai` DFlash2 at **175.7 tok/s code / 158.5 tok/s prose (`24.9 ms/round`)** via a single batched `M=8` `ForwardBatch` verify, `_append_target_hidden_fused`, and `_update_target_mamba_state_after_verify`. `Qwen3_5MTP` uses `vLLM`'s 1-layer TP-sharded predictor (`eh_proj` + gated attn + `ParallelLMHead` + GPTQ `int4` MLP sidecar). |
| **TurboQuant 4-Bit KV (`TQ4`, `warp3090` Donor)** | `warp3090`: `csrc/ops/op_tq4kv.cu`, `csrc/tq/tq4_codebook.cuh`, `csrc/ggml/fattn-vec-tq.cuh`, `csrc/ops/op_fattn_mma.cu`, `src/forward/kvtier.rs` | **Why `warp3090` over `vLLM`:** `vLLM` lacks an `sm_86`-fast 4-bit KV path. Our proven recipe: (1) **`block_tq4_0` (`66 B` / 128 elems = `4.125 bpw`)** with energy-preserving per-row `FP16` norm `n = ‖x‖/‖x̂_unit‖`; (2) **Seed-42 Rademacher + WHT-256 rotation** (`R = W · diag(s)` on K/V write and Q pre-rotate; `diag(s) · W` on O un-rotate); (3) **16-centroid Beta(256) Lloyd-Max codebook** in `__constant__` memory; (4) **`FirstLast2` `FP16` boundary layers** (slots `0`, `N-2`, `N-1` kept in `FP16` to prevent last-layer V Beta-fit outliers); (5) **Fused `fattn_mma_tq`** (`t <= 8` verify/chunk) + **GQA-8 `fattn_vec_tq`** (`t == 1` decode, `Q8_0`/`TQ4` K/V with L1 broadcast across 8 Q heads). |
| **`TP=2` Collectives (`warp3090` Donor)** | `warp3090`: `csrc/ops/op_all_reduce.cu`; `vllm/csrc/custom_all_reduce.cuh:10-282` | `vLLM`'s `CustomAllReduce` requires P2P (`_can_p2p`); on non-P2P PCIe (`topo PHB` on 2× 3090 Ti), `vLLM` disables `CustomAllReduce` and falls back to NCCL SHM. Our graph-capturable pinned-sysmem two-hop `FP16` all-reduce (`~18 µs`) replaces NCCL SHM while keeping `vLLM`'s exact TP sharding layout. |
| **GPU Sampler & Structured Output** | `v1/sample/sampler.py:22-62, 229-425`, `v1/sample/ops/topk_topp_sampler.py:202`, `v1/structured_output/backend_xgrammar.py:36,72-75` | 9-step GPU pipeline: `raw logits → fp32 → whitelist → bad_words → min_tokens/bias → penalties → temperature → min_p → top_k/top_p (ignoring -inf grammar masks) → Gumbel/greedy sample → logprobs`. Grammar state machines maintain `max_rollback_tokens = num_speculative_tokens` for speculative rejection rollback. |
| **Reasoning, Tool Calls, & Vision** | `parser/qwen3.py:1-240`, `tool_parsers/hermes_tool_parser.py:35-92`, `model_executor/models/qwen3_vl.py:376-647, 1979-2365` | `Qwen3Parser` handles `<think>...</think>` (`enable_thinking`), implicit `</think>` on `<tool_call>`, Qwen3-XML `<function=><parameter=>`, and Hermes JSON `<tool_call>`. Vision reuses `Qwen3_VisionTransformer` (`patch=16`, `temporal_patch=2`, `spatial_merge=2`, 3D `mRoPE`). |

---

## 3. Continuous Oracle & Telemetry Contract (Applies to Every Phase)

Every milestone must pass three automated checks before merging:

1. **Layer & Op Tensor Parity (`w34-oracle dump-diff`)**
   - Offline Python driver runs `vLLM 0.31.0` with forward hooks dumping per-layer inputs/outputs (`norm`, `qkv`, `qk_norm_rope`, `attn`/`gdn`, `mlp`, `logits`).
   - Rust test harness feeds the exact input tensors into our kernel/layer and asserts tight `FP16` tolerances (`max_abs_err`, `rel_l2`, cosine similarity) and 100% argmax agreement outside thin top-2 gaps (`< 0.10` logprob).
2. **End-to-End Greedy-Diff Oracle (`w34-oracle greedy-diff`)**
   - Runs the 12-prompt canonical suite across `4B`, `9B`, and `27B` against `vLLM` (`temp=0`, `ignore_eos=True`). Any divergence must fall within a verified near-tie (`top-2 logprob gap < 0.10`, target token in `vLLM` top-5).
3. **Side-by-Side `nsys` Kernel Scoreboard (`w34-oracle profile-diff`)**
   - Captures `nsys` SQLite traces of **both `vLLM 0.31.0` and `warp-34`** on identical workloads and prints a side-by-side table of **per-round GPU kernel time (`ms/round`) and launch counts** by category:
     - `marlin_gemm` (trunk + heads)
     - `full_attn` (FlashAttention prefill + paged decode)
     - `gdn_ssm` (`conv1d` + `chunk_gated_delta_rule` / `fused_recurrent`)
     - `norms_rope_act` (`GemmaRMSNorm`, `fused_qk_rmsnorm_rope_gate`, `silu_and_mul`, `RMSNormGated`)
     - `tp_comm` (`all_reduce`)
     - `spec_decode` (drafter forward + `rejection_sampler`)
     - `host_gap` (inter-kernel / round-boundary GPU idle time)
   - **Rule:** No kernel or subsystem graduates if its category time in `profile-diff` is slower than `vLLM` without an explicit, documented hardware reason.

---

## 4. Phased Implementation Plan

### Phase 0: `vLLM` Oracle, `nsys` Diff Harness, & Device Substrate
**Goal:** Establish the measurement harness and core GPU memory/stream primitives before writing model code.
- **0.1 `vLLM` Reference Runner & Tensor Dumper (`tools/oracle/`):**
  - `vllm_dump.py`: Dumps per-op and per-layer tensors from `vLLM 0.31.0` on `Qwen3.5-4B-W4A16-GPTQ` and `Qwen3.5-9B-W4A16-GPTQ`.
  - `vllm_greedy.py`: Generates canonical greedy token IDs + top-5 logprobs across short, long-context, code, and prose prompts.
  - `profile_diff.py`: Runs `nsys profile` on `vLLM` and `warp-34`, exports SQLite, and produces the side-by-side kernel category scoreboard (`ms/round`, achieved GB/s, launch count, idle gap).
- **0.2 CUDA Runtime & Arena Substrate (`w34-kernels`, `w34-host::cuda`):**
  - Per-device context (`CudaDevice`), stream pool (compute stream, async D2H copy stream, P2P/sysmem comm stream), event pool, cuBLAS handle, and preallocated `DeviceArena` (`FP16`-native tensor views).
- **0.3 Unified Safetensors & HF Hub Loader (`w34-host::loader`):**
  - Resolves local paths or HuggingFace Hub repo IDs (`~/.cache/huggingface/hub/`).
  - Parses `config.json` (`qwen3_5`, `qwen3_5_moe`, `qwen3`) and `quantization_config`:
    - `compressed-tensors` (`pack-quantized` W4A16 g128, static/group `actorder`, `ignore` regexes)
    - `gptq` (`bits=4, group_size=128, sym=true, desc_act`)
    - `awq` (`bits=4, group_size=128`)
    - `auto-round` (`auto_round:auto_gptq` / `auto_awq`)
    - Unquantized `BF16` / `FP16` (cast to `FP16` at load, except `FP32` GDN `A_log`/`dt_bias`).
- **Exit Gate:** `profile_diff.py` produces a clean baseline breakdown of `vLLM 0.31.0` on `4B`, `9B`, and `27B`; loader unit tests parse all 5 cached checkpoint families on disk.

---

### Phase 1: `vLLM`-Aligned `FP16` Kernel Suite & Per-Op Parity
**Goal:** Port and microbenchmark every layer kernel in isolation against `vLLM` tensor dumps (`dump-diff`) and `nsys` kernel timings (`profile-diff`).
- **1.1 Weight GEMMs (`csrc/marlin/`):**
  - Port `vLLM`'s `gptq_marlin_repack` / `awq_marlin_repack` and `Marlin` `W4A16` / `W8A16` (`kind-18` for `lm_head` / `eh_proj` / `draft_head`) kernels with `FP16` inputs and `FP16` outputs.
  - Include `W4A8` (`INT8` dynamic activation × `INT4` weight) Marlin path for compute-bound prefill and large verify batches on `sm_86` (320 TOPS INT8 vs 80 TFLOPS FP16-FP32acc).
- **1.2 Norms, RoPE, & Activations (All `FP16` I/O):**
  - `gemma_rms_norm_f16` & `fused_add_gemma_rms_norm_f16` (`1.0 + weight` unit-offset, FP32 internal variance, `FP16` in/out).
  - `rms_norm_gated_silu_f16` (`RMSNormGated(norm_before_gate=True, activation="silu")` for GDN output).
  - `fused_qk_rmsnorm_rope_gate_f16`: Single kernel splitting `qkv` (`q || gate`, `k`, `v`), applying per-head `q_norm`/`k_norm`, and applying partial interleaved 3D `mRoPE` (`[3, num_tokens]`).
  - `silu_and_mul_f16` (`gate_up` → `FP16` input to `down_proj` with zero cast).
  - `apply_attn_output_gate_f16` (`attn_out * sigmoid(gate)` in `FP16`).
- **1.3 Paged Full Attention (`csrc/attn/`):**
  - `FP16` Q/K/V/O FlashAttention prefill (`varlen` via `query_start_loc`) and paged decode/verify (`block_tables`, `seq_lens`, `head_dim=256`, GQA 4:1 / 8:1), writing `FP16` output directly.
- **1.4 Gated DeltaNet (`csrc/gdn/`):**
  - `FP16` input/output `causal_conv1d_fn` (prefill) and `causal_conv1d_update` (decode & multi-token speculative verify via `spec_state_indices`).
  - `chunk_gated_delta_rule` (prefill chunked scan) and `fused_sigmoid_gating_delta_rule_update` (decode & speculative verify with in-kernel `q/k` L2-norm, `FP16` I/O, and `FP32` recurrent `ssm_state`).
- **Exit Gate:** 100% of Phase 1 kernels pass `dump-diff` against `vLLM` layer dumps on `4B`/`9B` and match or beat `vLLM`'s per-kernel durations in `profile-diff`.

---

### Phase 2: Single-GPU & `TP=2` Model Spine (`w34 generate` & `w34 bench`)
**Goal:** Assemble the complete Qwen3.5 dense forward pass (`0.8B` / `4B` / `9B` / `27B`) for both `TP=1` and `TP=2` over paged `FP16` KV + pooled GDN state, with GPU-side sampling.
- **2.1 `BlockPool` & `MambaStatePool` (Basic Allocation):**
  - Global paged `FP16` KV block pool (`block_size = 16`) + GDN state slot pool (`conv_state` `FP16`, `ssm_state` `FP32`).
- **2.2 Unified `Qwen3_5Model` Forward Pass (`TP=1` & `TP=2`):**
  - Single layer loop supporting `tp_size ∈ {1, 2}`:
    - Column-parallel `qkv_proj`, `gate_up_proj`, GDN `in_proj`; row-parallel `o_proj`, `down_proj`, GDN `out_proj`; vocab-parallel `embed_tokens` and `lm_head`.
    - Pinned-sysmem two-hop `FP16` (`__half2`) `all_reduce` + `all_gather` for `TP=2` over PCIe Gen4 x8.
- **2.3 GPU-Side Batched Sampler (`SamplerV1`):**
  - Implements `vLLM`'s exact 9-step GPU pipeline (`vllm/v1/sample/sampler.py:22-62`): whitelist → bad_words → min_tokens/bias → repetition/frequency/presence penalties → temperature → min_p → top_k/top_p → Gumbel-max / argmax → optional top-K logprobs, returning only `int32` token IDs over `async_output_copy_stream`.
- **2.4 CUDA Graph Bucketing (`CUDAGraphRunner`):**
  - Captures decode/verify graphs across padded token buckets (`M ∈ {1, 2, 4, 8, 16, 24, 32, 48, 64}`) with `NULL_BLOCK_ID` padding.
- **2.5 CLI (`w34 generate` & `w34 bench`):**
  - Offline generation and `pp`/`tg`/batch benchmark runner.
- **Exit Gate:**
  - `w34-oracle greedy-diff`: 12/12 green on `4B`, `9B`, and `27B` (`TP=1` and `TP=2`).
  - `w34-oracle profile-diff`: `9B` and `27B` `TP=1` and `TP=2` plain decode and prefill match or beat `vLLM 0.31.0` wall time (`ms/round`).

---

### Phase 3: V1 Scheduler, `SGLang` `MambaRadixCache`, & OpenAI Server (`w34 serve`)
**Goal:** Full continuous batching, chunked prefill, `SGLang`-grade hybrid KV+GDN radix prefix caching (compatible with speculative decode!), and OpenAI-compatible HTTP serving.
- **3.1 Token-Budget Scheduler (`SchedulerV1`):**
  - Schedules up to `max_num_batched_tokens` per step, mixing active decode requests and chunked prefill slices without decode stalls; performs longest-prefix matching against the radix tree before scheduling prefill chunks.
- **3.2 `MambaRadixCache` + `HybridReqToTokenPool` (`SGLang` `extra_buffer` Architecture):**
  - **Dual-Lock / Dual-LRU Radix Tree (`mamba_radix_cache.py:73-140`):**
    - Each `TreeNode` holds paged KV block indices (`value`) and an optional chunk-aligned GDN state slot (`mamba_value`), with independent lock counters (`full_lock_ref >= mamba_lock_ref`) and separate LRU lists (`prev/next` vs `mamba_prev/mamba_next`).
    - Internal nodes can evict their bulky GDN state (`mamba_evicted`) while retaining their full-attention KV pages, or back up GDN checkpoints to host/INT8 (`mamba_checkpoint_pool`).
  - **Ping-Pong State Donation & Speculative Compatibility (`memory_pool.py:35-130`):**
    - `HybridReqToTokenPool` maintains `mamba_ping_pong_track_buffer` (`extra_buffer` strategy) to snapshot GDN state at radix chunk boundaries (`mamba_cache_chunk_size`) during prefill/decode and donate completed chunks into the radix tree without stalling the GPU.
    - Works seamlessly with both `MTP` (`clone_with_new_mamba` draft pool) and `DFlash2` (`spec_state_size` scratch verify slots + target hidden state cache).
- **3.3 OpenAI HTTP/SSE Server & CLI (`w34 serve`, `w34 chat`):**
  - Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`, `/health`, `/metrics` (Prometheus TTFT, ITL, KV pool utilization, radix KV & GDN hit rates).
  - **Qwen3 Unified Reasoning & Tool Parser (`vllm/parser/qwen3.py` parity):**
    - `chat_template_kwargs: {"enable_thinking": bool}`, streaming `delta.reasoning_content` (`<think>...</think>`) vs `delta.content`, implicit `</think>` on `<tool_call>`, and both **Qwen3-XML** (`<function=...><parameter=...>`) and **Hermes JSON** `<tool_call>` extraction.
  - **Structured Outputs (`response_format` / `json_schema` / Regex / GBNF):**
    - Token bitmask generation applied as `-inf` mask prior to top-k/top-p on GPU, with rollback buffer (`max_rollback_tokens`) for speculative decoding.
- **Exit Gate:** Concurrent 8-agent and 16-agent serve benchmarks on `4B`/`9B`/`27B` meet or beat `vLLM` / `SGLang` TTFT, ITL, and aggregate tok/s; multi-turn prefix-cache warm TTFT drops by >80% **both with and without speculative decoding enabled**.

---

### Phase 4: Speculative Decoding (`MTP` & `DFlash2` Proposers)
**Goal:** Land both native `Qwen3_5MTP` (`vLLM` donor) and `DFlash2DraftModel` (`SGLang` `DFlashWorkerV2` donor) behind a unified `SpecDecodeProposer` + `RejectionSampler` architecture that preserves `MambaRadixCache` prefix reuse.
- **4.1 Speculative Verify & Zero-Copy GDN/Radix Rollback:**
  - `MambaPool` allocates `spec_state_size` (`1 + k`) GDN state slots per active request; `fused_sigmoid_gating_delta_rule_update` steps through `spec_state_indices_tensor`.
  - GPU `RejectionSampler` (`accepted + recovered + bonus`) updates `seq_lens`, active GDN slot index (`_update_target_mamba_state_after_verify`), and radix ping-pong checkpoints on device without D2H stalls.
- **4.2 Native `MTPProposer` (`qwen3_5_mtp.py` + `clone_with_new_mamba`):**
  - 1-layer TP-sharded MTP block (`eh_proj` `fc` + gated full-attention + `ParallelLMHead` + optional GPTQ `int4` draft MLP sidecar).
  - Runs `k` device-resident draft steps inside captured CUDA graphs.
- **4.3 `DFlash2Proposer` (`SGLang` `dflash_worker_v2.py` parity):**
  - Fuses target hidden state capture (`_append_target_hidden_fused` at `target_layer_ids = [5, 19, 33, 47, 61]`), runs the TP-sharded 5-layer sliding-window cross-attention drafter in a single `M=8` pass, and feeds the 8 draft tokens into a single batched `M=9` (`1 + 8`) target verify pass.
- **Exit Gate:**
  - `9B` and `27B` `TP=2` seated `MTP` (`k=2, 3`) and `DFlash2` (`M=8`) match or beat `vLLM 0.31.0` (on `MTP`) and `SGLang v0.5.19` (`175.7 tok/s` code / `158.5 tok/s` prose on 27B `DFlash2`) in `profile-diff`, **with `MambaRadixCache` prefix caching active**.

---

### Phase 5: Paged `TQ4` 4-Bit KV Cache (`warp3090` Donor), Vision, & `Qwen3.6-35B-A3B` MoE
**Goal:** Port our battle-tested `warp3090` `TQ4` KV stack into the paged `MambaRadixCache` block pool, then add multimodal vision and MoE.
- **5.1 Paged TurboQuant 4-Bit KV (`kv_cache_dtype = "tq4"` or `"k8v4"`):**
  - **Ported directly from `warp3090` (`op_tq4kv.cu`, `tq4_codebook.cuh`, `fattn-vec-tq.cuh`, `op_fattn_mma.cu`, `kvtier.rs`), upgraded to `FP16` I/O and Paged Block Tables:**
    1. **Block & Row Geometry:** `block_tq4_0 { __half d; uint8_t qs[64]; }` (`66 B` per 128 elements = `4.125 bpw`; `132 B` per `d=256` head row vs `512 B` in `FP16` = **3.88× raw compression**), with energy-preserving per-row norm `d = ‖x‖ / ‖x̂_unit‖`. Supports both symmetric `tq4` (`K=tq4, V=tq4`) and hybrid `k8v4` (`K=q8_0` `272 B` w/ `dp4a`, `V=tq4` `132 B`).
    2. **Hadamard Rotation (`R = W · diag(s)`):** Seed-42 Rademacher sign diagonal + register/smem WHT-256 (`op_tq4kv.cu:wht256`, avoiding the `sm_86` `ptxas` cross-stage RMW hazard) fused into KV cache write, Q pre-rotation (`q' = W · diag(s) · q`), and O un-rotation (`o = diag(s) · W · o'`).
    3. **16-Centroid Beta(256) Lloyd-Max Codebook:** Stored in `__constant__` memory (`kTq4D256Centroids[16]`, `kTq4D256Boundaries[15]`) for branchless register quantization and on-the-fly dequantization.
    4. **`FirstLast2` `FP16` Boundary Layers (`f16b`):** Full-attention layer slot `0` and the last 2 full-attention layer slots (`N-2`, `N-1`) remain in `FP16` (`3/16` layers on 27B → **3.4× effective KV compression**), eliminating the post-WHT V-distribution tail outliers that otherwise degrade long-context quality.
    5. **Fused Paged Kernels on `sm_86`:**
       - **Decode (`t == 1`):** `flash_attn_ext_vec_tq` with GQA-8 head packing (one threadblock serves all 8 Q heads sharing a KV head so each `TQ4`/`Q8_0` KV page is fetched once into L1) and parallel KV-split reduction.
       - **Verify / Small-Chunk (`t <= 8`):** `fattn_mma_tq` dequantizing `block_tq4_0` pages directly into swizzled `half2` shared-memory MMA tiles (`nstages=0` override to prevent misaligned `cp.async` on 132B rows).
- **5.2 Multimodal Vision (`Qwen3_VisionTransformer`):**
  - Image preprocessing (smart resize + bicubic + normalize), 27-layer ViT (`patch=16`, `temporal_patch=2`, `spatial_merge=2`, axial 2D RoPE), deepstack/MLP merger (`1152 → 4096/5120`), visual embedding splice, and 3D `mRoPE` media position IDs.
- **5.3 `Qwen3.6-35B-A3B` Hybrid MoE (`Qwen3_5MoeForConditionalGeneration`):**
  - Top-k router softmax/sigmoid + `moe_wna16` fused Marlin MoE kernel + `shared_expert` + `shared_expert_gate` (`qwen3_next.py:143-259`).
- **Exit Gate:**
  - Single-card 27B (`TQ4` + `FirstLast2`) passes 32K/64K/100K needle-in-a-haystack and `greedy-diff` gates within 24 GB VRAM, and composes cleanly with `MambaRadixCache` and `TP=2`.
  - Multimodal vision passes `Qwen3.5-VL` parity against `vLLM`.
  - `Qwen3.6-35B-A3B` MoE passes `greedy-diff` and `profile-diff` against `vLLM`.

---

## 5. Summary Milestone Checklist

| Phase | Milestone | Primary Correctness Gate | Primary Performance Gate |
|---|---|---|---|
| **P0** | Oracle, `nsys` `profile-diff`, CUDA arena, Multi-format Loader | Loader parses all 5 checkpoint formats; tensor dump hooks verified | `profile-diff` baseline captured for `vLLM 0.31.0` on `4B` / `9B` / `27B` |
| **P1** | `FP16` Kernel Suite (Marlin, Norms, Fused QK-RoPE-Gate, Attn, GDN) | `w34-oracle dump-diff` bit/ulp parity per op vs `vLLM` | Every kernel ≤ `vLLM` duration in `profile-diff` |
| **P2** | `TP=1` & `TP=2` Model Spine, Paged KV + GDN Pool, GPU Sampler, Graphs | `w34-oracle greedy-diff` 12/12 on `4B` / `9B` / `27B` (`TP=1`, `TP=2`) | `w34 bench` & `profile-diff` ≥ `vLLM` on `TP=1` & `TP=2` |
| **P3** | V1 Scheduler, Hybrid Prefix Cache, OpenAI Server, Reasoning/Tools/JSON | OpenAI API + tool/reasoning/schema test suite + APC invariance | 8/16-agent `w34 serve` ≥ `vLLM` TTFT, ITL, & steady tok/s |
| **P4** | Speculative Decode (`MTP` + `DFlash2` behind V1 `SpecDecodeProposer`) | Lossless greedy verify + acceptance rate ≥ `0.98×` `vLLM` | `9B` & `27B` `TP=2` `MTP` / `DFlash2` ≥ `vLLM` spec-decode tok/s |
| **P5** | Paged `TQ4` KV, `Qwen3.5-VL` Vision, & `Qwen3.6-35B-A3B` MoE | 64K needle + VL image parity + MoE `greedy-diff` | 1-GPU 27B 64K fits 24 GB; MoE tok/s ≥ `vLLM` |
