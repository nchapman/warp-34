# warp-34 — Architecture & Donor Reference Map

> **Posture:** `warp-34` is a clean-room Rust + CUDA inference engine for **Qwen 3.5 / 3.6 / 3.8 hybrid models** on **1–4× RTX 30-series (`sm_86`) and 40-series (`sm_89`) GPUs** (primary reference targets: RTX 3090 / 3090 Ti and RTX 4090).
>
> **Source-of-Truth Rubric:** Every subsystem has **one explicit reference donor** chosen by where the mechanism is proven:
> 1. **`vLLM 0.31.0` (`8352b2427`)** — Primary donor for the **`FP16` model spine**, **Marlin GEMMs** (`W4A16` / `W8A16` / `W4A8` / `moe_wna16`), **layer fusions** (`GemmaRMSNorm`, `fused_qk_rmsnorm_rope_gate`, GDN `chunk` + `fused_recurrent`), **native `MTP`**, **9-step GPU sampler**, and **Qwen3 reasoning/tool/vision parsers**.
> 2. **`SGLang v0.5.19` (`0bcd822377` / `lmsysorg/sglang:v0.5.19`)** — Primary donor for **Hybrid Prefix Caching (`MambaRadixCache` + `HybridReqToTokenPool` `extra_buffer` strategy)** and **`DFlash2` (`DFlashWorkerV2` `M=8` / `M=16` speculative verify + GDN state management)**, where `vLLM 0.31.0` is either inert on `TP=2` (`DFlash2` `tau=1.000` bug) or lacks radix prefix reuse under hybrid speculation.
> 3. **`warp3090` (in-house proven `sm_86` IP)** — Primary donor for **TurboQuant 4-bit KV (`TQ4`)** (`block_tq4_0` 4.125 bpw, Rademacher+WHT-256 rotation, Beta(256) Lloyd-Max codebook, rotated `Q8_0` boundary layers, fused `fattn_mma_tq` + GQA-8 `fattn_vec_tq`) and **Pinned-Sysmem / NVLink P2P `TP=2` & `TP=4` Collectives**.
>
> **Core Rule:** Nothing lands without **both** a correctness check and a performance measurement against `vLLM` / `SGLang` on the exact same checkpoint, prompt, and batch shape.

---

## 1. Architectural Non-Negotiables (Lessons from `warp3090`)

1. **Subsystem-Specific Reference Oracle & Donor Matrix (No Ad-Hoc Hybrids).**
   - No `llama.cpp` / `ggml` `f32` activation spine, no GGUF loader, no `llama-bench` side-quests.
   - Each subsystem is verified against its authoritative oracle (`vLLM 0.31.0` for model spine/Marlin/MTP/sampler/parsers; `SGLang v0.5.19` for `MambaRadixCache` and `DFlashWorkerV2`; `warp3090` for `TQ4` and sysmem/NVLink collectives) and must prove an end-to-end wall-time or capacity win.
2. **Explicit Precision Policy (`FP16` Target Spine + `BF16` Drafter Preservation + `FP32` State/Reductions).**
   - **Target model spine (`FP16` / `__half`):** Every target hidden state, residual stream, norm I/O, RoPE tensor, attention Q/K/V/O, GDN activation I/O, and Marlin GEMM I/O is `FP16` — eliminating the ~891 `f32 ↔ f16` adapter casts per round that bottlenecked `warp3090`.
   - **Drafter `BF16` preservation (`__nv_bfloat16`):** Checkpoints whose dynamic range exceeds `FP16` max (`65504`) — specifically the **4B `DFlash2` drafter** — retain native `BF16` weights and activations (`sm_86` and `sm_89` support native `BF16` MMA and elementwise ops). Never blindly downcast `BF16` drafter weights to `FP16`.
   - **`FP32` persistent state & reductions:** GDN recurrent `ssm_state` (`[H_v, 128, 128]` in global memory), GDN `A_log`/`dt_bias`, RMSNorm variance accumulators, attention softmax accumulators, and sampler logits use `FP32`.
3. **One unified V1 token-budget forward pass with static `TapPlan` & 4-Length Transaction.**
   - Every step flattens scheduled tokens into `[total_tokens, hidden_size]` accompanied by `query_start_loc`, `positions: [3, N_max]` (fixed plane stride), and `valid_token_mask` / `valid_req_mask`.
   - Prefill chunks, prefix-hit suffix prefill, `M=1` decode, batched decode, `MTP` `1+k` verify, and `DFlash2` `M=8` (`27B`) / `M=16` (`4B`) verify all execute through the **same** model spine and expose a static `TapPlan` for capturing post-residual target hidden states.
4. **Logical Page IDs Decoupled from Per-Layer Physical Layout (`KvLayerLayout`) + Independent Versioned `StateBundle` Pool.**
   - Logical page allocation (`page_id = pos / B`, `B = 16`) is shared across a cache namespace, while each full-attention layer resolves physical addresses through a typed `KvLayerLayout` descriptor (`k_base/v_base`, `k_format/v_format`, independent `k_transform/v_transform`, `k_page_stride_bytes`, `k_head_stride_bytes`, `k_token_stride_bytes`). This allows `FP16`, `TQ4` (`132 B/row`), `Q8_0` (`272 B/row`), boundary-anchored mixed policies, and `DFlash2` draft-context KV (`head_dim=128`) to plug into the exact same pool and radix tree with zero rework.
5. **Single weight residency per matrix.**
   - Never stage duplicate quantized representations of the same weight matrix in VRAM. Any `W4A8` path must unpack from the single resident `W4A16` Marlin tile layout in registers/shared memory rather than duplicating weights.

---

## 1.5 Foundational Seam Contracts

Five foundational seams govern how `vLLM`, `SGLang`, and `warp3090` components compose across the engine:

1. **KV Cache & Attention Descriptor Seam (`docs/contracts/cache-layout.md`):**
   - **Logical vs. Physical:** `token_slot = physical_page_id * B + u` (`B = 16`) is a logical allocation identifier, never multiplied by a single global "bytes per token".
   - **Per-Layer `KvLayerLayout`:** Stores `cache_namespace` (`TargetFullAttn`, `MtpFullAttn`, `DFlashDraftContext`), `local_kv_heads`, `head_dim ∈ {128, 256}`, `k_format / v_format ∈ {FP16, BF16, TQ4, Q8_0}`, independent `k_transform / v_transform ∈ {Identity, RademacherWht256}`, and explicit byte strides (`page_stride_bytes`, `head_stride_bytes`, `token_stride_bytes`). Note that `TQ4` rows (`132 B` for `d=256`) are not 16-byte aligned at row boundaries even when page bases (`2112 B`) are aligned; kernels use byte-offset + `block_tq4_0`-aware loads.
   - **`AttentionBatch` Descriptor:** Distinguishes `context_lens_before` from `seq_lens_after`, carries `query_start_loc`, `block_tables[R_max, P_max]`, `write_slot_mapping[N_max]`, `valid_token_mask`, and mask geometry (`Causal` vs non-causal query-relative left window `window=2048` for 27B `DFlash2`), so prefix-hit prefill and speculative verify share the same descriptor.
2. **GDN Native CUDA Donor & Independent Versioned `StateBundle` Seam (`docs/contracts/state-transaction.md`):**
   - **Executable CUDA C++ Donor:** Because `vLLM 0.31.0` dispatches GDN via Python/Triton (`flash_linear_attention`), `w34-kernels` ports native CUDA C++ GDN kernels from **`vllm.cpp` (`src/vt/cuda/cuda_gdn.cu` at `50d425bec`)** cross-checked against `vLLM` FLA golden dumps and `warp3090`'s recurrence math.
   - **Explicit Input / Output / Spec / Checkpoint Indexing:** Kernels never assume `state_out == state_in`. Every GDN launch takes `state_in_indices[req]`, `spec_state_indices[req, verify_step]`, `checkpoint_dst_indices[req, event]`, `checkpoint_token_positions[req, event]`, and `has_initial_state[req]`.
   - **Three Distinct Chunk Geometries:** KV page size `B = 16`; GDN prefill compute tile `BT = 64` (`4` KV pages); and `MambaRadixCache` checkpoint interval `C = k * 64` (satisfying `C >= max_spec_verify_rows`).
   - **Explicit VRAM Ledger:** One 27B GDN `StateBundle` (`48` layers × `48` value heads × `128 × 128` `FP32` + `conv_state` `FP16`) is `~144 MiB` unsharded (`~72 MiB/rank` on `TP=2`, `~36 MiB/rank` on `TP=4`). Active + `spec_state` + ping-pong + radix checkpoint slots are budgeted upfront by `CapacityPlanner`.
3. **Speculative Commit Transaction & Draft-Context Prefix Seam (`docs/contracts/execution-contract.md`):**
   - **Four Tracked Lengths per Request:**
     - `processed_len`: tokens whose target KV and GDN state have been computed and committed.
     - `emitted_len`: committed output tokens visible to the caller.
     - `reserved_len`: private writable KV/state capacity reserved before speculative verify.
     - `draft_context_len`: tokens materialized inside the proposer's own KV/context cache.
   - **Why `processed_len != emitted_len` after verify:** On canonical 27B `DFlash2` (`block_rows = 8`: `1` anchor + `7` proposals = `8` target verify rows; `16/15/16` on 4B), if `a ∈ [0..7]` draft tokens are accepted, the target forward has consumed `1 + a` tokens (`processed_len += 1 + a`), while `1 + a` tokens are emitted (`a` accepted drafts + `1` recovered/bonus token) and that recovered/bonus token becomes the **next step's unconsumed anchor**.
   - **Warm Prefix Restore Includes Proposer Context:** A `MambaRadixCache` hit restores **target KV pages + deepest valid GDN `StateBundle` checkpoint + proposer draft-context KV** (or triggers fast target tap projection into draft KV on suffix prefill) so speculation starts warm immediately on prefix hits.
4. **Composite CUDA Graph Key & Valid-Row Isolation:**
   - **Composite `GraphKey`:** Graphs are keyed by `(execution_mode, request_bucket, token_bucket, query_len_class, kv_layout_mode, tap_plan_id)` — never by `total_tokens` alone (since `8 reqs × 1 tok` decode and `1 req × 8 tok` verify have different GDN scan strides and causal masks).
   - **Valid-Row Masking:** Every graph-captured kernel respects `valid_token_mask` / `valid_req_mask` (or writes to a dedicated trash sink slot) so padded rows never mutate live GDN states, increment RNG states, overwrite `TapPlan` slots, or corrupt radix checkpoints.
5. **Cold-Start `TP=2` / `TP=4` Initialization State Machine & Typed Collectives (`docs/contracts/collectives.md`):**
   - **Preventing CUDA 13 Lazy-Load Deadlocks:** `w34-kernels` enforces a strict `prepare_execution_plan()` state machine before entering `READY`: create contexts → allocate arenas & mapped sysmem/P2P buffers → load modules & warm every reachable kernel specialization independently on each GPU → `cudaDeviceSynchronize` all GPUs → warm collectives jointly → capture CUDA graphs → enter `READY`.
   - **System-Scope Visibility & Typed Payloads:** Collectives use direct 1-hop NVLink P2P VRAM reads/writes when peer access is available (`CommBuffer::DeviceVram`) or pinned-sysmem two-hop exchange (`CommBuffer::MappedHost`) with system-scope release/acquire fences and epoch parity, exposing distinct typed entrypoints for `FP16` hidden all-reduce, `BF16` draft activations, `FP32` logprob/maxima reductions, and `INT32` token IDs/counters (never casting token IDs through `FP16`).

---

## 2. Cross-Checked `vLLM 0.31.0` Reference Map (`file:line`)

Verified against the local `vLLM 0.31.0` runtime (`~/.local/share/mise/installs/python/3.12.12/lib/python3.12/site-packages/vllm/`) and `vendor/vllm/`:

| Subsystem | `vLLM 0.31.0` Source (`file:line`) | Key Implementation Details Uncovered by Audit |
|---|---|---|
| **Dense Hybrid Model (`0.8B` / `4B` / `9B` / `27B`)** | `model_executor/models/qwen3_5.py:475` (`Qwen3_5ForConditionalGeneration`) & `qwen3_next.py:307-455` | Both `Qwen3.5-*` and `Qwen3.8-27B` checkpoints use `model_type="qwen3_5"` / `qwen3_5_text`. Interleaves 3× `"linear_attention"` (`QwenGatedDeltaNetAttention`) + 1× `"full_attention"` (`Qwen3NextAttention`). Norms use **`GemmaRMSNorm` (`1.0 + weight` unit-offset, `qwen3_5.py:42`)**. |
| **MoE Hybrid Models (`Qwen3.6-35B-A3B` & `Qwen3.8-Flash-Next`)** | `model_executor/models/qwen3_5.py:704` (`Qwen3_5MoeForConditionalGeneration`) & `qwen3_next.py:143-259`; [`QWEN_FLASH_NEXT.md`](QWEN_FLASH_NEXT.md) | Uses `Qwen3NextSparseMoeBlock`: routed `top_k` experts + `shared_expert` + `shared_expert_gate` (`ReplicatedLinear`), with `moe_wna16` (`layers/quantization/moe_wna16.py:103`) Marlin kernel, plus `HyperConnection` (`hc_count=4`), Per-Layer Embeddings (`PLE`), Quality-Scored Sparse Attention (`QSA`), and tiered CPU/GPU 512-expert offloading on `Flash-Next`. |
| **Full Attention + Fused QK-Norm/RoPE/Gate** | `model_executor/models/qwen3_next.py:307-420` & `layers/fused_qk_norm_rope.py` | `q_proj` outputs packed `q \|\| gate` (`2 * num_heads * head_dim`, `head_dim=256`). `vLLM` fuses Q/K split + per-head `q_norm`/`k_norm` + partial interleaved 3D `mRoPE` (`partial_rotary_factor=0.25`, `mrope_section=[11,11,10]`, `theta=1e7`) into **`fused_qk_rmsnorm_rope_gate`**, followed by FlashAttention and `out *= sigmoid(gate)`. |
| **Gated DeltaNet (GDN) Linear Attention** | `vllm`: `layers/mamba/gdn/qwen_gdn_linear_attn.py:242-354, 483-495, 1332-1670`; `vllm.cpp`: `src/vt/cuda/cuda_gdn.cu` (`50d425bec`) | State dtypes: **`conv_state` = `FP16`**, **`ssm_state` = `FP32` in global memory** (`tp_replicated=False`, sharded across TP ranks). Output norm is **`RMSNormGated(norm_before_gate=True, activation="silu")`**. Because `vLLM` dispatches Python/Triton FLA kernels at runtime, `w34-kernels` ports native CUDA C++ GDN kernels (`causal_conv1d_fn/update`, `chunk_gated_delta_rule`, `fused_sigmoid_gating_delta_rule_update`) from `vllm.cpp/src/vt/cuda/cuda_gdn.cu` cross-checked against `vLLM` FLA golden dumps, taking explicit `state_in_indices`, `spec_state_indices`, and `checkpoint_dst_indices`. |
| **Quantization & Marlin Repack** | `model_executor/layers/quantization/utils/marlin_utils.py:247,752`, `compressed_tensors_wNa16.py`, `auto_gptq.py:225`, `auto_awq.py:101-156`, `inc/inc.py:38` | `compressed-tensors` (`pack-quantized` W4A16/W8A16/W4A8, static/group actorder, ignore list for `lm_head`, `mtp.*`, `visual.*`), standard `gptq`, `awq`, and `auto-round` (`auto_round:auto_gptq` / `auto_awq`) all repack at load time into the unified Marlin tile layout (`gptq_marlin_repack` / `awq_marlin_repack`). |
| **Hybrid Radix Prefix Cache & Pool (`SGLang` Donor)** | `sglang/srt/mem_cache/mamba_radix_cache.py:73-140` (`MambaRadixCache`, `TreeNode`) & `memory_pool.py:1-130` (`HybridReqToTokenPool`, `MambaPool`) | **Why `SGLang` over `vLLM`:** `vLLM` cannot share prefixes across hybrid Mamba/GDN requests when draft models (`DFlash2`/spec) are attached. `SGLang`'s `TreeNode` maintains **dual lock refs & dual LRU lists**: `full_lock_ref` (path-locked full-attn KV pages) and `mamba_lock_ref` (node-local GDN state snapshot, `full_lock_ref >= mamba_lock_ref`), so intermediate GDN states can be evicted (`mamba_evicted`) without evicting KV pages. `HybridReqToTokenPool` uses `enable_mamba_extra_buffer` ping-pong track buffers (`req_index_to_mamba_ping_pong_track_buffer_mapping`) to donate chunk-aligned GDN snapshots into the radix tree, `spec_state_size` scratch slots for speculative verify, and `clone_with_new_mamba` to isolate draft-model (`MTP`) state. |
| **CUDA Graphs & Async Output** | `v1/worker/gpu_model_runner.py:280-345, 761-777, 2282-2360` & `v1/cudagraph_dispatcher.py:36-80, 132-151` | Captures graphs keyed by composite `GraphKey(execution_mode, req_bucket, tok_bucket, query_len_class, kv_layout_mode, tap_plan_id)` with `valid_token_mask` / `valid_req_mask` and `NULL_BLOCK_ID` padding. Double-buffered pipelined decode steps overlap host bookkeeping with GPU execution. |
| **Native `MTP` (`vLLM`) & `DFlash2` (`SGLang` Donor)** | `vllm/model_executor/models/qwen3_5_mtp.py:1-134`; `sglang/srt/speculative/dflash_worker_v2.py:458, 1485, 1530-1578, 2087-2096` | **Why `SGLang` for `DFlash2`:** `vLLM 0.31.0`'s attached `DFlash2` is inert on `TP=2` (`tau=1.000`), whereas `SGLang v0.5.19` runs 27B + `incoai` DFlash2 via a single batched **`M=8` (`1` anchor + `7` proposals = `8` verify rows; `16/15/16` on 4B)** `ForwardBatch` verify, `_append_target_hidden_fused`, and `_update_target_mamba_state_after_verify`. `Qwen3_5MTP` uses `vLLM`'s 1-layer TP-sharded predictor (`eh_proj` + gated full-attention + `ParallelLMHead` + dense or MoE FFN sidecar). |
| **TurboQuant 4-Bit KV (`TQ4`, `warp3090` Donor)** | `warp3090`: `csrc/ops/op_tq4kv.cu`, `csrc/tq/tq4_codebook.cuh`, `csrc/ggml/fattn-vec-tq.cuh`, `csrc/ops/op_fattn_mma.cu`, `src/forward/kvtier.rs` | **Why `warp3090` over `vLLM`:** `vLLM` lacks an `sm_86`/`sm_89`-fast 4-bit KV path. Our recipe: (1) **`block_tq4_0` (`66 B` / 128 elems = `4.125 bpw`, `132 B/row`, `3.88×` raw)** with independent per-128 least-squares `FP16` scales; (2) **Seed-42 Rademacher + WHT-256 rotation** (`R = W · diag(s)` on K/V write and Q pre-rotate; `diag(s) · W` on O un-rotate); (3) **16-centroid Beta(256) Lloyd-Max codebook** in `__constant__` memory; (4) **Boundary-anchored `tq4`** (`4/16` boundary layers in rotated `Q8_0` `272 B/row`, `12/16` in rotated `TQ4` `132 B/row` → **`3.07×` effective KV compression**, `21,376 B/tok` on 27B); (5) **Paged `fattn_mma_tq`** + **GQA-grouped / Split-KV attention**. |
| **`TP=2` & `TP=4` Collectives (`warp3090` Donor)** | `warp3090`: `csrc/ops/op_all_reduce.cu:239-268`; `vllm/csrc/custom_all_reduce.cuh:10-282` | Uses direct 1-hop NVLink P2P VRAM transfers (`CommBuffer::DeviceVram`) when peer access is supported, or graph-capturable pinned-sysmem two-hop exchange (`CommBuffer::MappedHost`) on non-P2P PCIe (`topo PHB` / `NODE`), with payload-proportional CTA sizing, system-scope release/acquire fences, and typed entrypoints (`FP16`, `BF16`, `FP32`, `INT32`). |
| **GPU Sampler & Structured Output** | `v1/sample/sampler.py:22-62, 229-425`, `v1/sample/ops/topk_topp_sampler.py:202`, `v1/structured_output/backend_xgrammar.py:36,72-75` | 9-step GPU pipeline: `raw logits → fp32 → whitelist → bad_words → min_tokens/bias → penalties → temperature → min_p → top_k/top_p (ignoring -inf grammar masks) → Gumbel/greedy sample → logprobs`. Grammar state machines maintain `max_rollback_tokens = num_speculative_tokens` for speculative rejection rollback. |
| **Reasoning, Tool Calls, & Vision** | `parser/qwen3.py:1-240`, `model_executor/models/qwen3_vl.py:376-647, 1979-2365` | `Qwen3Parser` handles `<think>...</think>` (`enable_thinking`), implicit `</think>` on `<tool_call>`, and Qwen3-XML `<function=><parameter=>` tool calls. Vision reuses `Qwen3_VisionTransformer` (`patch=16`, `temporal_patch=2`, `spatial_merge=2`, 3D `mRoPE`). |

---

## 3. Continuous Oracle & Telemetry Verification

Every subsystem is verified by three automated checks:

1. **Layer & Op Tensor Parity (`w34 dump-diff`)**
   - Offline Python drivers (`tools/oracle/vllm_dump.py`, `sglang_dump.py`) run `vLLM 0.31.0` / `SGLang v0.5.19` with forward hooks dumping per-layer inputs/outputs (`norm`, `qkv`, `qk_norm_rope`, `attn`/`gdn`, `mlp`, `logits`, and DFlash2 target hidden taps).
   - Rust test suites and `w34 dump-diff` feed the exact input tensors into our kernels/layers and assert tight `FP16`/`BF16` tolerances (`max_abs_err`, `rel_l2`, cosine similarity) and `TP=1 == TP=2` parity.
2. **End-to-End Greedy-Diff Oracle (`w34 greedy-diff`)**
   - Runs the 12-prompt canonical suite across `4B`, `9B`, and `27B` against `vLLM` (`temp=0`, `ignore_eos=True`), verifying lossless speculative decoding (`--verify-lossless`) and warm radix prefix restoration (`--verify-radix-warm`).
3. **Side-by-Side `nsys` Kernel Scoreboard (`w34 profile-diff`)**
   - Captures `nsys` SQLite traces of **both `vLLM 0.31.0` / `SGLang v0.5.19` and `warp-34`** on identical workloads and renders a side-by-side table of **per-round GPU kernel time (`ms/round`) and launch counts** by category:
     - `marlin_gemm` (trunk + heads)
     - `full_attn` (FlashAttention prefill + paged decode)
     - `gdn_ssm` (`conv1d` + `chunk_gated_delta_rule` / `fused_recurrent`)
     - `norms_rope_act` (`GemmaRMSNorm`, `fused_qk_rmsnorm_rope_gate`, `silu_and_mul`, `RMSNormGated`)
     - `tp_comm` (`all_reduce`)
     - `spec_decode` (drafter forward + `rejection_sampler`)
     - `host_gap` (inter-kernel / round-boundary GPU idle time)

---

## 4. Subsystem Architecture

### 4.1 Device Substrate, Loader, & Kernel Suite (`crates/w34-kernels/`, `crates/w34-host/src/{cuda,loader}.rs`)
- **CUDA Runtime & `CapacityPlanner`:** Per-device context (`CudaDevice`), stream pool (compute, async D2H copy, auxiliary GDN/H2D/PLE streams), event pool, cuBLAS/cuBLASLt handles, preallocated `DeviceArena`, and `CapacityPlanner` budgeting weights, KV pages, GDN `StateBundle`s, proposer draft-context KV, and graph workspaces.
- **Unified Safetensors & HF Hub Loader:** Resolves local directories or HuggingFace Hub repo IDs with native download-on-first-use, pinned revisions, and offline mode (`docs/MODELS.md`). Supports `compressed-tensors` (`W4A16`/`W8A16`), `gptq`, `awq`, `auto-round`, FP8 `PLE`, and unquantized `FP16`/`BF16`.
- **Kernel Suite (`crates/w34-kernels/csrc/ops/`):**
  - `op_marlin.cu`: `gptq_marlin_repack`, `awq_marlin_repack`, `w34_marlin_repack_int8`, `w34_marlin_quantize_f16_u4b8` (online GPU `MTP` `draft_lm_head` quantization), symmetric/asymmetric `W4A16` (`U4B8` / `U4` with zero-points), `W8A16` (`U8B128`), and small-N `w34_small_n_gemm_f16` / `w34_small_n_gemm_dual_f16`.
  - `op_norms_rope_act.cu`: `gemma_rms_norm`, `fused_add_gemma_rms_norm`, `rms_norm_gated_silu_f16`, `fused_qk_rmsnorm_rope_gate` (with FP16-rounded rotary cache parity), `silu_and_mul`, and `apply_attn_output_gate`.
  - `op_kv_cache.cu`, `op_paged_attn.cu`, & `op_paged_fa2.cu`: Descriptor-driven `kv_cache_write`, GQA-grouped `mma.sync` and Split-KV paged attention across `FP16`, `BF16`, `Q8_0`, and `TQ4` (`132 B` rows with Seed-42 Rademacher + WHT-256 rotation and Beta(256) codebook), plus FlashAttention-2 prefill.
  - `op_gdn.cu` & `op_gdn_wy.cu`: Native CUDA `causal_conv1d` and `chunk_gated_delta_rule` (`BT = 64`) / `fused_sigmoid_gating_delta_rule_update` (`__launch_bounds__(128, 8)`) with explicit `state_in_indices`, `spec_state_indices`, and `checkpoint_dst_indices`, plus multi-stream `in_proj_ba` overlap.
  - `op_moe.cu`, `op_moe_router.cu`, & `op_moe_cpu_bridge.cu`: Top-K softmax/sigmoid router, grouped `mma.sync` / `sparse_mma` `W4A16` & `FP16` MoE execution (`35B-A3B` 256 experts and `Flash-Next` 512 experts across `EP=1`, `EP=2`, `EP=4`), GPU LRU hot cache, and async mapped-memory GPU↔CPU mailbox for host AVX-512 / AVX2 / portable cold-expert offload.
  - `op_collectives.cu`: Pinned-sysmem and NVLink P2P `TP=2` / `TP=4` collectives (`all_reduce_f16`, `all_reduce_residual_gemma_rms_norm_f16`, `all_reduce_bf16`, `all_reduce_f32`, `broadcast_i32`, `all_gather_top1_f32_i32`, `gather_vocab_f16`) with payload-proportional CTA sizing.
  - `op_lora.cu`: Graph-capturable multi-LoRA shrink+expand (`w34_lora_shrink_expand_f16`) supporting mixed per-token `token_lora_ids` (`-1` base + up to 62 stacked PEFT adapters), block-diagonal fused projections, and host Hadamard pre-rotation ($H_B^T H_B = I$) for `W4A16-HGPTQ` base models.

### 4.2 Hybrid Backbone, Continuous Batching, & `MambaRadixCache` (`crates/w34-host/src/{cache,model,radix,scheduler}.rs`)
- **Unified `Qwen3_5Engine` (`TP=1`, `TP=2`, `TP=4`):** Executes dense (`0.8B` / `4B` / `9B` / `27B`) and MoE (`35B-A3B` / `Flash-Next`) models over `KvPool` + versioned `StateBundlePool` with static `TapPlan` capture and composite `CUDAGraphRunner` buckets (`B ∈ {1, 2, 4, 8, 16}`).
- **`SchedulerV1` & `MambaRadixCache`:** Token-budget continuous batching (`max_num_batched_tokens`) with chunked prefill aligned to radix checkpoint intervals (`C = k * 64`). Dual-lock (`full_lock_ref >= mamba_lock_ref`) and dual-LRU radix tree with 2-slot ping-pong track buffers (`checkpoint_dst_indices`) donating only committed GDN checkpoints.

### 4.3 Speculative Decoding (`MTP` & `DFlash2`: `crates/w34-host/src/{speculative,mtp,dflash,proposer}.rs`)
- **Host & GPU `RejectionSampler`:** Consecutive greedy and canonical stochastic residual verification over `[anchor, draft_1, ..., draft_k]`, committing `1 + a` target KV and GDN `StateBundle` steps (`processed_len += 1 + a`), emitting `a` accepted drafts + `1` bonus/recovered token, and atomically rolling back `StateTx`, `SamplerV1`, and `TokenBitmaskConstraint` on rejection.
- **Native `Qwen3_5MTP` & `DFlash2`:** Per-rank CUDA-Graphed `Qwen3_5MtpRank` (dense and MoE, plus batched `B = 2..4` multi-sequence verify) and native-`BF16` `DFlash2Rank` (`M=16` on `4B`, `M=8` on `27B` with `window=2048`), with warm `MambaRadixCache` prefix hits restoring both target and proposer context (`PrefixMemo`).
- **Ratified `sm_86` HBM/PCIe Proposer Rooflines:**
  - **`4B + DFlash2` (`1,228–1,388 tok/s` across `FP16`/`Q8_0`/`TQ4`, `3.69 ms/r` vs `SGLang` `3.40 ms/r`):** Bounded by unquantized `2.45 GB` `BF16` drafter + `FP16` `lm_head` HBM bandwidth (`2.43 ms` theoretical 100%-HBM floor at `1,008 GB/s`).
  - **`35B-A3B MoE TP=2 + MTP` (`445.7 tok/s` unprofiled / `340.2 tok/s` under `nsys` on PCIe `topo PHB`):** Bounded by replicated `FP16` `lm_head` HBM floor (`2.02 ms/pass`) + `86×` PCIe Gen4 x8 sysmem all-reduce rounds when P2P/NVLink is absent.

### 4.4 Multimodal Vision, Structured Outputs, & HTTP Serving (`crates/w34-host/src/{vision,constraints,chatml,server,engine_worker}.rs`)
- **Multimodal Vision (`Qwen3_VisionTransformer`):** PNG, JPEG, and PPM decoding, SSRF-hardened opt-in remote image fetching (`--allow-remote-images`), `smart_resize` + bicubic + normalize preprocessing, 24/27-layer ViT with biased cuBLASLt GEMMs and single-round FP16 position interpolation, 2×2 patch merger, visual embedding splice, content-hashed `MambaRadixCache` keys, and 3D `mRoPE`.
- **GPU-Resident `SamplerV1` & `<think>`-Aware Structured Outputs:** GPU-resident stochastic sampler (`penalties → logit_bias → temperature → min_p → top_k/top_p → Gumbel-max`), in-graph `FastGreedy`, and pure-Rust `xgrammar-rs` JSON Schema / Regex / EBNF / Lark / Structural Tag bitmasks (`TokenBitmaskConstraint`) with vLLM-style per-request reasoning-deferred activation and native Qwen3-XML structural-tag enforcement. Compilation is shared and cached; matcher state is request-local. See [`STRUCTURED_OUTPUTS.md`](STRUCTURED_OUTPUTS.md).
- **Concurrent Continuous-Batching Server (`w34 serve`):** Background `engine_worker` driving `SchedulerV1` with 2-slot pipelined decode overlap (`~20 µs` inter-graph gap) and vLLM-compatible multi-LoRA routing (`--lora-modules` / `--lora`), serving OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`), Anthropic (`/v1/messages`, `/v1/messages/count_tokens`), Decision APIs (`/v1/systemone`, `/v1/decisions`), Prometheus `/metrics`, `/api/metrics`, and the embedded React/Vite dashboard.

---

## 5. Core Invariant Test Suite (`crates/w34-host/tests/`)

Five foundational integration tests (compiled under `--features cuda` and executed via `mise run check` on 1–4× GPU hosts with cached checkpoints) enforce cross-subsystem invariants:

1. **Fragmented Paged Attention across Heterogeneous Row Strides (`tests/backbone_and_collectives.rs` — `test_fragmented_paged_attention_and_tq4_strides`):**
   - Identical logical sequence produces bit-identical attention outputs whether physical `page_id`s in `block_tables` are contiguous or randomly permuted, across both `512 B` (`FP16`) and `132 B` (`TQ4`) row layouts and `context_lens_before > 0` prefix-hit suffixes.
2. **GDN Continuation, Checkpoint, & Rejection Parity (`tests/backbone_and_collectives.rs` — `test_gdn_continuation_checkpoint_and_rejection_parity`):**
   - One-shot prefill vs chunked prefill (`BT = 64`), checkpoint restore from `checkpoint_dst_indices`, and multi-step verify with forced rejection at every step `a ∈ [0..k]` produce bit-identical `conv_state` (`FP16`), `ssm_state` (`FP32`), and layer outputs.
3. **CUDA Graph Replay, Bucketed Metadata Mutation, & Padded-Row Isolation (`tests/decode_graphs.rs` — `decode_buckets_replay_mutated_requests_match_eager` & `pipelined_fast_greedy_burst_matches_single_and_handles_staggered_finish_and_chunk_boundary`; `tests/backbone_and_collectives.rs` — `test_cuda_graph_spec_rollback_and_tp2_collectives`):**
   - Replaying captured `GraphKey` decode buckets (`B ∈ {1, 2, 4, 8, 16}`) across permuted `block_tables`, `mamba_state_indices`, `seq_lens`, and padded (`valid_token_mask = 0`) rows matches eager execution bit-for-bit, leaves inactive GDN slots untouched, and verifies `StateTx` speculative verify/rollback slot scheduling.
4. **Cold-Start Multi-GPU Initialization & Collective Stress (`tests/backbone_and_collectives.rs` — `test_cuda_graph_spec_rollback_and_tp2_collectives` & `test_tp2_p2p_and_tp4_collectives`):**
   - Fresh process startup runs `prepare_execution_plan()`, exercises every reachable kernel specialization, and verifies zero lazy-module-load stalls and exact typed collective outputs (`FP16`, `BF16`, `FP32`, `INT32`, including token IDs `> 65504`).
5. **Warm Prefix-Hit Speculation & Cross-Boundary Rejection (`tests/radix_cache_and_scheduler.rs` — `test_warm_prefix_hit_speculation_and_cross_boundary_rejection`):**
   - Request A populates `MambaRadixCache` (target KV + GDN checkpoint + draft-context namespace); Request B hits the cached prefix, runs speculative verify across a `C = 64` checkpoint boundary, forces a mid-block rejection, and matches a cold non-speculative reference token-for-token and bit-for-bit in `conv_state` and `ssm_state`.
