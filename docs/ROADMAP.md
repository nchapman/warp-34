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

1. **Subsystem-Specific Reference Oracle & Donor Matrix (No Ad-Hoc Hybrids).**
   - No `llama.cpp` / `ggml` `f32` activation spine, no GGUF loader, no `llama-bench` side-quests.
   - Each subsystem is verified against its authoritative oracle (`vLLM 0.31.0` for model spine/Marlin/MTP/sampler/parsers; `SGLang v0.5.19` for `MambaRadixCache` and `DFlashWorkerV2`; `warp3090` for `TQ4` and PCIe sysmem `TP=2` AllReduce) and must prove an end-to-end wall-time or capacity win.
2. **Explicit Precision Policy (`FP16` Target Spine + `BF16` Drafter Preservation + `FP32` State/Reductions).**
   - **Target model spine (`FP16` / `__half`):** Every target hidden state, residual stream, norm I/O, RoPE tensor, attention Q/K/V/O, GDN activation I/O, and Marlin GEMM I/O is `FP16` — eliminating the ~891 `f32 ↔ f16` adapter casts per round that bottlenecked `warp3090`.
   - **Drafter `BF16` preservation (`__nv_bfloat16`):** Checkpoints whose dynamic range exceeds `FP16` max (`65504`) — specifically the **4B `DFlash2` drafter** (`dflash2.md`) — retain native `BF16` weights and activations (`sm_86` supports native `BF16` MMA and elementwise ops). Never blindly downcast `BF16` drafter weights to `FP16`.
   - **`FP32` persistent state & reductions:** GDN recurrent `ssm_state` (`[H_v, 128, 128]` in global memory), GDN `A_log`/`dt_bias`, RMSNorm variance accumulators, attention softmax accumulators, and sampler logits use `FP32`.
3. **One unified V1 token-budget forward pass with static `TapPlan` & 4-Length Transaction.**
   - Every step flattens scheduled tokens into `[total_tokens, hidden_size]` accompanied by `query_start_loc`, `positions: [3, N_max]` (fixed plane stride), and `valid_token_mask` / `valid_req_mask`.
   - Prefill chunks, prefix-hit suffix prefill, `M=1` decode, batched decode, `MTP` `1+k` verify, and `DFlash2` `M=8` (`27B`) / `M=16` (`4B`) verify all execute through the **same** model spine and expose a static `TapPlan` for capturing post-residual target hidden states.
4. **Logical Page IDs Decoupled from Per-Layer Physical Layout (`KvLayerLayout`) + Independent Versioned `StateBundle` Pool.**
   - Logical page allocation (`page_id = pos / B`, `B = 16`) is shared across a cache namespace, while each full-attention layer resolves physical addresses through a typed `KvLayerLayout` descriptor (`k_base/v_base`, `k_format/v_format`, independent `k_transform/v_transform`, `k_page_stride_bytes`, `k_head_stride_bytes`, `k_token_stride_bytes`). This allows `FP16`, `TQ4` (`132 B/row`), `K8V4` (`272 B` K / `132 B` V), `FirstLast2` mixed boundary layers, and `DFlash2` draft-context KV (`head_dim=128`) to plug into the exact same pool and radix tree with zero rework.
5. **Single weight residency per matrix.**
   - Never stage duplicate quantized representations of the same weight matrix in VRAM. Any `W4A8` path must unpack from the single resident `W4A16` Marlin tile layout in registers/shared memory rather than duplicating weights.

---

## 1.5 Foundational Seam Contracts (Preventing Cross-Phase Rewrites)

Before any Phase 1 kernel ABI is frozen, five foundational seams govern how `vLLM`, `SGLang`, and `warp3090` components compose across Phases 0–5:

1. **KV Cache & Attention Descriptor Seam (`docs/contracts/cache-layout.md`):**
   - **Logical vs. Physical:** `token_slot = physical_page_id * B + u` (`B = 16`) is a logical allocation identifier, never multiplied by a single global "bytes per token".
   - **Per-Layer `KvLayerLayout`:** Stores `cache_namespace` (`TargetFullAttn`, `MtpFullAttn`, `DFlashDraftContext`), `local_kv_heads`, `head_dim ∈ {128, 256}`, `k_format / v_format ∈ {FP16, BF16, TQ4, Q8_0}`, independent `k_transform / v_transform ∈ {Identity, RademacherWht256}`, and explicit byte strides (`page_stride_bytes`, `head_stride_bytes`, `token_stride_bytes`). Note that `TQ4` rows (`132 B` for `d=256`) are not 16-byte aligned at row boundaries even when page bases (`2112 B`) are aligned; kernels use byte-offset + `block_tq4_0`-aware loads.
   - **`AttentionBatch` Descriptor:** Distinguishes `context_lens_before` from `seq_lens_after`, carries `query_start_loc`, `block_tables[R_max, P_max]`, `write_slot_mapping[N_max]`, `valid_token_mask`, and mask geometry (`Causal` vs non-causal query-relative left window `window=2048` for 27B `DFlash2`), so prefix-hit prefill and speculative verify work from Phase 1.
2. **GDN Native CUDA Donor & Independent Versioned `StateBundle` Seam (`docs/contracts/state-transaction.md`):**
   - **Executable CUDA C++ Donor:** Because `vLLM 0.31.0` dispatches GDN via Python/Triton (`flash_linear_attention`), `w34-kernels` ports native CUDA C++ GDN kernels from **`vllm.cpp` (`src/vt/cuda/cuda_gdn.cu` at `50d425bec`)** cross-checked against `vLLM` FLA golden dumps and `warp3090`'s recurrence math.
   - **Explicit Input / Output / Spec / Checkpoint Indexing:** Kernels never assume `state_out == state_in`. Every GDN launch takes `state_in_indices[req]`, `spec_state_indices[req, verify_step]`, `checkpoint_dst_indices[req, event]`, `checkpoint_token_positions[req, event]`, and `has_initial_state[req]`.
   - **Three Distinct Chunk Geometries:** KV page size `B = 16`; GDN prefill compute tile `BT = 64` (`4` KV pages); and `MambaRadixCache` checkpoint interval `C = k * 64` (satisfying `C >= max_spec_verify_rows`).
   - **Explicit VRAM Ledger in Phase 0:** One 27B GDN `StateBundle` (`48` layers × `48` value heads × `128 × 128` `FP32` + `conv_state` `FP16`) is `~144 MiB` unsharded (`~72 MiB/rank` on `TP=2`). Active + `spec_state` + ping-pong + radix checkpoint slots are budgeted upfront in Phase 0.
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
5. **Cold-Start `TP=2` Initialization State Machine & Typed Collectives (`docs/contracts/collectives.md`):**
   - **Preventing CUDA 13 Lazy-Load Deadlocks (`dflash2.md:947`):** `w34-kernels` enforces a strict `prepare_execution_plan()` state machine before entering `READY`: create contexts → allocate arenas & mapped sysmem buffers → load modules & warm every reachable kernel specialization independently on each GPU → `cudaDeviceSynchronize` both GPUs → warm collectives jointly → capture CUDA graphs → enter `READY`.
   - **System-Scope Visibility & Typed Payloads:** Pinned-sysmem collectives preserve system-scope release/acquire fences and epoch parity, with distinct typed entrypoints for `FP16` hidden all-reduce, `BF16` draft activations, `FP32` logprob/maxima reductions, and `INT32` token IDs/counters (never casting token IDs through `FP16`).

---

## 2. Cross-Checked `vLLM 0.31.0` Reference Map (`file:line`)

Verified against the local `vLLM 0.31.0` runtime (`~/.local/share/mise/installs/python/3.12.12/lib/python3.12/site-packages/vllm/`) and `vendor/vllm/`:

| Subsystem | `vLLM 0.31.0` Source (`file:line`) | Key Implementation Details Uncovered by Audit |
|---|---|---|
| **Dense Hybrid Model (`0.8B` / `4B` / `9B` / `27B`)** | `model_executor/models/qwen3_5.py:475` (`Qwen3_5ForConditionalGeneration`) & `qwen3_next.py:307-455` | Both `Qwen3.5-*` and `Qwen3.8-27B` checkpoints use `model_type="qwen3_5"` / `qwen3_5_text`. Interleaves 3× `"linear_attention"` (`QwenGatedDeltaNetAttention`) + 1× `"full_attention"` (`Qwen3NextAttention`). Norms use **`GemmaRMSNorm` (`1.0 + weight` unit-offset, `qwen3_5.py:42`)**. |
| **MoE Hybrid Model (`Qwen3.6-35B-A3B`)** | `model_executor/models/qwen3_5.py:704` (`Qwen3_5MoeForConditionalGeneration`) & `qwen3_next.py:143-259` | Uses `Qwen3NextSparseMoeBlock`: routed `top_k` experts + `shared_expert` + `shared_expert_gate` (`ReplicatedLinear`), with `maybe_fuse_shared_experts` (`qwen3_5.py:262-288`) and `moe_wna16` (`layers/quantization/moe_wna16.py:103`) Marlin kernel. |
| **Full Attention + Fused QK-Norm/RoPE/Gate** | `model_executor/models/qwen3_next.py:307-420` & `layers/fused_qk_norm_rope.py` | `q_proj` outputs packed `q \|\| gate` (`2 * num_heads * head_dim`, `head_dim=256`). `vLLM` fuses Q/K split + per-head `q_norm`/`k_norm` + partial interleaved 3D `mRoPE` (`partial_rotary_factor=0.25`, `mrope_section=[11,11,10]`, `theta=1e7`) into **`fused_qk_rmsnorm_rope_gate`**, followed by FlashAttention and `out *= sigmoid(gate)`. |
| **Gated DeltaNet (GDN) Linear Attention** | `vllm`: `layers/mamba/gdn/qwen_gdn_linear_attn.py:242-354, 483-495, 1332-1670`; `vllm.cpp`: `src/vt/cuda/cuda_gdn.cu` (`50d425bec`) | State dtypes: **`conv_state` = `FP16`**, **`ssm_state` = `FP32` in global memory** (`tp_replicated=False`, sharded across TP ranks). Output norm is **`RMSNormGated(norm_before_gate=True, activation="silu")`**. Because `vLLM` dispatches Python/Triton FLA kernels at runtime, `w34-kernels` ports native CUDA C++ GDN kernels (`causal_conv1d_fn/update`, `chunk_gated_delta_rule`, `fused_sigmoid_gating_delta_rule_update`) from `vllm.cpp/src/vt/cuda/cuda_gdn.cu` cross-checked against `vLLM` FLA golden dumps, taking explicit `state_in_indices`, `spec_state_indices`, and `checkpoint_dst_indices`. |
| **Quantization & Marlin Repack** | `model_executor/layers/quantization/utils/marlin_utils.py:247,752`, `compressed_tensors_wNa16.py`, `auto_gptq.py:225`, `auto_awq.py:101-156`, `inc/inc.py:38` | `compressed-tensors` (`pack-quantized` W4A16/W8A16/W4A8, static/group actorder, ignore list for `lm_head`, `mtp.*`, `visual.*`), standard `gptq`, `awq`, and `auto-round` (`auto_round:auto_gptq` / `auto_awq`) all repack at load time into the unified Marlin tile layout (`gptq_marlin_repack` / `awq_marlin_repack`). |
| **Hybrid Radix Prefix Cache & Pool (`SGLang` Donor)** | `sglang/srt/mem_cache/mamba_radix_cache.py:73-140` (`MambaRadixCache`, `TreeNode`) & `memory_pool.py:1-130` (`HybridReqToTokenPool`, `MambaPool`) | **Why `SGLang` over `vLLM`:** `vLLM` cannot share prefixes across hybrid Mamba/GDN requests when draft models (`DFlash2`/spec) are attached. `SGLang`'s `TreeNode` maintains **dual lock refs & dual LRU lists**: `full_lock_ref` (path-locked full-attn KV pages) and `mamba_lock_ref` (node-local GDN state snapshot, `full_lock_ref >= mamba_lock_ref`), so intermediate GDN states can be evicted (`mamba_evicted`) without evicting KV pages. `HybridReqToTokenPool` uses `enable_mamba_extra_buffer` ping-pong track buffers (`req_index_to_mamba_ping_pong_track_buffer_mapping`) to donate chunk-aligned GDN snapshots into the radix tree, `spec_state_size` scratch slots for speculative verify, and `clone_with_new_mamba` to isolate draft-model (`MTP`) state. |
| **CUDA Graphs & Async Output** | `v1/worker/gpu_model_runner.py:280-345, 761-777, 2282-2360` & `v1/cudagraph_dispatcher.py:36-80, 132-151` | Captures graphs keyed by composite `GraphKey(execution_mode, req_bucket, tok_bucket, query_len_class, kv_layout_mode, tap_plan_id)` with `valid_token_mask` / `valid_req_mask` and `NULL_BLOCK_ID` padding. `AsyncGPUModelRunnerOutput` copies sampled token IDs on `async_output_copy_stream` to overlap host bookkeeping with GPU execution. |
| **Native `MTP` (`vLLM`) & `DFlash2` (`SGLang` Donor)** | `vllm/model_executor/models/qwen3_5_mtp.py:1-134`; `sglang/srt/speculative/dflash_worker_v2.py:458, 1485, 1530-1578, 2087-2096` | **Why `SGLang` for `DFlash2`:** `vLLM 0.31.0`'s attached `DFlash2` is inert on `TP=2` (`tau=1.000`), whereas `SGLang v0.5.19` runs `0df81ea` 27B + `incoai` DFlash2 at **175.7 tok/s code / 158.5 tok/s prose (`24.9 ms/round`)** via a single batched **`M=8` (`1` anchor + `7` proposals = `8` verify rows; `16/15/16` on 4B)** `ForwardBatch` verify, `_append_target_hidden_fused`, and `_update_target_mamba_state_after_verify`. `Qwen3_5MTP` uses `vLLM`'s 1-layer TP-sharded predictor (`eh_proj` + gated full-attention + `ParallelLMHead` + GPTQ `int4` MLP sidecar). |
| **TurboQuant 4-Bit KV (`TQ4`, `warp3090` Donor)** | `warp3090`: `csrc/ops/op_tq4kv.cu`, `csrc/tq/tq4_codebook.cuh`, `csrc/ggml/fattn-vec-tq.cuh`, `csrc/ops/op_fattn_mma.cu`, `src/forward/kvtier.rs` | **Why `warp3090` over `vLLM`:** `vLLM` lacks an `sm_86`-fast 4-bit KV path. Our proven recipe: (1) **`block_tq4_0` (`66 B` / 128 elems = `4.125 bpw`, `132 B/row`, `3.88×` raw)** with energy-preserving per-row `FP16` norm `n = ‖x‖/‖x̂_unit‖`; (2) **Seed-42 Rademacher + WHT-256 rotation** (`R = W · diag(s)` on K/V write and Q pre-rotate; `diag(s) · W` on O un-rotate; independent `k_transform/v_transform` for `K8V4`); (3) **16-centroid Beta(256) Lloyd-Max codebook** in `__constant__` memory; (4) **`FirstLast2` `FP16` boundary layers** (`3/16` layers in `FP16`, `13/16` in `TQ4` → **`2.52×` effective KV compression**, `26,016 B/tok` on 27B); (5) **Paged `fattn_mma_tq`** + **GQA-8 `fattn_vec_tq`** + bounded-memory streaming prefill/wide-verify fallback. |
| **`TP=2` Collectives (`warp3090` Donor)** | `warp3090`: `csrc/ops/op_all_reduce.cu:239-268`; `vllm/csrc/custom_all_reduce.cuh:10-282` | `vLLM`'s `CustomAllReduce` requires P2P (`_can_p2p`); on non-P2P PCIe (`topo PHB` on 2× 3090 Ti), `vLLM` falls back to NCCL SHM. Our graph-capturable pinned-sysmem two-hop exchange (`~18 µs`) uses system-scope release/acquire fences and typed entrypoints (`FP16`, `BF16`, `FP32`, `INT32`). |
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

### Phase 0: `vLLM` / `SGLang` Oracle, `nsys` Diff Harness, Foundational Contracts, & Device Substrate
**Goal:** Establish the measurement harness, formal seam contracts, VRAM capacity planner, and core GPU memory/stream primitives before writing model code.
- **0.1 Foundational Seam Contracts (`docs/contracts/`):**
  - Write and lock `execution-contract.md` (lengths, token flattening, `valid_token_mask`, `GraphKey`, `TapPlan`), `cache-layout.md` (`KvLayerLayout`, `AttentionBatch`, logical page IDs, namespaces, COW rules), `state-transaction.md` (GDN `StateBundle`, `spec_state_indices`, ping-pong checkpointing, chunk intervals `B=16`, `BT=64`, `C=k*64`), `collectives.md` (`prepare_execution_plan()` cold-start state machine, system-scope fences, typed payloads), and `precision-policy.md` (`FP16` target, `BF16` drafter preservation, `FP32` persistent `ssm_state`/reductions).
- **0.2 `vLLM` & `SGLang` Reference Runners & Tensor Dumpers (`tools/oracle/`):**
  - `vllm_dump.py` & `sglang_dump.py`: Dump per-op/per-layer tensors, `DFlash2` target hidden taps (`[5, 19, 33, 47, 61]`), and partial-rejection GDN states from `vLLM 0.31.0` and `SGLang v0.5.19`.
  - `vllm_greedy.py`: Generates canonical greedy token IDs + top-5 logprobs across short, long-context, code, and prose prompts.
  - `profile_diff.py`: Runs `nsys profile` on `vLLM`, `SGLang`, and `warp-34`, exports SQLite, and produces the side-by-side kernel category scoreboard (`ms/round`, achieved GB/s, launch count, idle gap).
- **0.3 CUDA Runtime, `prepare_execution_plan()` State Machine, & VRAM Capacity Planner (`w34-kernels`, `w34-host::cuda`):**
  - Per-device context (`CudaDevice`), stream pool (compute, async D2H copy, sysmem comm), event pool, cuBLAS handle, and preallocated `DeviceArena`.
  - `CapacityPlanner` computing exact VRAM budgets for weights + KV pages + active/speculative/ping-pong/radix GDN `StateBundle`s (`72 MiB/rank` per 27B `TP=2` snapshot) + proposer draft-context KV + graph workspaces.
- **0.4 Unified Safetensors & HF Hub Loader (`w34-host::loader`):**
  - Resolves local paths or HuggingFace Hub repo IDs (`~/.cache/huggingface/hub/`).
  - Parses `config.json` (`qwen3_5`, `qwen3_5_moe`, `qwen3`, `dflash2`) and `quantization_config`:
    - `compressed-tensors` (`pack-quantized` W4A16 g128, static/group `actorder`, `ignore` regexes)
    - `gptq` (`bits=4, group_size=128, sym=true, desc_act`)
    - `awq` (`bits=4, group_size=128`)
    - `auto-round` (`auto_round:auto_gptq` / `auto_awq`)
    - Unquantized `FP16` / `BF16` (preserving `BF16` on drafters where required by `precision-policy.md`, and `FP32` on GDN `A_log`/`dt_bias`).
- **Exit Gate:** `profile_diff.py` produces clean baseline breakdowns of `vLLM 0.31.0` and `SGLang v0.5.19` on `4B`, `9B`, and `27B`; loader unit tests and `CapacityPlanner` validate all 5 cached checkpoint families on disk.

---

### Phase 1: Donor-Aligned Kernel Suite & Per-Op Parity Against Final Descriptors
**Goal:** Port and microbenchmark every layer kernel in isolation against `vLLM` / `SGLang` tensor dumps (`dump-diff`) and `nsys` kernel timings (`profile-diff`) using the final `KvLayerLayout`, `AttentionBatch`, and `StateBundle` descriptors.
- **1.1 Weight GEMMs (`csrc/marlin/` + cuBLAS `FP16`/`BF16`):**
  - Port `vLLM`'s `gptq_marlin_repack` / `awq_marlin_repack` and `Marlin` `W4A16` / `W8A16` (`kind-18` for `lm_head` / `eh_proj` / `draft_head`) kernels with `FP16` I/O.
  - Evaluate `W4A8` (`INT8` dynamic activation × `INT4` weight) only if it unpacks from the single resident `W4A16` Marlin tile layout with a measured wall-time win on `sm_86`.
  - Unquantized `FP16` and `BF16` GEMM wrappers for `DFlash2` / `MTP` sidecars.
- **1.2 Norms, RoPE, & Activations (`FP16` Target + `BF16` Drafter Variants):**
  - `gemma_rms_norm` & `fused_add_gemma_rms_norm` (`1.0 + weight` unit-offset, FP32 internal variance).
  - `rms_norm_gated_silu_f16` (`RMSNormGated(norm_before_gate=True, activation="silu")` for GDN output).
  - `fused_qk_rmsnorm_rope_gate`: Single kernel splitting `qkv` (`q || gate`, `k`, `v`), applying per-head `q_norm`/`k_norm`, and applying partial interleaved 3D `mRoPE` (`[3, N_max]` with fixed plane stride).
  - `silu_and_mul` and `apply_attn_output_gate` (`attn_out * sigmoid(gate)`).
- **1.3 Descriptor-Driven Paged Attention (`csrc/attn/`):**
  - Driven by `KvLayerLayout` + `AttentionBatch` supporting both `head_dim = 256` (target) and `head_dim = 128` (27B `DFlash2` drafter), GQA 4:1 / 8:1, `context_lens_before` vs `seq_lens_after` (prefix-hit prefill), and both causal and non-causal query-relative left-window (`window=2048`) masks.
  - Includes fragmented-page reader test harness exercising both `512 B` (`FP16`) and `132 B` (`TQ4` synthetic) row strides across randomized page IDs.
- **1.4 Gated DeltaNet (`csrc/gdn/`, ported from `vllm.cpp` `cuda_gdn.cu` & verified vs `vLLM` FLA):**
  - `FP16` I/O `causal_conv1d_fn` (prefill) and `causal_conv1d_update` (decode & verify).
  - `chunk_gated_delta_rule` (`BT = 64` chunked scan) and `fused_sigmoid_gating_delta_rule_update` (decode & verify with in-kernel `q/k` L2-norm, `FP16` I/O, and global-memory `FP32` `ssm_state`).
  - Exposes `state_in_indices`, `spec_state_indices`, and `checkpoint_dst_indices` from Day 1.
- **1.5 Pinned-Sysmem `TP=2` Collectives (`csrc/comm/`):**
  - Port `warp3090`'s system-scope two-hop PCIe Gen4 x8 exchange with typed entrypoints (`FP16` hidden all-reduce, `BF16` draft activations, `FP32` top-1/top-K candidate reductions, `INT32` token IDs/counters) and cold-start `prepare_execution_plan()` warmup verification.
- **Exit Gate:** 100% of Phase 1 kernels pass `dump-diff` against `vLLM`/`SGLang` dumps on `4B`/`9B`, pass **Decisive Gate Tests #1 (Fragmented Paged Attn), #2 (GDN Continuation & Rejection Parity), and #4 (Cold-Start `TP=2` Init)**, and match or beat `vLLM`'s per-kernel durations in `profile-diff`.

---

### Phase 2: Single-GPU & `TP=2` Model Spine, Complete Execution Skeleton, & Composite CUDA Graphs
**Goal:** Assemble the complete Qwen3.5 dense forward pass (`0.8B` / `4B` / `9B` / `27B`) for `TP=1` and `TP=2` over heterogeneous-ready `KvPool` + versioned `StateBundlePool`, prove multi-request/taps/synthetic-rejection eagerly, and then capture composite CUDA Graph families.
- **2.1 Namespace-Aware `KvPool` & Versioned `StateBundlePool`:**
  - Logical page allocator (`block_size = 16`) backed by per-layer slabs described by `KvLayerLayout` + versioned GDN `StateBundlePool` (active, speculative scratch, and checkpoint slots).
- **2.2 Unified `Qwen3_5Model` Forward Pass (`TP=1` & `TP=2`) with Static `TapPlan`:**
  - Single layer loop supporting `tp_size ∈ {1, 2}`, `valid_token_mask`, and static `TapPlan` (capturing post-residual hidden states at configured layers, e.g., normalized `[5, 19, 33, 47, 61]` on 27B, across prefill, suffix-prefill, decode, and verify).
  - Eagerly verified on unequal-length multi-request batches, non-contiguous page tables, state-slot remapping, and **synthetic speculation (`1 + k`) with forced rejection at every step** (`processed_len` vs `emitted_len` transaction).
- **2.3 GPU-Side Batched Sampler (`SamplerV1`):**
  - Implements `vLLM`'s 9-step GPU pipeline (`vllm/v1/sample/sampler.py:22-62`) with transactional history and request/position-seeded RNG, plus `TP=2` global candidate selection (`(max_val, int32_token_id)` / local top-K merge).
- **2.4 Composite CUDA Graph Runner (`CUDAGraphRunner`):**
  - Captures graph families keyed by `GraphKey(execution_mode, req_bucket, tok_bucket, query_len_class, kv_layout_mode, tap_plan_id)` over fixed-address arena buffers (`positions[3, N_max]` with fixed plane stride, `valid_token_mask` / `valid_req_mask`).
- **2.5 CLI (`w34 generate` & `w34 bench`):**
  - Offline generation and `pp`/`tg`/batch benchmark runner.
- **Exit Gate:**
  - Passes **Decisive Gate Test #3 (Graph Replay Mutation without stale writes)** and synthetic speculative rollback tests.
  - `w34-oracle greedy-diff`: 12/12 green on `4B`, `9B`, and `27B` (`TP=1` and `TP=2`).
  - `w34-oracle profile-diff`: `9B` and `27B` `TP=1` and `TP=2` plain decode and prefill match or beat `vLLM 0.31.0` wall time (`ms/round`).

---

### Phase 3: V1 Scheduler, `SGLang` `MambaRadixCache`, & OpenAI Server (`w34 serve`)
**Goal:** Full continuous batching, chunked prefill aligned to radix checkpoint intervals, `SGLang`-grade hybrid KV+GDN+DraftContext radix prefix caching, and OpenAI-compatible HTTP serving.
- **3.1 Token-Budget Scheduler (`SchedulerV1`):**
  - Schedules up to `max_num_batched_tokens` per step, mixing active decode/verify requests and chunked prefill slices aligned to the radix checkpoint grid (`C = k * 64`); performs longest-prefix matching before scheduling prefill chunks.
- **3.2 `MambaRadixCache` + `HybridReqToTokenPool` (`SGLang` `extra_buffer` Architecture via Logical-Page Adapter):**
  - **Dual-Lock / Dual-LRU Radix Tree (`mamba_radix_cache.py:73-140`):**
    - Adapts `SGLang`'s token-slot slices cleanly to logical `page_id` arrays (`B = 16`) + chunk-aligned GDN `StateBundle` checkpoints (`mamba_value`), with independent lock counters (`full_lock_ref >= mamba_lock_ref`) and dual LRU lists.
    - Restores a prefix hit only up to the deepest node possessing a valid GDN `StateBundle` checkpoint (plus optional draft-context KV namespace), and enforces copy-on-write on partial tail pages and event-safe GPU recycling.
  - **Ping-Pong State Donation & Speculative Transaction Compatibility (`memory_pool.py:35-130`):**
    - `HybridReqToTokenPool` uses 2-slot ping-pong track buffers (`checkpoint_dst_indices`) to snapshot GDN state at chunk boundaries (`C`) and donate only **committed** checkpoints into the radix tree.
    - Verified in Phase 3 against a **synthetic speculative proposer** forcing partial rejections across chunk boundaries (**Decisive Gate Test #5**).
- **3.3 OpenAI HTTP/SSE Server & CLI (`w34 serve`, `w34 chat`):**
  - Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`, `/health`, `/metrics` (Prometheus engine latency histograms, request/token counters, KV/GDN occupancy, and hybrid prefix-token reuse).
  - **Production observability:** embedded React/Vite dashboard at `/` plus versioned `/api/metrics` JSON from the same telemetry as Prometheus. Distinguish active/evictable/free target resources, enqueued/running requests, and engine timing from HTTP timing. Use request-average time per output token rather than misleading speculative SSE-drain ITL. See [`METRICS.md`](METRICS.md) for the implemented contract and intentionally deferred features.
  - **Qwen3 Unified Reasoning & Tool Parser (`vllm/parser/qwen3.py` parity):**
    - `chat_template_kwargs: {"enable_thinking": bool}`, streaming `delta.reasoning_content` (`<think>...</think>`) vs `delta.content`, implicit `</think>` on `<tool_call>`, and both **Qwen3-XML** (`<function=...><parameter=...>`) and **Hermes JSON** `<tool_call>` extraction.
  - **Structured Outputs (`response_format` / `json_schema` / Regex / GBNF):**
    - Candidate-prefix token bitmask generation applied prior to top-k/top-p on GPU, with transactional rollback (`max_rollback_tokens`) for rejected speculative tokens.
- **Exit Gate:** Concurrent 8-agent and 16-agent serve benchmarks on `4B`/`9B`/`27B` meet or beat `vLLM` / `SGLang` TTFT, ITL, and aggregate tok/s; multi-turn prefix-cache warm TTFT drops by >80% and passes **Decisive Gate Test #5 (Prefix-Hit Synthetic Speculation & Partial Rejection)**.

---

### Phase 4: Speculative Decoding (`MTP` & `DFlash2` Proposers on Top of Existing Transactions)
**Goal:** Plug trained `Qwen3_5MTP` (`vLLM` donor) and `DFlash2DraftModel` (`SGLang` `DFlashWorkerV2` donor) proposers into the Phase 2/3 speculative commit transaction and `MambaRadixCache`.
- **4.1 GPU `RejectionSampler` & Transactional Commit:**
  - Evaluates `[anchor, draft_1, ..., draft_k]` (both greedy and stochastic residual verification), commits `1 + a` target KV and GDN `StateBundle` steps (`processed_len += 1 + a`), emits `a` accepted drafts + `1` recovered/bonus token (`emitted_len += 1 + a`, which becomes the next unconsumed anchor), rolls back grammar/draft-context state on rejection, and publishes only committed radix checkpoints.
- **4.2 Native `MTPProposer` (`qwen3_5_mtp.py` + `MtpFullAttn` Namespace):**
  - 1-layer TP-sharded MTP block (`eh_proj` `fc` + gated full-attention with dedicated `MtpFullAttn` KV namespace + `ParallelLMHead` + optional GPTQ `int4` draft MLP sidecar), running `k` draft steps inside captured CUDA graphs.
- **4.3 `DFlash2Proposer` (`SGLang` `dflash_worker_v2.py` & `dflash2.md` Parity):**
  - Consumes target hidden states captured via Phase 2's `TapPlan` (`[5, 19, 33, 47, 61]` on 27B), projects committed taps into the `DFlashDraftContext` KV namespace (`head_dim = 128`, non-causal query-relative left window `2048` on 27B; `BF16` full attention on 4B), runs the 5-layer drafter to propose **7 tokens (`block_rows = 8`: `1` anchor + `7` proposals on 27B; `16/15/16` on 4B)** with `TP=2`-correct global candidate selection, and verifies all **8 rows (`M=8`)** in one batched target pass.
- **Exit Gate:**
  - `9B` and `27B` `TP=2` seated `MTP` (`k=2, 3`) and `DFlash2` (`M=8`) match or beat `vLLM 0.31.0` (on `MTP`) and `SGLang v0.5.19` (`175.7 tok/s` code / `158.5 tok/s` prose on 27B `DFlash2`) in `profile-diff`, **with warm `MambaRadixCache` prefix hits restoring both target and proposer context**.

---

### Phase 5: Paged `TQ4` 4-Bit KV Cache (`warp3090` Donor), Vision, & `Qwen3.6-35B-A3B` MoE
**Goal:** Drop optimized `TQ4` / `Q8_0` paged kernels behind `KvLayerLayout`, then add multimodal vision and MoE as independent milestone tracks.
- **5.1 Paged TurboQuant 4-Bit & 8-Bit KV (`--kv-format tq4|tq4_all|q8_0|fp16`, `tq4` default):**
  - **Ported from `warp3090` (`op_tq4kv.cu`, `tq4_codebook.cuh`, `fattn-vec-tq.cuh`, `op_fattn_mma.cu`, `kvtier.rs`) into `KvLayerLayout` Paged Slabs:**
    1. **Block & Page Geometry:** `block_tq4_0` (`132 B` per `d=256` head row: two independent 128-wide least-squares `FP16` scales `d0, d1 = grp_norm * <v, c> / ||c||^2` + `128 B` packed 4-bit centroids = `4.125 bpw`, `3.88×` raw compression; `2,112 B` per 16-row head page) alongside `block_q8_0` (`272 B` per `d=256` head row, `1.88×` raw compression). *(Note: hybrid `K8V4` was evaluated and retired in `628e3b1` after plain `Q8_0` scored `95.3%–97.7%` exact token match vs `K8V4`'s `40.6%–55.8%`.)*
    2. **Hadamard Rotation (`R = W · diag(s)`):** Seed-42 Rademacher sign diagonal + single-warp register WHT-256 (`wht256_warp`) fused into KV page write, Q pre-rotation (`q' = W · diag(s) · q`), and O un-rotation (`o = diag(s) · W · o'`).
    3. **16-Centroid Beta(256) Lloyd-Max Codebook:** Stored in `__constant__` memory (`kTq4D256Centroids[16]`, `kTq4D256Boundaries[15]`).
    4. **Four Clean KV Policies (`tq4` default, `tq4_all`, `q8_0`, `fp16`):** Default `tq4` keeps boundary full-attention ordinals `{0, 1, N-2, N-1}` in **WHT-256 rotated `Q8_0`** (`272 B/row`) and interior layers in rotated LSQ `TQ4` (`132 B/row`) — delivering **`3.07×` effective KV compression** on 27B (`21,376 B/tok`, `5,344 MiB` at `262,144` tokens, fitting single-card `27B TP=1` at full `262k` context within 24 GB VRAM while matching `fp16` greedy parity `672/1536` vs `673/1536`); `tq4_all` compresses all layers to `132 B/row` (`3.88×`); `q8_0` uses unrotated `Q8_0` (`1.88×`).
    5. **Full Paged Shape Coverage on `sm_86` (`TP=1` & `TP=2`):**
       - Descriptor-driven `kv_cache_write` + 4-warp online-softmax `paged_attention_f16` across `t == 1` decode, `t <= 16` speculative verify, and `t > 16` chunked/prefix-hit prefill without materializing full-context `FP16` KV in VRAM (tensor-core GQA head-sharing & split-KV long-context acceleration tracked in Phase 6.2).
- **5.2 Multimodal Vision (`Qwen3_VisionTransformer`):**
  - PNG, JPEG, and binary PPM (`P6`) decoding (OpenAI `data:image/...;base64,...` + CLI `--image <path|url>`), image preprocessing (smart resize + bicubic + normalize), 24/27-layer ViT (`patch=16`, `temporal_patch=2`, `spatial_merge=2`, axial 2D RoPE), deepstack/MLP merger (`1152 → 4096/5120`), visual embedding splice, multimodal radix cache hashing, and 3D `mRoPE` media position IDs.
- **5.3 `Qwen3.6-35B-A3B` Hybrid MoE (`Qwen3_5MoeForConditionalGeneration` + Native MoE `MTP`):**
  - Top-k router softmax/sigmoid + two-stage fused `W4A16` / `FP16` MoE kernels + `shared_expert` + `shared_expert_gate` (`qwen3_next.py:143-259`), plus native 256-expert MoE `Qwen3_5MTP` speculative proposer (`TP=1` & `TP=2`).
- **Exit Gate:**
  - Single-card 27B (`tq4` at full `262,144`-token context) and `TP=2` 27B pass fragmented needle-in-a-haystack and `greedy-diff` gates within 24 GB VRAM alongside `MambaRadixCache` and speculative decode.
  - Multimodal vision passes `TP=1 == TP=2` `dump-diff` and multimodal `MambaRadixCache` warm-hit & content-hash isolation gates.
  - `Qwen3.6-35B-A3B` MoE passes `dump-diff` and `profile-diff` against `vLLM` (`1.52×` plain `TP=2` decode; `425.5 tok/s` with native MoE `MTP`).

---

### Phase 6: Roofline Polish, Proposer CUDA Graphs, Concurrent Serving, & Long-Context/VL Hardening
**Goal:** Polish and push Phases 1–5 to the `sm_86` hardware roofline across proposer latency, long-context `TQ4` attention, batched MoE prefill/verify, true multi-agent continuous-batching HTTP serving, and real-image `Qwen3.5-VL` / `262k` accuracy gates.
- **6.1 Proposer CUDA Graphs & Drafter Roofline (`MTP` & `DFlash2`, enhancing Phase 4):**
  - Capture per-rank CUDA Graphs for `Qwen3_5MtpRank` `M=1` draft steps (both dense and `35B-A3B` MoE, launched from a single host thread on `TP=2` without per-draft thread scopes) and `DFlash2Rank::forward` (`M=16` on `4B`, `M=8` on `27B`).
  - Optimize `DFlash2` `lm_head` / `BF16` GEMM dispatch (`vec8` fused RMSNorm/add + fast 2-stage block-radix top-k), meeting the **`1,100+ tok/s`** `4B TP=1 + DFlash2` target (**`1,228–1,388 tok/s`** measured across `FP16`/`Q8_0`/`TQ4`, beating `SGLang v0.5.19` streamed decode).
  - **Ratified `sm_86` HBM/PCIe Rooflines & Reopen Levers:**
    - **`4B + DFlash2` `< 2.0 ms/round` native-`BF16` stretch goal (`3.69 ms/r` measured vs `SGLang` `3.40 ms/r`):** Ratified as HBM-bandwidth-bounded on RTX 3090 Ti (`2.45 GB` unquantized `BF16` drafter + `FP16` `lm_head` = `2.43 ms` theoretical 100%-HBM floor at `1,008 GB/s`). **Reopen levers:** `W4A16`/`W8A16` quantized `DFlash2` drafter weights + vocabulary top-K shortlisting.
    - **`35B-A3B MoE TP=2 + MTP` `500+ tok/s` stretch goal (`445.7 tok/s` unprofiled / `340.2 tok/s` under `nsys`, `+31%` over pre-polish baseline after `MTP` vocabulary top-k + `vec8` RMSNorm):** Ratified as HBM + PCIe-bounded under the replicated `FP16` `lm_head` contract (`2.02 ms` 100%-HBM floor per pass + `86×` PCIe Gen4 x8 `topo PHB` sysmem all-reduce rounds without NVLink/P2P). **Reopen levers:** `TP=2` vocabulary-sharded `lm_head` (`124,160` cols/rank) + `W8A16` quantized `lm_head` + NVLink/P2P interconnect.
- **6.2 GQA-Grouped, Split-KV, & `mma.sync` Paged `TQ4` / `Q8_0` Attention (enhancing Phase 1.3 & Phase 5.1):**
  - Share each decoded `TQ4` (`132 B`) / `Q8_0` (`272 B`) / `FP16` KV tile across all GQA query heads per KV head (`4:1` / `8:1` register/smem broadcast, plus `mma.sync` tiles for `M ∈ [2..16]` speculative verify and chunked prefill), eliminating `4×–8×` redundant KV global-memory reads and WHT/dequant work.
  - Add parallel split-KV reduction across SMs for long contexts (`seq_len >= 4,096` out to `262,144`) so single-sequence decode and `M=4..16` verify saturate `1,008 GB/s` DRAM bandwidth at `32k–262k` context.
- **6.3 Token-Grouped Marlin `mma.sync` MoE Kernel (`moe_wna16`, enhancing Phase 5.3):**
  - Add token-grouped `(expert_id, token_slot)` sorting/binning + Marlin `mma.sync` `W4A16` tiles (`vLLM` `moe_wna16` donor) for `M >= 4` (speculative verify, multi-request continuous batching, and chunked prefill), while preserving the low-overhead warp-per-row GEMV path for `M = 1..3`.
- **6.4 `<think>`-Aware Structured Outputs, 100%-Schema Tool Calling, & Multi-Turn Agent ChatML (enhancing Phase 3.3 & Phase 4.1):**
  - **Real Grammar / JSON-Schema Engine (`TokenBitmaskConstraint` upgrade):** Replace the 1-token `{` stub in `parser.rs` with full JSON (`response_format: {"type": "json_object"}`), JSON Schema (`{"type": "json_schema", "json_schema": {..., "strict": true}}`), Regex/CFG, and **Structural Tags** (`llguidance` / native Rust token-trie FSM) plugged directly into our existing transactional `RejectionSampler` rollback (`speculative.rs`) so constrained decoding works losslessly alongside `MTP` and `DFlash2`.
  - **Reasoning-Deferred Constraint Activation:** Keep grammar unconstrained while inside `<think>...</think>` and latch schema enforcement immediately after `</think>` (`enable_in_reasoning = false` by default, matching `vLLM 0.31.0`).
  - **Qwen 3.5 / 3.6 Native Tool-Call Schema Guarantees & `tool_choice`:** Honor `tool_choice` (`"none"`, `"auto"`, `"required"`, and named `{"type": "function", "function": {"name": "..."}}`) and constrain `<tool_call>` arguments against each declared tool's JSON Schema in Qwen 3.5 / 3.6's native `Qwen3-XML` (`<function=name><parameter=key>...`) or `Hermes-JSON` format (avoiding the generic-JSON format mismatch noted on Qwen 3.6).
  - **Faithful Multi-Turn `chat_template.jinja` Serialization (`format_chatml_prompt`):** Always emit `<tool_call><function=...>` instructions when `system` + `tools` are both present, serialize historical `assistant.tool_calls` + `reasoning_content`, and wrap `role: "tool"` turns in `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>`.
- **6.5 Concurrent Continuous-Batching `w34 serve` & Complete OpenAI Controls (enhancing Phase 3.1 & Phase 3.3):**
  - Replace per-request `Mutex<Qwen3_5Engine>` serialization and post-hoc SSE buffering in `crates/w34-host/src/server.rs` with a dedicated background engine worker driving `SchedulerV1::step()` over an MPSC request queue and per-request bounded SSE channels (live token-by-token `delta.reasoning_content`, `delta.content`, and `delta.tool_calls` streaming + client-disconnect cancellation freeing `KvPool` & `StateBundlePool` slots).
  - Unify `SamplingParams` across non-speculative, speculative, and scheduled paths (fixing non-speculative `w34 serve` ignoring `temperature`/`top_p`/`top_k`/penalties/`seed`), and support `stop` strings (with partial-match streaming holdback), `stop_token_ids`, `logit_bias`, `min_tokens`, `ignore_eos`, `logprobs`/`top_logprobs`, `stream_options.include_usage`, and multi-request `GraphKey` decode buckets (`B ∈ {1, 2, 4, 8, 16}`).
- **6.6 GPU-Resident Stochastic `SamplerV1`, Real-Image `Qwen3.5-VL` Parity, & `32k–262k` Long-Context Gates (enhancing Phase 2.3, Phase 5.1, & Phase 5.2):**
  - Move non-greedy `SamplerV1` (`penalties → logit_bias → temperature → min_p → top_k/top_p → Gumbel-max`) and grammar bitmask application onto GPU so stochastic/structured requests never copy full-vocab (`248,320`) logits to host.
  - Verify real PNG/JPEG multimodal prompts on `Qwen3.5-4B` / `27B` / `35B-A3B` against `vLLM 0.31.0` `Qwen3.5-VL` golden dumps (`dump-diff` + `greedy-diff`: dense checkpoints enforce `rel_l2 <= 0.05` across `visual_embeddings`, `embed`, and `final_norm`; sparse MoE checkpoints `num_experts > 1`, where native `vLLM` exhibits `0.13–0.15` self-repeat `final_norm` drift across 40 discrete top-8/256 layers, enforce `sparse_moe_decomposed` — `visual_embeddings` & `embed` `rel_l2 <= 0.005` + exact `mRoPE`, same-input layer-0 MoE `rel_l2 <= 0.005`, `8/8` `greedy_exact` + `step0_top1_exact`, and `final_norm <= 0.20` native repeatability envelope), add opt-in `--allow-remote-images` HTTP(S) image fetching with strict timeout/byte caps, and run automated `32k / 64k / 131k / 262k` needle-in-a-haystack benchmarks comparing `tq4` vs `tq4_all` on 1× RTX 3090 Ti.
- **Exit Gate:**
  - 100% JSON Schema & multi-turn tool-call schema adherence across `<think>` + `MTP`/`DFlash2` speculative verification (`response_format: json_schema` + `tool_choice: auto|required|named`).
  - `nsys` `profile-diff` shows `full_attn` and end-to-end speculative decode matching or beating `vLLM 0.31.0` / `SGLang v0.5.19` on `4B`, `27B`, and `35B-A3B` MoE (within the ratified `6.1` `sm_86` HBM/PCIe proposer rooflines).
  - Concurrent 8-agent and 16-agent `w34 serve` HTTP/SSE load benchmark beats `vLLM` / `SGLang` in TTFT, ITL, and aggregate `tok/s` with live per-token SSE streaming.
  - `32k / 64k / 128k / 262k` 1-GPU `27B` `tq4` needle retrieval and real-image `Qwen3.5-VL` `vLLM` oracle parity pass 100%.

---

## 5. Five Decisive Pre-Freeze Gate Tests (Through-Line Verification)

Before declaring the Phase 1/2/3 spine frozen for production proposers and `TQ4`, five focused integration tests enforce that no hidden rewrite awaits in Phases 3–5:

1. **Gate 1 — Fragmented Paged Attention across Heterogeneous Row Strides (Phase 1):**
   - Identical logical sequence produces bit-identical attention outputs whether physical `page_id`s in `block_tables` are contiguous or randomly permuted, across both `512 B` (`FP16`) and synthetic `132 B` (`TQ4`) row layouts and `context_lens_before > 0` prefix-hit suffixes.
2. **Gate 2 — GDN Continuation, Checkpoint, & Rejection Parity (Phase 1):**
   - One-shot prefill vs chunked prefill (`BT = 64`), checkpoint restore from `checkpoint_dst_indices`, and multi-step verify with forced rejection at every step `a ∈ [0..k]` produce bit-identical `conv_state` (`FP16`), `ssm_state` (`FP32`), and layer outputs.
3. **Gate 3 — CUDA Graph Replay Under Metadata Mutation (Phase 2):**
   - Replaying captured `GraphKey` instances while permuting request order, `block_tables` page IDs, `mamba_state_indices`, `seq_lens`, and padded (`valid_token_mask = 0`) rows produces exact eager parity and leaves inactive GDN slots, RNG states, and `TapPlan` buffers untouched.
4. **Gate 4 — Cold-Start `TP=2` Initialization & Collective Stress (Phase 1/2):**
   - Fresh process startup runs `prepare_execution_plan()`, exercises every reachable kernel specialization under asymmetric rank arrival, and verifies zero lazy-module-load stalls and exact typed collective outputs (`FP16`, `BF16`, `FP32`, `INT32`).
5. **Gate 5 — Warm Prefix-Hit Speculation & Cross-Boundary Rejection (Phase 3):**
   - Request A populates `MambaRadixCache` (target KV + GDN checkpoint + draft-context namespace); Request B hits the cached prefix, runs speculative verify across a `C = 64` checkpoint boundary, forces a mid-block rejection, and matches a cold non-speculative reference token-for-token.

---

## 6. Summary Milestone Checklist

| Phase | Milestone | Primary Correctness Gate | Primary Performance Gate |
|---|---|---|---|
| **P0** | Seam Contracts, `vLLM`/`SGLang` Oracles, `profile-diff`, `CapacityPlanner`, Loader | All 5 contracts ratified; loader + capacity planner verify all 5 checkpoint families | `profile-diff` baselines captured for `vLLM 0.31.0` & `SGLang v0.5.19` (`4B`/`9B`/`27B`) |
| **P1** | Kernel Suite (`Marlin`, Norms/RoPE, Descriptor Attn, `vllm.cpp` GDN, `TP=2` Comm) | `dump-diff` parity vs `vLLM`/`SGLang` + **Gates #1, #2, #4** green | Every kernel ≤ `vLLM` / `SGLang` duration in `profile-diff` |
| **P2** | `TP=1` & `TP=2` Model Spine, `TapPlan`, Synthetic Spec Rollback, Composite Graphs | `greedy-diff` 12/12 (`4B`/`9B`/`27B`, `TP=1`/`2`) + **Gate #3** green | `w34 bench` & `profile-diff` ≥ `vLLM` on `TP=1` & `TP=2` |
| **P3** | V1 Scheduler, `SGLang` `MambaRadixCache`, OpenAI Server, Reasoning/Tools/JSON | OpenAI API + tool/reasoning/schema suite + **Gate #5** green | 8/16-agent `w34 serve` ≥ `vLLM`/`SGLang` TTFT, ITL, & steady tok/s |
| **P4** | Speculative Decode (`MTP` + `DFlash2` `8/7/8` atop Phase 2/3 Transactions) | Lossless greedy verify + warm prefix-hit proposer restore + `tau` parity | `9B` & `27B` `TP=2` `MTP` / `DFlash2` ≥ `vLLM` / `SGLang` (`175.7` code / `158.5` prose tok/s) |
| **P5** | Paged `TQ4` (`tq4` `3.07×` / `tq4_all` `3.88×` / `q8_0` `1.88×`), `Qwen3.5-VL` Vision, & `35B-A3B` MoE + MoE `MTP` | Fragmented needle + PNG/JPEG Vision `TP=1==TP=2` + MoE & MoE `MTP` `dump-diff` | 1-GPU 27B `262k` fits 24 GB (`48.8 tok/s`); `35B-A3B` `TP=2` `221.6 tok/s` (`425.5` w/ `MTP`) |
| **P6** | Proposer Graphs, GQA/Split-KV `TQ4`, `moe_wna16` `mma.sync`, 100%-Schema JSON/Tool Calls, Concurrent `w34 serve`, & `262k`/VL Gates | 100% JSON/tool schema adherence (`<think>` + spec) + `Qwen3.5-VL` parity + `32k–262k` needle | Proposer & `full_attn` ≤ `vLLM`/`SGLang` `ms/r`; 8/16-agent `w34 serve` ≥ `vLLM`/`SGLang` |
