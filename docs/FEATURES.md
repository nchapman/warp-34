# warp-34 — Comprehensive Front-to-Back Feature Inventory

`warp-34` is a clean-room Rust + CUDA inference engine engineered for **Qwen 3.5 / 3.6 / 3.8 hybrid models** (GDN linear attention + full attention + MoE + Vision + MTP/DFlash2 + Paged TQ4 KV) targeting **1–4× RTX 30-series (`sm_86`) and 40-series (`sm_89`) GPUs** (primary targets: RTX 3090 / 3090 Ti, RTX 4090, RTX 3060 Ti, and RTX 4060 Ti).

This document inventories implemented capabilities, APIs, execution contracts, kernel specializations, and tooling. **Implementation is not production qualification.** Scope-specific grades, evidence, limitations, and release gates live in [`READINESS.md`](READINESS.md); the launch checklist lives in [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).

The 22 section numbers are stable inventory references. Readiness IDs group their bullets into independently assessable contracts; implementation details inherit the assessment of their owning capability, not a separate production guarantee.

---

## 1. Command-Line Interface (CLI) & Operational Tooling
*Location: `crates/w34-host/src/cli/`, `crates/w34-host/src/main.rs`*

* **CLI Dispatch & Subcommands:**
  * `generate`: Standalone single/multimodal generation supporting plain, MTP, and DFlash2 execution across TP=1/2/4.
  * `bench`: Multi-stage benchmarking framework (`--stage decode | radix | scheduler | vision | all`) with configurable prompt/generation lengths, batch sizes, and JSON export (`--out <json>`).
  * `greedy-diff`: Automated token-for-token validation harness against reference golden outputs (`--golden <dir|json>`) with `--verify-lossless` (speculative parity) and `--verify-radix-warm` (prefix cache parity) verification.
  * `dump-diff`: Layer-by-layer tensor inspection harness dumping per-layer inputs/outputs across norms, attention, GDN, and MLP against `vLLM` / `SGLang` Safetensors fixtures (`--compare-tp2`).
  * `profile-diff`: Side-by-side SQLite/JSON performance profiler comparing round latencies (`ms/round`) and launch counts across 7 kernel categories against `vLLM 0.31.0` and `SGLang v0.5.19`.
  * `serve`: HTTP server implementing OpenAI Chat, OpenAI Responses, Anthropic Messages, and TypeSafe Decision model APIs.
  * `chat`: Single-turn terminal execution runner formatting ChatML prompts, evaluating reasoning and tool blocks via `Qwen3Parser`, and reporting generation statistics (prompt/cached/generated tokens, TTFT, decode tok/s) to stderr.
* **CLI Flags & Argument Validation:**
  * **Tensor Parallelism:** `--tp 1 | 2 | 4`.
  * **KV Cache Formats:** `--kv-format tq4 | tq4_all | q8_0 | fp16 | bf16` (default: `tq4`).
  * **Speculative Decoding:** `--speculative mtp | dflash2`, `--draft-model <id|path>`, `--mtp-k 1..3` (default 3), `--kv-pages <n>` (injects default of 256 target pages / 4,096 tokens when speculative decoding is active).
  * **Serving Options:** `--api-key <val>` or `--api-key-env <NAME>`, `--served-model-name <alias>`, `--max-num-seqs <1..16>` (default 16), `--max-model-len <positive integer>`.
  * **Model Hub & Offline Options:** `--revision <ref>`, `--draft-revision <ref>`, `--offline` (reads strictly from `HF_HUB_CACHE`).
  * **Multimodal Options:** `--image <path|url>`, `--allow-remote-images` (SSRF-protected HTTP/HTTPS retrieval).

---

## 2. Multi-API HTTP Gateway & Serving Architecture
*Location: `crates/w34-host/src/server/`, `crates/w34-host/src/server/http.rs`*

* **Asynchronous Axum Server Architecture:**
  * Complete decoupling of HTTP parsing, JSON serialization, and vision preprocessing onto an asynchronous CPU worker pool (`preprocessing`), protecting the bounded GPU `EngineWorker` from event-loop stalls.
  * Admission control semaphore (`admission`) limiting concurrent in-flight requests and guarding against memory exhaustion under burst traffic.
  * Bounded payload size (`BODY_LIMIT = 32 MiB`) and body timeout (`BODY_TIMEOUT = 10s`) returning HTTP 408/413 appropriately.
* **OpenAI Chat Completions API (`POST /v1/chat/completions`):**
  * Full ChatML compatibility with non-streaming JSON responses and streaming Server-Sent Events (`SSE`, `text/event-stream`).
  * Per-token streaming chunks with monotonic ping keep-alives and `usage` emission (emitted as a separate chunk with `choices: [] + usage` when `stream_options.include_usage` is set).
  * `reasoning_effort` mapping (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) emitting real-time streaming `reasoning_content` deltas.
  * Parallel tool-calling support with incremental function call argument streaming.
  * Dedicated input token counting endpoint: `POST /v1/chat/completions/input_tokens` (and unversioned alias `/chat/completions/input_tokens`).
  * Legacy completions fallback: `POST /v1/completions` with parallel token/logprob arrays.
* **OpenAI Responses API (`POST /v1/responses`):**
  * Implementation of the Responses routes and behaviors documented in [`GENERATIVE_APIS.md`](GENERATIVE_APIS.md), not a blanket claim of upstream specification completeness (`crates/w34-host/src/responses_api/`).
  * In-memory bounded session store (`ResponseStore`, default capacity 256 responses, 64 per credential scope) partitioned by credential-derived scope and supporting request chaining via `previous_response_id`.
  * Cryptographically unguessable 128-bit response IDs (`resp_<32hex>`) generated via `/dev/urandom`.
  * Stateful lifecycle routes: `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`, and `POST /v1/responses/input_tokens`.
  * Specialized 0-indexed monotonic `sequence_number` SSE event streaming (`response.created`, `response.in_progress`, `response.output_item.added`, `response.output_text.delta`, `response.completed`).
* **Anthropic Messages API (`POST /v1/messages`):**
  * Direct ingestion of Anthropic Messages wire format (`crates/w34-host/src/anthropic_api.rs`) translating system blocks, thinking blocks, user images, and `tool_use`/`tool_result` content.
  * Transparent Claude Code `cch=` billing-header stripping to maximize `MambaRadixCache` prefix-sharing hits.
  * Supports thinking budgets (`budget_tokens >= 1024`), adaptive thinking, and redacted thinking compatibility.
  * Streaming Anthropic SSE event format (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`) with typed tool use IDs (`toolu_*`).
  * Dedicated token count endpoint: `POST /v1/messages/count_tokens`.
* **Health, Discovery & Prometheus Telemetry:**
  * `GET /health`: Fast probe returning `{"status":"ok"}`.
  * `GET /v1/models`: OpenAI-compatible model catalog reporting active model ID and aliases.
  * `POST /tokenize` & `POST /detokenize`: Direct tokenization and detokenization endpoints with admission limiting.
  * `GET /metrics`: Standard Prometheus metrics export (`text/plain; version=0.0.4`) exposing TTFT, ITL, queue latencies, and cache utilization.
  * `GET /api/metrics`: Real-time JSON telemetry snapshot (`no-store`) powering operational dashboards.

---

## 3. Specialized Decision-Model Serving Engine
*Location: `crates/w34-host/src/decision.rs`, `crates/w34-host/src/clef.rs`*

* **TypeSafe Decision Model Endpoints:**
  * Exposes `POST /v1/systemone` (and unversioned `/systemone`) and `POST /v1/decisions` (and `/decisions`) for specialized decision pipelines without modifying generative routes.
  * **Clef (`ClefModel`):** Joint schema head (`ClefJointHead` + `joint_head.safetensors`) performing multi-question joint cross-attention over document text and multimodal image tokens.
  * **Kev (`KevModel`):** Pointer head (`KevPointerHead` + `head.safetensors`) computing query-key dot-product attention over `<|box_end|>` and `<|fim_suffix|>` anchors.
  * **Nimble (`NimbleModel`):** Parallel schema prompt scoring with `SafeJson` validation and calibrated temperature across up to 26 single-letter options (`A..Z`).
  * **Tev1 (`Tev1Model`):** Generative classification scoring over options `A..X` (up to 24 choices).
  * **OpenJev:** 52-choice evaluation (`A..Z` and `a..z`) with `true`-first probability normalization.
  * **Xor / Lev:** Dual forward and reverse option-order probability averaging to eliminate positional prompt bias.

---

## 4. API Authentication & Web Dashboard
*Location: `crates/w34-host/src/api_auth.rs`, `crates/w34-host/src/dashboard.rs`*

* **Digest-Based API Credential Layer:**
  * Opt-in protection for all inference, tokenization, model catalog, and response routes (while leaving health, metrics, and dashboard public).
  * `ApiKey` holds exclusively a SHA-256 fixed-length digest (`[u8; 32]`) and deliberately omits `Debug` implementations to prevent log leakage.
  * Constant-time comparison (`subtle::ConstantTimeEq`) guards against timing side-channels.
  * Accepts both RFC 6750 `Authorization: Bearer <token>` and `x-api-key` headers.
* **Embedded Self-Contained Web Dashboard:**
  * Self-contained read-only React dashboard compiled and embedded directly into the binary via `include_dir!`.
  * Strict Content Security Policy (`CSP: default-src 'none'; script-src 'self'; style-src 'self' ...`), anti-sniff headers, and path traversal guards (`.` and `..` rejection).
  * Serves `GET /` (`index.html`), `GET /favicon.svg`, and immutable cached static assets (`/assets/*`).

---

## 5. Native Rust XGrammar Structured Outputs & Constraints
*Location: `crates/w34-host/src/constraints.rs`, `crates/w34-host/src/request_constraints.rs`*

* **High-Performance Grammar Compilation:**
  * Pinned integration with native `xgrammar-rs 0.2.8` and Hugging Face `tokenizers 0.22`, eliminating custom Python/Triton grammars and runtime fallbacks.
  * Shared 512 MiB grammar compiler cache reusing compiled schemas across concurrent requests.
  * Supports JSON Schema (`response_format` and `structured_outputs`), JSON Object mode, regular expressions, Lark grammars, and EBNF syntax (`::=` auto-detection).
  * Choice lists: Compiles arbitrary lists of strings into structural tag alternatives (`choice: [...]`).
  * Schema sanitization and validation: Rejects non-disjoint unions, unsupported `$ref` scopes, and ambiguous constraint specs before dispatching to the model.
* **GPU Token Bitmasking & Speculative Rollback (`TokenBitmaskConstraint`):**
  * Directly computes packed 32-bit bitmasks (`valid_mask: Arc<Vec<u32>>`) applied inside the GPU sampler.
  * Speculative rejection rollback: Tracks matcher token history with `try_rollback_tokens` capability to roll back grammar state on rejected speculative drafts.
  * Decoupled reasoning gate: Maintains `<think>...</think>` tokens outside grammar constraints, activating grammar enforcement only upon reasoning termination.
  * Request stop-token injection: Binds request-specific stop tokens into the grammar EOS set, masking premature EOS emission until grammar terminal states are reached.

---

## 6. Tool-Calling & Reasoning Parsing Subsystem
*Location: `crates/w34-host/src/parser/`, `crates/w34-host/src/tool_constraints.rs`, `crates/w34-host/src/chatml.rs`*

* **Qwen3 XML Structural Tool System:**
  * Canonical Qwen XML envelope representation: `<tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>`.
  * Pinned `max_whitespace_cnt: 1` in native XGrammar rules prevents infinite empty-argument whitespace loops.
  * Independent per-parameter native schema compilation (`native_schema::compile`): Resolves local `$ref` targets using a bounded traversal budget and compiles grammars for argument typing.
  * Supports `tool_choice`: `auto`, `none` (disables tool generation and extraction), `required` (enforces at least one tool call), and named functions (`{type: "function", function: {name: ...}}`).
  * Non-strict schema compliance: Parameters accept raw strings, numbers, booleans, and JSON objects/arrays without requiring rigid schema strictness.
* **Streaming Parser & Reasoning State Machine (`Qwen3Parser`):**
  * Token-by-token state tracking across Reasoning (`<think>`), Content, and Tool Envelopes.
  * Emits incremental deltas (`reasoning_delta`, `content_delta`, and completed `tool_calls`).
  * Implicit `</think>` auto-transition: Automatically terminates reasoning mode when a `<tool_call>` tag appears inside the thinking block.
  * Duplicate tag absorption: Silently absorbs duplicate `</think>` tags emitted in content state.

---

## 7. Execution Engine & Continuous Batching Scheduler
*Location: `crates/w34-host/src/scheduler.rs`, `crates/w34-host/src/engine_worker/`*

* **Token-Budget Scheduler (`SchedulerV1`):**
  * Continuous batching with unified multi-request scheduling up to `max_num_batched_tokens` (default: 2,048) and `max_num_seqs` (default: 16).
  * Chunked prefill strictly aligned to the `MambaRadixCache` checkpoint boundary (`C = k * 64` tokens).
  * Mixed-batch scheduling: Bounds prefill chunk size to `mixed_chunk_prefill_size` (default: 256 tokens) whenever active decode/speculative verify sequences are present, protecting inter-token latency (p99 ITL).
  * Dynamic scratch-arena sizing: Automatically selects scratch allocations (`COMPACT_SCRATCH_ARENA_MAX_BYTES` 256 MiB, `EXTREME_LONGCTX_SCRATCH_ARENA_BYTES` 336 MiB, `LARGE_MODEL_TP1_SCRATCH_ARENA_BYTES` 384 MiB, `DEFAULT_SCRATCH_ARENA_BYTES` 512 MiB, `TIERED_MOE_SCRATCH_ARENA_BYTES` 1,056 MiB).
* **Engine Worker & Pipeline Overlap:**
  * Lock-free bounded channel ingress with immediate cancellation on caller disconnect, avoiding head-of-line blocking.
  * Double-buffered pipelined decode loop overlapping host scheduler bookkeeping and sampling setup with GPU execution.
  * Transactional multi-rank execution (`coordinated::replay_ranks`): Replays CUDA Graph submissions across TP ranks through an explicit `Upload -> Capture -> Launch` transaction barrier.

---

## 8. CUDA Graphs & Memory Arena Architecture
*Location: `crates/w34-host/src/cuda.rs`, `crates/w34-host/src/model/rank_runtime.rs`*

* **CUDA Graph Engine:**
  * Pre-captures execution graphs across plain decode (`decode_graph`), batched decode (`batched_decode_graphs` for batch sizes 1..16), and speculative verification (`spec_verify_graph` for M_verify 2..16).
  * Valid-token/valid-req masking: Padded rows write to designated sink locations, preventing corrupted recurrent states, RNG drift, or invalid KV writes.
  * Cold-start warmup protocol (`prepare_execution_plan()`): Resolves all dynamic symbols and executes warmup iterations before graph capture to prevent CUDA 13 lazy-loading deadlocks.
* **Pinned Host & Device Arenas:**
  * `DeviceArena` and sub-allocated buffers: Zero-allocation runtime operations using pre-allocated contiguous memory pools.
  * Zero-copy mapped-host embedding fallback (`plan_host_embed_offload`): Automatically offloads untied `embed_tokens` to pinned system memory on VRAM-constrained GPUs (e.g. 9B models on 8 GB RTX 3060 Ti) while preserving GPU execution for active layers.

---

## 9. Hybrid Radix Prefix Caching (`MambaRadixCache`)
*Location: `crates/w34-host/src/radix.rs` (Donor: `SGLang v0.5.19` `mamba_radix_cache.py`)*

* **SGLang-Donor Hybrid Radix Tree:**
  * Unifies full-attention KV cache pages and GDN recurrent `StateBundle` checkpoints into a single prefix tree.
  * **Dual-Lock Reference Counting:** Enforces `full_lock_ref >= mamba_lock_ref`. Retaining a node locks `mamba_lock_ref` on that node and `full_lock_ref` along its entire ancestral path `[node..root)`.
  * **Dual LRU Eviction Lists:** Manages `full_lru` and `mamba_lru` independently. Prefill hits refresh `full_lru` along the path to the root and `mamba_lru` on the terminal node.
  * **Internal Tombstoning:** Intermediate recurrent states can be evicted (`mamba_value = None`) to reclaim `~72 MiB` GDN state bundles while keeping underlying KV pages intact for long-context descendants.
  * **Ping-Pong Track Buffers (`ReqRadixState`):** 2-slot ping-pong tracking buffers capture intermediate GDN states at 64-token boundaries, donating only fully committed (`emitted_len`) snapshots into the global tree upon verify success.

---

## 10. Decoupled Memory Pools & State Transactions
*Location: `crates/w34-host/src/cache.rs`*

* **Typed KV Pool (`KvPool`):**
  * Logical `page_id = token_pos / 16` addressing completely decoupled from per-layer physical byte strides.
  * Layer-specific `KvLayerLayout` descriptors configure base pointers, head dimensions (128 or 256), formats, transforms, and stride bytes per layer.
  * Unified page allocation bitmap across all full-attention layers with lockstep TP-rank clamping.
* **Transactional GDN State Pool (`StateBundlePool` & `StateTx`):**
  * Pre-allocated pool of persistent GDN state bundles (`conv_state` FP16 + `ssm_state` FP32 in global memory).
  * Explicit 6-phase state machine: `Idle`, `Prefilling`, `Decoding`, `SpecDrafting`, `SpecVerifying`, `RolledBack`.
  * Winner-take-all speculative resolution: Reclaims losing speculative state slots upon rejection without data movement or memory fragmentation.

---

## 11. Multimodal & Vision Transformer Pipeline
*Location: `crates/w34-host/src/vision/`, `crates/w34-host/src/scheduled_vision.rs`, `crates/w34-kernels/csrc/ops/op_vision*.cu`*

* **Qwen3-VL Vision Architecture:**
  * Native implementation of `Qwen3_VisionTransformer` parameterized by model `VisionConfig` (e.g. 24 layers, 1,024 hidden size, 4,096 intermediate size, 16 heads on 4B/8B).
  * High-fidelity image preprocessing: `smart_resize` preserving aspect ratios up to 200:1, bicubic resampling, and normalization.
  * 2D bilinear `pos_embed` interpolation, 2D axial rotary position embeddings (RoPE), and `PatchMerger` with spatial merging.
  * 3D `mRoPE` coordinate generation across temporal, height, and width axes (`[3, N_max]`).
* **Multimodal Prefix Caching & Security:**
  * FNV-1a visual token content hashing integrates image feature embeddings directly into `MambaRadixCache` keys, enabling instant warm prefix restoration on repeated image queries.
  * SSRF-hardened remote image loader (`remote_images.rs`): Enforces 16 MiB download limits, 10s deadlines, zero redirect following, and strict IP blacklisting against private, loopback, link-local, carrier-grade NAT, and cloud metadata subnets.
* **Vision CUDA Kernels (`op_vision.cu`, `op_vision_fa2.cu`):**
  * Custom 2D RoPE and patch-merging CUDA kernels.
  * Dedicated Vision FlashAttention-2 (`op_vision_fa2.cu`) optimized for non-causal visual token self-attention.

---

## 12. Speculative Decoding Subsystem (MTP & DFlash2)
*Location: `crates/w34-host/src/speculative.rs`, `crates/w34-host/src/mtp/`, `crates/w34-host/src/dflash.rs`*

* **Native Multi-Token Prediction (`MTP`):**
  * Clean-room implementation of `Qwen3_5MTP` / `Qwen3_5MoeMTP` (vLLM donor, `crates/w34-host/src/mtp.rs`).
  * Single-layer TP-sharded predictor with projection, gated attention, dense/MoE FFN, and borrowed target LM head.
  * Online LM-head quantization: Quantizes unquantized target LM heads into a private `MarlinW4A16` (group size 64) proposal head on the fly when VRAM headroom permits.
  * Pre-captured proposal CUDA Graphs supporting up to 4 concurrent speculative slots (`MAX_MTP_BATCH_SLOTS = 4`).
* **DFlash2 Block Speculation:**
  * Clean-room 5-layer native BF16 speculative drafter (`DFLASH2_NUM_LAYERS = 5`, SGLang `DFlashWorkerV2` donor).
  * Verifies `M = K + 1` proposals (up to 16 tokens, e.g. M=8 on 27B/80B models, M=16 on 4B models) in a single batched forward pass.
  * Projects committed target taps into an isolated, disposable `DFlashDraftContext` KV namespace (`head_dim = 128`).
  * Low-rank candidate selector kernel (`op_dflash2.cu`, `DFLASH2_MAX_TOP_K = 16`, selector rank up to 1,024).
* **Speculative Transaction Contract & Rejection Sampling:**
  * Enforces the 4-length transactional invariant across verify rounds: `processed_len` (target consumed), `emitted_len` (caller visible), `reserved_len` (allocated headroom), and `draft_context_len` (proposer cache).
  * Speculative commit & rollback (`crates/w34-host/src/speculative.rs`): Atomically commits accepted drafts, emits bonus anchor tokens, and rolls back grammar matcher state.
  * Lossless dynamic speculative handoff (`tests/speculative_handoff.rs`): Dynamically yields speculative resources to incoming batched requests without dropping tokens or breaking state.

---

## 13. Model Architecture Support Matrix
*Location: `crates/w34-host/src/model/`, `crates/w34-kernels/csrc/ops/op_hc.cu`, `op_ple.cu`, `op_qsa.cu`*

* **Target Model Family Implementations:**
  * **Qwen 3.5 Dense Models (0.8B, 2B, 4B, 9B, 27B):** Hybrid architecture interleaving 3× Gated DeltaNet (GDN) linear attention layers with 1× Full Attention layer.
  * **Qwen 3.6 MoE Models (35B-A3B):** Hybrid GDN + Full Attention with routed experts (`num_experts_per_tok = 8`) plus shared expert.
  * **Qwen 3.8-Flash-Next MoE (80B total / 3B active, `qwen4_exp`):**
    * 512 total experts with 10 active experts per token (`num_experts_per_tok = 10`).
    * **HyperConnection (`HC`, `op_hc.cu`):** 4-stream multi-residual gating and mixing with per-stream `GemmaRMSNorm`, low-rank SiLU projections, and sigmoid stream routing.
    * **Per-Layer Embeddings (`PLE`, `op_ple.cu`):** Sharded zero-copy mmap F8_E4M3 embedding tables with stateful causal dilated depthwise conv1d and signed-square-root sigmoid gating.
    * **Quality-Scored Sparse Attention (`QSA`, `op_qsa.cu`):** Top-64 block indexer scoring (1,024 active tokens) with ReLU-sum MQA and sparse paged GQA decode.
* **Precision Policy Invariants:**
  * End-to-end `FP16` target model spine (hidden states, residuals, norms, attention Q/K/V/O, Marlin activations).
  * Native `BF16` preservation for DFlash2 drafters to prevent dynamic range overflow.
  * `FP32` persistent recurrent states and reduction trees (`ssm_state` `[H_v, 128, 128]`, RMSNorm variance, softmax denominators, and logit distributions).

---

## 14. Model Loading & Marlin Quantization Engine
*Location: `crates/w34-host/src/loader/`, `crates/w34-kernels/csrc/marlin/`, `op_marlin.cu`*

* **Universal Checkpoint Ingestion:**
  * Direct memory-mapped SafeTensors loading with automated shard resolution from local directories or Hugging Face Hub (`HF_HUB_CACHE`).
  * Supported formats: Unquantized `FP16`/`BF16`, `GPTQ`, `AWQ`, `Compressed-Tensors` (`pack-quantized`), and `AutoRound`.
* **Hardware-Accelerated Marlin Repack:**
  * Loads quantized checkpoints and repacks weights at startup into unified Marlin tile formats (`W4A16` U4B8, `W4A16-ZP`, `W8A16`, and `W4A8`).
  * Supports group sizes 0 (channelwise), 32, 64, and 128 with static/dynamic act-order.
  * Marlin GEMM kernels optimized for `sm_86` and `sm_89` with asynchronous global-to-shared copies (`cp.async`) and mma tensor core operations.
  * 1-Slice grid invariance: Enforces deterministic tile-count divisor CTA grids for `M <= 16` shapes, guaranteeing identical FP32 reduction trees between TP=1 and TP=2.

---

## 15. Tiered MoE & Asynchronous CPU Offload Subsystem
*Location: `crates/w34-host/src/moe_offload/`, `crates/w34-host/src/moe_cpu_bridge/`, `crates/w34-kernels/csrc/ops/op_moe*.cu`*

* **512-Expert Tiered Offloading Engine:**
  * Designed for memory-constrained execution of 80B models (Qwen3.8-Flash-Next) on 8 GB–24 GB GPUs.
  * `ExpertPackLayout`: Contiguous 4KiB-aligned DMA layout containing U4B8 weights and unpermuted FP16 scales.
  * **Smooth BIP-LRU Cache:** Manages VRAM hot expert cache with a smoothed `k_protected` floor (`tail_budget = (total_pairs <= 40) ? 40 : min(40, k_hot / 2)`), preserving anchor slots during speculative verify rounds.
  * Dual execution modes: `Auto` dynamically selects between fast `GpuLru` (for small-batch decode/verify) and `StreamedPrefill` (for long prefill sequences).
* **Asynchronous Hybrid CPU Bridge:**
  * Offloads cold expert evaluation for decode and speculative verify tokens (`M <= 8`) to multi-threaded host workers (AVX-512 and AVX2 optimized).
  * Double-buffered pinned host mailbox with lock-free sequence counters and atomic job stealing.
  * Asynchronously accumulates CPU-evaluated cold expert down-projections into GPU activation buffers over PCIe while the GPU computes resident hot experts.
* **Unified GPU LRU Pull Kernel (`op_moe_router.cu`):**
  * Consolidated `moe_pull_mapped_experts_kernel` performs DMA transfers from host memory to VRAM hot slots in a single GPU-driven launch.

---

## 16. Recurrent Linear Attention Kernels (Gated DeltaNet)
*Location: `crates/w34-kernels/csrc/ops/op_gdn.cu`, `op_gdn_wy.cu`, `op_gdn_wy_chunk64.cu`*

* **Native CUDA C++ GDN Implementation:**
  * Ported and hardened from `vllm.cpp` (`50d425bec`), verified against `vLLM` FLA golden dumps.
  * State dtypes: `conv_state` in `FP16` (`[slots, conv_dim, 3]`), `ssm_state` in `FP32` (`[slots, H_v, 128, 128]`).
  * `gdn_causal_conv1d_f16_kernel`: Stateful causal 1D convolution with SiLU activation and slot-based snapshot writing.
  * `gdn_fused_recurrent_delta_rule_kernel`: Decode (`q_len = 1`) and speculative verify (`q_len = 1 + k`) recurrence with fused sigmoid gating and delta update rules.
* **Chunked GDN-WY Prefill Kernels:**
  * 3-stage chunked prefill algorithm over 64-token tiles (`BT = 64`):
    * Stage 1: Computes intra-chunk Householder-like representation (`W` and `U` matrices).
    * Stage 2: Evaluates inter-chunk recurrent state propagation (`Delta H`).
    * Stage 3: Produces final outputs with FP32 accumulators.
  * Continuation/checkpoint tests exercise exact parity for their declared geometries; this is not a universal 0-ULP guarantee across every prefill kernel and full-model configuration.

---

## 17. Attention & TurboQuant 4-Bit KV Cache Kernels
*Location: `crates/w34-kernels/csrc/tq/`, `op_kv_cache.cu`, `op_hadamard.cu`, `op_paged_attn.cu`, `op_paged_fa2.cu`*

* **TurboQuant 4-Bit KV Cache (`TQ4`):**
  * Pinned in-house implementation with **`3.07×` effective KV compression** for the documented 27B layout (21,376 bytes/token). Historical 262k-context runs on a single 24 GB GPU are recorded in [`STATE.md`](STATE.md); capacity and quality depend on the checkpoint, other memory allocations, and workload, and require release-specific qualification.
  * `block_tq4_0`: 66 bytes per 128 elements (4.125 bpw) with independent per-128 least-squares FP16 scales.
  * Randomized orthonormal rotation: Seed-42 Rademacher + Fast Walsh-Hadamard Transform (WHT-256) eliminates outlier dimensions prior to quantization.
  * 16-centroid `Beta(256)` Lloyd-Max codebook stored in `__constant__` memory and register shuffle caches.
  * **Boundary-Anchored KV Policy:** The last 4 full-attention layers remain in unrotated `Q8_0` (272 B/row) while preceding layers use `TQ4` (132 B/row), limiting quantization error. TQ4 is lossy: historical greedy token agreement varies by model and prompt; see [`STATE.md`](STATE.md) and the quality gate in [`READINESS.md`](READINESS.md).
* **Paged Attention & FlashAttention Kernels:**
  * Cooperative 1-warp GQA decode kernel with register-shuffled TQ4 centroid dequantization.
  * Tensor core MMA decode and prefill kernels (`mma.sync.m16n8k16`) with Split-KV grid reductions.
  * Integrated Paged FlashAttention-2 (`op_paged_fa2.cu`) supporting variable-length causal batches.

---

## 18. Fused Layer Norms, RoPE, Activations & Small-N GEMMs
*Location: `crates/w34-kernels/csrc/ops/op_norms_rope_act.cu`*

* **Fused Operators:**
  * `GemmaRMSNorm`: Vectorized (vec8) RMSNorm with `(1.0 + weight)` unit offset and FP32 variance accumulation.
  * `fused_qk_rmsnorm_rope_gate`: Fuses Q/K split, per-head RMSNorm, partial interleaved 3D `mRoPE` (rotary dimension 64, sections `[11, 11, 10]`), and output projection gating into a single memory pass.
  * `silu_and_mul`: Fused SwiGLU / SiLU activation and elementwise multiplication with optional Hadamard transform injection.
  * `RMSNormGated`: Fused post-GDN normalization with SiLU gating (`norm_before_gate = true`).
* **M-Invariant Small-N GEMM Kernels:**
  * Custom `small_n_gemm_4x4_f16_kernel` and `small_n_gemm_dual_4x4_f16_kernel` handling skinny projection matrices (`N <= 2560`, PLE projections `N = 10240`).
  * Eliminates cuBLAS kernel switching between `M = 1`, `M = 4`, and `M = 8`, ensuring bit-exact 0-ULP reduction invariance across plain and speculative decode steps.

---

## 19. Cross-GPU Collectives & Tensor Parallelism
*Location: `crates/w34-kernels/csrc/ops/op_collectives.cu`*

* **High-Efficiency TP Collectives:**
  * Primary verified implementation: Pinned-sysmem TP=2 (`op_collectives.cu`) over PCIe Gen4 x8 (e.g. 2× RTX 3090 Ti / 3060 Ti / 4060 Ti / 4090).
  * System-scope release/acquire memory fences (`st.release.sys.global` / `ld.acquire.sys.global`) with monotonic epoch parity: small in-kernel transfers use one barrier; the large Copy-Engine DMA path uses two barriers.
  * Dynamic DMA cutoff (`kDmaThresholdBytes = 64 KiB`): Selects in-kernel register writeback for small transfers and asynchronous Copy-Engine DMA for bulk payloads.
  * Strictly typed collective entrypoints: Typed routines for `FP16` hidden states, `BF16` draft tokens, `FP32` reductions, and `INT32` token IDs (never casting integer tokens through float collectives).
  * Implemented NVLink P2P VRAM staging and TP=4 execution plans (`crates/w34-host/src/cuda/collectives.rs`), with historical 4× RTX 3090 qualification in [`STATE.md`](STATE.md). These require separate hardware/topology requalification for a release.

---

## 20. GPU Sampler & Speculative Rejection
*Location: `crates/w34-kernels/csrc/ops/op_sampler.cu`, `op_rejection.cu`*

* **9-Step Persistent GPU Sampler:**
  * GPU sampling uses device scratch buffers, reductions, and specialized fast paths; random draws are supplied by host `SamplerV1`. The policy includes:
    1. Grammar token bitmask application (`-INFINITY` masking).
    2. Logit bias adjustments.
    3. Repetition, frequency, and presence penalty applications.
    4. Temperature scaling.
    5. Min-p dynamic threshold filtering.
    6. Top-k and Top-p (nucleus) sorting and cumulative mass truncation.
    7. Seeded sampling using host-generated, position-keyed SplitMix64 uniform draws passed in the row configuration.
    8. Fast-greedy argmax shortcut with warp/block reductions for eligible rows (not a one-cycle latency guarantee).
    9. FP32 logprob extraction with configurable `top_logprobs` (up to 20 in the full sampler).
* **GPU Speculative Rejection Sampler:**
  * Parallel verification of speculative draft tokens against target acceptance probabilities (`p / q`).
  * Supports both greedy acceptance and stochastic residual distribution sampling (`sample max(p - q, 0)`).

---

## 21. Real-Time Telemetry & Verification Framework
*Location: `crates/w34-host/src/telemetry.rs`, `tools/oracle/`*

* **Thread-Safe Metrics & Prometheus Monitoring:**
  * Non-blocking CPU snapshots tracking request counts (submitted, running, completed, failed, cancelled).
  * Token accounting: Separates prompt, computed, radix-cached, and generated tokens.
  * High-resolution latency histograms: Time-To-First-Token (`ttft`), Queue Wait Time (`queue`), End-to-End Latency (`e2e`), and Inter-Token Latency (`time_per_output_token`).
  * Memory pool utilization: Tracks live capacity, active pages, cached/evictable pages, and free physical pages across KV and GDN pools.
* **Continuous Parity Oracle Suite (`tools/oracle/`):**
  * Automated Python/Triton test harnesses running `vLLM 0.31.0` and `SGLang v0.5.19` side-by-side with `warp-34`.
  * Provides a 12-prompt canonical multi-turn/reasoning oracle suite. Exact token matches, numerical tensor tolerances, and adjudicated divergences are distinct outcomes; historical results are not a guarantee of universal token equivalence or a fresh release pass.

---

## 22. Containerization & Multi-Architecture Fatbinary Packaging
*Location: `Dockerfile`*

* **Optimized Docker Deployment:**
  * Multi-stage build process decoupling the Node/pnpm React dashboard compilation from the CUDA/Rust binary build.
  * Runtime image based on `ubuntu:26.04` (historical shared-link runtime filesystem size ~775.6 MB; the ~575.4 MB static variant was an experiment, not the current default; see [`CUBLAS_PACKAGING_EXPERIMENT.md`](CUBLAS_PACKAGING_EXPERIMENT.md)) without shipping multi-GB CUDA toolkits, headers, or driver stubs.
  * Shared cuBLAS dynamic linkage (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`) stored in a dedicated cacheable filesystem layer.
  * Security hardening: Runs under an unprivileged system user (`w34`), mounts standard Hugging Face caches (`HF_HUB_CACHE`), and defines container health checks querying `GET /health`.
* **Multi-Arch Fatbinary Support:**
  * Compiles native SASS cubins for both Ampere (`sm_86`) and Ada Lovelace (`sm_89`) plus compute_86 PTX fallback.
  * Embedded device code compression enabled (`W34_NVCC_EXTRA=--compress-mode=size`).
  * Dynamic hardware detection (`w34_device_props`) adapts CTA grid dimensions and shared memory allocations at runtime to the specific GPU SM count (from 28 SMs on RTX 3060 to 142 SMs on RTX 4090).
