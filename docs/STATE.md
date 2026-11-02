# warp-34 — Session Orientation & Current State

## Status: Phases 0–6 complete (with ratified `sm_86` HBM/PCIe proposer rooflines in `ROADMAP.md` §4 `6.1`)

The [Phase 6 criterion-by-criterion evidence ledger](PHASE6_EVIDENCE.md) records
every Phase 6 exit-gate verification artifact, SHA-256 hash, donor comparison,
and ratified `sm_86` hardware roofline.

`warp-34` has completed **Phase 0** (Foundational Seam Contracts, Oracle & `nsys` Diff Harness, CUDA Runtime & `CapacityPlanner`, Unified Safetensors Loader), **Phase 1** (Donor-Aligned Kernel Suite + Decisive Gate Tests #1, #2, #4), **Phase 2** (Qwen 3.5 / 3.8 Hybrid Backbone on `TP=1` and `TP=2`, `KvPool`, `StateBundlePool`, `CUDAGraphRunner` Gate #3, and `w34` CLI), **Phase 3** (`SchedulerV1` token-budget continuous batching, `SGLang`-donor `MambaRadixCache` + `HybridReqToTokenPool` `extra_buffer` ping-pong GDN checkpoint donation passing **Decisive Gate Test #5**, `SamplerV1` + structured-output bitmask + `Qwen3Parser`, and OpenAI HTTP/1.1 + SSE server `w34 serve` / `w34 chat`), **Phase 4** (Speculative Decoding: `RejectionSampler`, Native Dense + MoE `Qwen3_5MTP`, Native `BF16` `DFlash2`, and fixed-shape `M_verify = K + 1` speculative verify CUDA Graphs), **Phase 5** (Paged `TQ4` (`tq4` default with rotated `Q8_0` boundary layers + per-128 least-squares `TQ4` interior at `3.07×`, plus `tq4_all` at `3.88×` and `q8_0` at `1.88×`, fitting `27B` full `262,144`-token context on 1× RTX 3090 Ti), Multimodal Vision `Qwen3_VisionTransformer` with PNG/JPEG/PPM decoding + 3D `mRoPE` + multimodal `MambaRadixCache` hashing + OpenAI `image_url`, and `Qwen3.5` / `3.6` Hybrid MoE `Qwen3_5MoeForConditionalGeneration` `35B-A3B` on `TP=1` and `TP=2`), and **Phase 6** (per-rank `MTP`/`DFlash2` CUDA Graphs, GQA/Split-KV/FA2 attention, grouped-MMA MoE, `<think>`-aware structured outputs & native tool-call schemas, scheduled Axum/SSE `w34 serve` with in-graph `FastGreedy` + 2-slot pipelined decode overlap + multi-stream GDN `in_proj_ba` fork-join beating `SGLang v0.5.19` at `N=8` and `N=16`, GPU-resident stochastic `SamplerV1`, `6/6` canonical `Qwen3.5-VL` PNG/JPEG gates, and `8/8` 1-GPU `27B` `32k–262k` needle gates).

All 5 Decisive Pre-Freeze Gate Tests (**Gate #1–#5**) pass in `crates/w34-host/tests/phase0_1_2_gates.rs` and `crates/w34-host/tests/phase3_gates.rs`. All 4 dense target model configurations (`4B TP=1`, `9B TP=1`, `9B TP=2`, `27B TP=2`) pass `w34 greedy-diff`, `w34 dump-diff`, and `w34 profile-diff`, **beating `vLLM 0.31.0` by +31% to +83% in single-stream decode throughput** on 1–2× RTX 3090 Ti (`sm_86`) before speculative decoding or `TQ4` compression, and achieving a **95.2% warm prefix TTFT drop** (`145.68 ms` → `7.05 ms`) via `MambaRadixCache`.

---

## `Qwen/Qwen3.8-Flash-Next` (`qwen4_exp` / `Qwen4ExpForConditionalGeneration`) on 1× and 2× RTX 3090 Ti

- Implemented end-to-end native support for `Qwen/Qwen3.8-Flash-Next` (`model_type: "qwen4_exp"`, `Qwen4ExpForConditionalGeneration`) across 1× RTX 3090 Ti (`TP=1`) and 2× RTX 3090 Ti (`TP=2 / EP=2`), documented in [`QWEN_FLASH_NEXT.md`](QWEN_FLASH_NEXT.md):
  - **`QF-0` (Config, Hub, `PleMmapTable`, `CapacityPlanner`)**: Extended `ModelConfig` and `hub::files` for `qwen4_exp`, AutoRound + FP8 `PLE` hybrid checkpoints (`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`), zero-copy 128-shard `PleMmapTable` (`49.68 GiB` `F8_E4M3` mmap + prime-modulo `splitmix64` bigram/trigram hashing), and `QwenFlashNextCapacityPlan` for `TP=1` and `TP=2 / EP=2`.
  - **`QF-1` (`HyperConnection`, `PLE`, `QSA` CUDA Kernels)**: Added `op_hc.cu` (`w34_hc_expand_streams_f16`, `w34_hc_rmsnorm_f16`, `w34_hc_down_silu_inject_f16`, `w34_hc_mix_streams_f16`, `w34_hc_combine_streams_f16`, `w34_qwen4_mtp_broadcast_add_f16`), `op_ple.cu` (`w34_ple_dequant_fp8_to_f16`, `w34_ple_grouped_gemma_rmsnorm_f16`, `w34_ple_gate_scale_f16`, `w34_ple_short_conv_f16`), and `op_qsa.cu` (`w34_qsa_pre_index_norm_rope_f16`, `w34_qsa_update_index_k_cache_f16`, `w34_qsa_select_blocks_f16`, `w34_qsa_paged_attention_f16`).
  - **`QF-2` (Tiered 512-Expert MoE Offloading)**: Added `op_moe.cu` and `crates/w34-host/src/moe_offload.rs` supporting 512 routed + 1 shared expert per layer across 48 layers (`HostExpertStore` with parallel in-place `[K/8, N] -> [N, K/8]` transpose and `HostPackStorage::Mapped`/`Heap`, `HotExpertCache` VRAM slots, device-side `w34_moe_lru_resolve` + zero-copy PCIe UVA fallback decode, CPU/PCIe hybrid decode, and double-buffered async H2D streamed prefill across `TP=1` and `TP=2 / EP=2`).
  - **`QF-3` (48-Layer `Qwen4Exp` Spine, `StateBundlePool`, `KvPool`, `MambaRadixCache`, `CUDAGraphRunner`)**: Assembled the 48-layer `3:1` `GDN:QSA` spine with `PLE` (`ple_conv_state_buf` + `ple_history_tokens` in `StateBundlePool`, `qsa_raw_k_buf` + `qsa_comp_k_buf` in `KvPool`), `MambaRadixCache` prefix reuse, and CUDA Graph capture/replay. Verified on `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`:
    - `TP=1` (1× RTX 3090 Ti): `26.9–30.1 tok/s` decode (`" Paris, which is also the largest city in the country..."`), `MambaRadixCache` 64-token prefix reuse bit-identical (`1543.00 ms -> 347.32 ms` TTFT).
    - `TP=2 / EP=2` (2× RTX 3090 Ti): `41.95–42.31 tok/s` decode (`20.81 ms` TTFT), `MambaRadixCache` 64-token prefix reuse bit-identical (`654.10 ms -> 21.49 ms` TTFT, matching `TP=1` token-for-token `ids=[760, 6511, 314, 9338, 369, 760, 6511, 314]`).
  - **`QF-4` (Native `Qwen4` `MTP` Speculative Decoding)**: Implemented `Qwen4MtpSubsystems` and `forward_qwen4_inner` in `crates/w34-host/src/mtp.rs` with dual-stream `residual_linear_shared` (`pre_fc_norm_embedding` + `fc_embedding` + `pre_fc_norm_hidden` `[hc_count * H]` + `fc_hidden` + `w34_qwen4_mtp_broadcast_add_f16`), 1-layer `HC + QSA + MoE + final_hc_mixer`, and QSA step-0 top-k block index reuse (`skip_topk = step > 0`), verified losslessly on `TP=1` and `TP=2 / EP=2` in `crates/w34-host/tests/qwen4_flash_next.rs`.
  - **`QF-5` (Next Performance Milestones vs. Reference Headline Numbers — documented in [`QWEN_FLASH_NEXT.md` §5](QWEN_FLASH_NEXT.md#5-current-baseline-vs-reference-headline-numbers--next-performance-milestones-qf-5))**:
    - Current real-checkpoint measurements (`26.9–30.1 tok/s` on `TP=1`, `41.95–42.31 tok/s` on `TP=2 / EP=2`) reflect cold, un-speculated (`M=1` target-only) 16-token runs, matching the reference un-speculated cold/static hillclimb (`34.7–49.8 tok/s`).
    - To reach and beat the reference `MTP-3` + warm-cache headline numbers (`43–51 tok/s` on `1× 3090`, `84–111.5 tok/s` on `2× 3090`):
      1. Auto-discover and load `<snapshot>/runtime/mtp-int4-g32` (`compressed-tensors` `INT4 group_size=32` routed experts, `3.85 GiB`) in `Qwen3_5MtpRank::load_from_dir` so `MTP-3` runs on the real `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` checkpoint.
      2. Pre-seed `HotExpertCache` at startup from `<snapshot>/runtime/static_hot_cache_rankings.json` (`hot32` on `TP=1`, `hot84–104` on `TP=2 / EP=2`).
      3. Shard `HostPackStorage::Mapped` into per-layer / 2-GiB `PinnedHostBuffer` slabs (avoiding the 32-GiB single-allocation limit on `TP=1`'s `58.5 GiB` expert store) and wire the multi-core AVX2/AVX-512 cold-expert worker pool for `TP=1` decode.
      4. Add optional `MTP` draft `lm_head` active-vocabulary range filtering (`[[0, 65536], [248044, 248320]]`).

---

## Public GitHub publication and Git LFS

- Public repository: https://github.com/nchapman/warp-34 (`origin`, `main`).
- Migrated tracked `*.safetensors` history to Git LFS, including historical
  fixtures exceeding GitHub's 100 MB regular-file limit. Commit IDs changed;
  the serving-controls commit is now `d2890e3` (formerly `a7da87a`).
- Uploaded all 17 LFS objects reachable from `main` (~300 MB). `git lfs fsck`
  passes, all 521 pre-existing tracked working files are byte-identical after
  checkout, and no regular blob >=10 MB remains in any local branch history.
  The detached historical benchmark worktree was left unchanged.
- Local pre-migration recovery bundle: `/tmp/warp-34-pre-lfs-a7da87a.bundle`.
  See README for clone/setup instructions. No runtime code or fixture bytes changed;
  the preceding full-suite verification remains applicable.

## Production serving controls after oracle option audit

- Compared the pinned vLLM 0.31.0 and SGLang v0.5.19 CLI surfaces, then selected
  deployment controls rather than copying engine/backend tuning flags. Full
  contract and deferred scope: [`SERVING.md`](SERVING.md).
- **Optional** `--api-key` / `--api-key-env NAME`: no authentication by default.
  Bearer or `x-api-key` authentication covers inference, model-list, tokenization,
  token-count, and stored-Responses APIs, including unversioned aliases, before
  request preprocessing. Fixed-size SHA-256 digest comparison uses `subtle`;
  keys are not logged. Dashboard/assets, health, and metrics remain public by
  explicit policy; protect the whole listener with a proxy/firewall when needed.
- `--served-model-name` advertises a stable public alias without changing model
  acquisition. Without it the original model ID/path remains public (not the
  resolved Hub snapshot path). Request model-name routing remains permissive.
- `--max-num-seqs` is bounded to **1–16**, matching fixed decode graph IO, and
  reaches both GDN pool planning and worker scheduling. `--max-model-len` is an
  optional prompt + output admission ceiling, checked for all worker requests,
  including decisions, before enqueue. It does not allocate KV or extend model
  context; it cannot exceed actual allocated KV capacity. Defaults are unchanged.
- Serve-only options are rejected on other commands. `serve` now rejects unused
  generation/benchmark/oracle flags, especially `--batch-size` (use
  `--max-num-seqs`). Its regression test first failed by reaching model resolution,
  then passed after validation was added.
- Corrections to the initial audit: our KV formats are not a blanket superset of
  donor formats; a shared KV pool capacity is not a model context guarantee;
  remote-image enablement is not a domain allowlist. TLS/CORS can live at a proxy,
  but logging, watchdogs, and drain policy remain separate operational work.
  Retained FP16 target precision, SDK-owned Hub policy, automatic quantization,
  existing scheduler tuning, per-request seed, and `n=1` rather than adding knobs
  without a demonstrated product need.
- Verification: final `CUDA_VISIBLE_DEVICES=0,1 mise run check` passed (378.87 s;
  340 Rust tests/doctests passed, 3 existing opt-in tests ignored; 69 oracle,
  51 perf-suite, and 18 frontend tests; CPU/CUDA Clippy and formatting). Separate
  `mise run test-cpu` passed (191 tests, 1 ignored), as did `mise run build`,
  relative documentation-link checks, and `git diff --check`. Initial Prettier
  and two Clippy style failures were fixed before the successful full rerun.
- Live GPU integration covers protected/public routes, aliases, context rejection
  across four APIs (JSON and SSE), decision prefill rejection, and two-active
  concurrency. The release CLI also passed env-key-enabled and unauthenticated
  SSE smoke tests, alias/health/metrics checks, graceful SIGTERM shutdown, and
  rejection of a context cap larger than its allocated KV pool. GPUs are clear.
  Local logs: `/tmp/w34-serving-{check,cpu,build}.log`; CLI smoke script:
  `/tmp/w34-serving-cli-smoke.py`. No new kernel-performance or donor-speed claim.
- Review found no actionable issues introduced here. Two repeated pre-existing
  Hub suggestions were left out of scope: transport timeouts remain delegated
  to the SDK per the prior explicit decision; the progress-thread clock-subtraction
  edge case is unrelated to serving controls. No follow-up review was needed.

---

## Qwen3.8-Flash-Next branch (`qwen-flash-next`)

- QF-0 config/Hub parsing, zero-copy PLE mmap storage and TP=1/TP=2 capacity
  planning are committed in `8786fb1`; QF-1 HC/PLE/QSA primitives and local
  math-parity tests are committed in `71fb030`. These do **not** establish
  end-to-end Qwen4Exp generation or reference-dump parity for the assembled model.
- QF-2 is in progress. `moe_offload.rs` now defines a checked, page-aligned
  N-major U4B8 expert pack and budget-bounded immutable pinned `HostExpertStore`.
  It reuses `RawLinearHost` quantized loading, validates symmetric zero points
  and static group indices, and partitions whole experts for EP=1/EP=2.
- New CUDA helpers scatter staged packs into bounded cache slots and remap
  global routed IDs to local slots; HC/PLE/QSA and offload helpers are registered
  in cold-start warmup. The staging geometry regression was reproduced before
  fixing incomplete-group acceptance; scatter/remap GPU tests passed afterward.
- Checkpoint qualification loads two actual experts from the pinned albucino
  snapshot and compares TP=1 and EP=2 pack bytes against checkpoint words.
  After Clippy fixes and review-requested destination bounds/alignment checks,
  `CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --lib moe_offload
  -- --include-ignored --test-threads=1` passed **5/5**, including the explicit
  checkpoint qualification. The subsequent `mise run check` passed (exit 0),
  including CPU/CUDA Clippy, frontend/oracle/performance checks and the release
  CUDA suite. Host library results: **163 passed, 3 ignored**; the checkpoint
  qualification was run explicitly above rather than inferred from that suite.
  Evidence: session task `t-mvznr165.log`. No QF-2 completion claim yet.
- Still required: GPU hot-cache/LRU lifecycle, CPU-hybrid workers, FP32 partial
  accumulation and double-buffered streaming, followed by QF-3 model/cache/state/
  graph integration and QF-4 Qwen4 MTP plus single-/dual-card end-to-end oracles
  and benchmarks. The host store is not yet wired into model execution.

---

## Native Hugging Face model acquisition

- CLI model-loading commands now accept Hub IDs with **download-on-first-use** via
  the pinned native Rust `hf-hub 1.0.0` client, including native Xet transfers.
  Metadata and selected Safetensors weights are pinned to one commit before
  loading; local paths remain local and no repository Python code is executed.
- `--revision`, independent `--draft-revision`, and `--offline` are wired through
  generate/bench/greedy-diff/dump-diff/serve/chat and nested live profiling.
  Saved-profile comparison never downloads. Original model IDs remain API/report
  names; acquisition and argument validation precede CUDA initialization.
- The released SDK owns HF cache/home/token/endpoint settings, saved login,
  credential opt-out, HTTP transport, redirects, retries, and Xet transfers.
  No duplicate credential lookup, custom plaintext-endpoint guard, injected HTTP
  client, or upstream Git dependency. This is an explicit scope decision: rely
  on the library rather than impose a second authentication/transport policy.
  Online startup checks Hub metadata and needs a writable cache. Offline startup
  uses the SDK's `local_files_only` snapshot lookup and then validates required
  files; it makes no network requests. Library/fixture lookup remains cache-only
  and uses the SDK's cache-root resolver.
- Required-file manifests live in repository `.w34/`, alongside the standard Hub
  layout. Missing indexed shards, skipped file 404s, unsupported architectures,
  denied/gated access, and unsafe shard paths fail clearly. Asset selection keeps
  tokenizer/templates, processor metadata, and decision heads, without fetching
  alternate weight formats. See [`MODELS.md`](MODELS.md) for the full contract.
- Live qualification: downloaded the **1,324,107,408-byte** native 4B DFlash2
  checkpoint into a fresh temporary Hub cache through Xet, then generated 32
  tokens with the existing 4B target. Snapshot:
  `nchapman/Qwen3.5-4B-DFlash2@8d8bb62873a4fe6165e644163e9ff76e06ab78ba`.
  Repeated after SDK simplification with another empty cache: online and offline
  32-token outputs matched byte-for-byte, including with an unreachable endpoint
  and unreadable token path offline. A pinned target cache hit transferred no
  payload bytes and generated successfully. Local evidence:
  `/tmp/w34-hub-sdk-{online,offline,pinned}.log` and matching output files.
- Verification: `mise run check` (full release CUDA suite, CPU/CUDA Clippy,
  frontend checks, oracle/perf checks), `mise run test-cpu`, and `mise run build`
  passed. Fifteen Hub tests include child-process tests of the real production
  resolver with saved login, explicit token, token-file override, credential
  opt-out, endpoint override, HOME-less explicit cache, XDG cache, and offline
  reuse. Review's suggested extra authentication/transport policies were declined
  per the explicit SDK-delegation decision; SDK limitations are documented.
  Strengthened the offline test to prove that a reachable Hub receives no requests.
  Final post-review `mise run check` and `mise run test-cpu` also passed.
- Qualification note: an earlier live test refreshed cached `main` from
  `d5874365...` to newer weights `db853835...`, causing a DFlash oracle mismatch.
  Restored only that test-created ref change; the oracle and full suite then
  passed. Both snapshots remain cached. Subsequent live checks pin revisions so
  qualification refs do not change.

---

## Production metrics and embedded dashboard

- `w34 serve` now serves a warm-dark **React + TypeScript + Vite** dashboard at
  `/`, with aggregate throughput, engine TTFT percentiles, running/queued requests,
  token-weighted prefix reuse, and separate KV/GDN active/evictable/free capacity.
  The read-only Connect page provides API and Prometheus setup information.
- `/metrics` (Prometheus HELP/TYPE + seconds histograms) and `/api/metrics`
  (`schema_version: 1`, no-store JSON) share coherent CPU telemetry. Common worker
  lifecycle accounting replaces API-specific token counters and covers enqueue,
  admission, prefill, committed output, completion, cancellation/backpressure,
  shutdown, and panic. Decision-only requests do not fabricate generated tokens or
  TTFT. Pool gauges count logical resources once across TP ranks; speculative
  acceptance uses actual verified draft rows, not configured proposal maxima.
- Dashboard history is bounded to five minutes in the browser, uses cumulative
  counter/histogram deltas, resets across process restarts, pauses in hidden tabs,
  and visibly marks disconnection. No external fonts/CDNs, prompt content, speed
  tests, or mutation controls. Optional API-key authentication is now available
  (see above); dashboard/metrics remain public, so keep the listener private or
  behind an authenticated reverse proxy.
- Node **24.21.0** and pnpm **12.10.1** are pinned in mise. Locked frontend builds
  run before CUDA build/check/test/coverage tasks; Vite's hashed assets are embedded
  in the executable. Docker builds assets in a separate pinned Node stage; no Node
  runtime is shipped. Direct CUDA Cargo commands require `mise run dashboard-build`.
  See [`METRICS.md`](METRICS.md) for metric units, timing boundaries, migration,
  security, PromQL, and development commands.
- Verification: `mise run check` (full release CUDA suite, both Clippy modes,
  oracle/perf checks, and frontend checks), `mise run test-cpu`, 18 frontend tests,
  production frontend Docker-stage build, Prometheus Python-parser validation,
  and Chrome desktop/mobile inspection against the embedded live server passed.
  The GDN occupancy regression first failed on unavailable occupancy, then passed
  with authoritative state-pool free-slot accounting. Review found a missing
  `dashboard-build` dependency on coverage tasks; fixed and task graph verified.
- Live 4B TP=1 HTTP/SSE smoke: all 96 measured requests passed (256 output tokens,
  8/16 clients; 24 warmup requests). With the dashboard open in both variants,
  adding ~20 Hz scrapes of **both** `/metrics` and `/api/metrics` produced 264
  successful scrapes, **0.53 ms p95 / 1.04 ms max**. Median aggregate throughput:
  N=8 **1395.50 → 1395.23 tok/s**, N=16 **2218.28 → 2215.94 tok/s** without/with
  extra scrapes. This is a short same-binary scrape-overhead check, **not** a
  pre-instrumentation or vLLM/SGLang comparison. Raw local evidence:
  `/tmp/w34-dashboard-load.json`; no new donor performance qualification claimed.

---

## Code-quality refactor and worker error containment

- Multi-architecture 30-series (`sm_86`) & 40-series (`sm_89`) GPU support
  (`crates/w34-kernels/build.rs`, `Dockerfile`, `csrc/w34_cuda.h`,
  `csrc/ops/op_init.cu`, `csrc/ops/op_marlin.cu`, `csrc/ops/op_collectives.cu`,
  `csrc/ops/op_gdn.cu`, `src/ffi.rs`, `src/cuda.rs`, `src/model/rank.rs`,
  `tests/phase4_dflash.rs`, `docs/HARDWARE.md`):
  - Default `w34-kernels` and Docker builds now emit a multi-architecture
    fatbinary covering both `sm_86` (all RTX 30-series / Ampere GA10x) and
    `sm_89` (all RTX 40-series / Ada AD10x) plus PTX forward-compatibility,
    while supporting single- or multi-arch `W34_ARCH` overrides.
  - Added `w34_device_props` / `CudaDeviceProps` + `CudaArchFamily` with
    lock-free per-device caching in `op_init.cu` and startup validation in
    `prepare_execution_plan` and `ModelRank::load`.
  - Replaced hardcoded device `0` `cudaDeviceGetAttribute` calls and `84`-SM
    constants in `op_marlin.cu` and `model/rank.rs` with cached active-device
    properties (`sm_count * 64` lock ints) and continuous L2-proportional
    wide-`N` `max_par` scaling (`clamp(l2_cache_bytes / 384 KiB, 8, 128)`).
  - Raised `w34_comm::Signal::kMaxBlocks` in `op_collectives.cu` from `84` to
    `144` (compile-time struct array ceiling covering full AD102) while clamping
    launched blocks via `tp2_block_cap` to the live minimum SM count across TP
    peers so `< 84`-SM GPUs never launch more spin-waiting CTAs than SMs.
  - Replaced hardcoded `nh == 32` / `nh == 24` branches in `op_gdn.cu`
    single-request recurrent prefill with direct wave-cost minimization
    (`ceil(nh * v_tiles / sm_count) * ROWS`) over `ROWS` in `{4, 7, 8, 11}`
    using the card's reported `sm_count`.
  - Updated `phase4_dflash.rs` `actual_sglang_full_model_hidden_parity` and
    `vision_linear_bias.rs` `linear_bias_patch_shape_strict_reduction` to
    enforce 0-ULP on the provenance capture GPU (`RTX 3090 Ti`) and strict
    floating-point bounds (`relL2 < 1e-4` / `max_abs < 0.005`) when `cuBLAS`
    executes on a different 30/40-series GPU; updated `prepare_execution_plan`
    to warm all 4 GDN `ROWS` templates (`{4, 7, 8, 11}`) via
    `sms / {3, 4, 5}` plus `_strided` GDN/norm entrypoints, and updated
    `tools/oracle/vision_norm_probe.py` + `build_vision_fa2_probe.sh` to build
    `sm_86 + sm_89` fatbinaries.
  - **Live 2× RTX 4090 (`sm_89`) & 2× RTX 3090 Ti (`sm_86`) Kernel Optimization & Oracle Cross-Check (`Passes 1–5`):**
    - **Pass 1 (`op_marlin.cu`)**: Enabled `enable_m_invariant_split_k` for `prob_n == 3072 && (prob_n / 64) * 2 <= sms && prob_k >= 2560` (switching `9B TP=2` `qkv` / `in_z` `N = 3072, K = 4096` from `48/128` active SMs to `96/128` SMs with 2-way Split-K on 128-SM 4090, lifting `9B TP=2` decode `219.6 -> 230.0 tok/s` and `MTP k=2` `383.6 -> 400.7 tok/s`), while keeping `prob_n == 2560 && prob_k <= 4096` on 1-slice `{128, 64, 128}` (`128/128 = 100.0%` exact `4B` golden token parity across 3090 Ti and 4090) and `prob_n == 2048 && prob_k == 4096` on 1-slice (`TP=1 == TP=2` router top-8 agreement on `35B-A3B`).
    - **Pass 2 (`op_collectives.cu`)**: Replaced thread-0-only peer flag polling in `barrier_epoch_release` with single-writer peer flag + multi-warp read, buffered `v_reg` in registers (`SHARED: 128B` down from `24.7 KiB`, `STACK:0 LOCAL:0`), launched `512` threads for `H >= 4096`, software-pipelined `tx_stage(stage + 1)` before `wait_stage(stage)`, and raised `tp2_block_cap` to `<= 16` CTAs, cutting `TP=2` collective latency by `-9%` to `-14%` (`5.75 -> 5.20 us` at `M=1, H=5120`; `24.48 -> 21.84 us` at `M=16, H=5120`) and reducing 4090 `nsys` per-step collective time on `35B TP=2` from `536.2 us` to `509.1 us`.
    - **Pass 3 (`op_moe.cu`)**: Added `moe_topk_router_256_k8_f16_kernel` (`8` warps × `32` lanes = `256` experts in registers with warp-shuffle top-8 reduction, `1.73 us` vs `vLLM`'s `16.2 us` 2-stage router), plus `__launch_bounds__(128, 8)` (`64` regs/thread -> `8` blocks/SM = `32` warps/SM occupancy, `STACK:0 LOCAL:0`) and bitshift `w >> group_shift` in `moe_w4a16_gate_up_silu_f16_kernel` and `moe_w4a16_down_accum_f16_kernel<false, 32|64, 8>`, cutting 4090 `nsys` per-step MoE time by **`-11.4%`** on `35B TP=1` (`970.4 -> 859.5 us/step`) and **`-21.5%`** on `35B TP=2` (`719.2 -> 564.3 us/step`), lifting `35B-A3B` decode to **`242.4 tok/s`** (`TP=1`), **`308.4 tok/s`** (`TP=2`), and **`507.1 tok/s`** (`TP=2 MTP k=2`) on 4090, and **`213.6 tok/s`** (`TP=1`), **`287.2 tok/s`** (`TP=2`), and **`466.4 tok/s`** (`TP=2 MTP k=2`) on 3090 Ti.
    - **Pass 4 & 5 (`op_paged_attn.cu`, `op_init.cu`, `model/rank.rs`)**: Raised `max_short_splits` to `8` for `total_tokens <= 4` in `op_paged_attn.cu` and added `w34_stream_set_persisting_l2_window` (`16 MiB` persisting L2 window with `missProp = cudaAccessPropertyNormal` for `scratch_arena` on GPUs with `persisting_l2_max_bytes >= 16 MiB`).
    - **Verification**: All 136 unit tests, 27 integration test binaries, and 7 doctests pass on both 2× RTX 4090 (`sm_89`) and 2× RTX 3090 Ti (`sm_86`); `cuobjdump --dump-resource-usage` confirms `STACK:0 LOCAL:0` on both `sm_86` and `sm_89` fatbinaries; `w34 dump-diff`, `w34 greedy-diff`, and `w34 profile-diff` pass across `4B`, `9B`, `27B`, and `35B-A3B`.
- Quantization and model-loading audit (`loader.rs`, `model/weights.rs`,
  `model/rank.rs`, `mtp.rs`, `dflash.rs`, `tests/quant_formats.rs`):
  - Fixed `marlin_permute_scales_k_by_n` and `RawLinearHost::upload` when
    `size_k == group_size` (e.g. `K = 128, group_size = 128`) so grouped Marlin
    layers (`group_blocks > 0`) always use the 64-element 8×8 `scale_perm`
    transpose while channelwise W8A16 layers (`group_blocks == -1`,
    `group_size == 0` or `group_size > 128`) use `scale_perm_single`.
  - Added AWQ `w_bit` / `q_group_size` key-alias support in `parse_quant_awq`
    and standalone `quant_config.json` / `quantization_config.json` discovery in
    `ModelConfig::from_dir` and `DFlash2Config::load`; tightened `auto-round`
    `extra_config` validation (`group_size`, `sym`, `data_type`, `act_bits` on
    quantized overrides while allowing `bits >= 16` float skip rules) and
    `vllm_backend` resolution.
  - Added `.weight_shape` (`read_weight_shape_2d` with strict byte-length checks)
    and `embed_tokens` bit-width/symmetry resolution for `compressed-tensors`
    checkpoints, precedence for unquantized 2D `F16`/`BF16`/`F32` `.weight`
    over experimental `.weight_packed` sidecars (fixing `--speculative mtp` on
    `nchapman/Qwen3.8-27B-W4A16-GPTQ`), per-expert and shared-expert dimension
    and `group_size` validation in `load_moe_layer_weights`, accurate
    column-sharded/replicated `estimate_weight_arena_bytes` accounting (skipping
    unused `.qzeros` and `.weight_shape`), and 4-bit-only `experts_are_w4` MTP
    arena sizing.
  - Verified with `mise run check`, `mise run test-cpu`, the opt-in
    `cached_hf_quantized_and_unquantized_checkpoints` qualification test across
    all 10 cached dense/MoE/MTP checkpoints, and `review`.
- Workspace Clippy now enables `cognitive_complexity` with threshold **25** in
  `clippy.toml`, enforced by the existing CPU/CUDA all-target checks with
  `-D warnings`. No function-length or nesting limits are enabled, and no
  cognitive-complexity suppressions were added. Split worker scheduling and CLI
  parsing/command dispatch into focused helpers, extracted forward/vision
  diagnostics, and decomposed test fixtures/assertions while retaining coverage.
  Added CLI missing-value and help short-circuit regression coverage.
  Verification for this follow-up: `mise run check` (including the full release
  CUDA suite, oracle/perf checks, and both Clippy configurations) and
  `mise run test-cpu` pass; the diff review found no actionable issues. Explicitly
  ignored qualification tests remain ignored; no new standalone performance or
  donor parity runs were performed for this structural refactor.
- Split the 10,433-line `src/model.rs` into a 176-line public facade and focused
  `src/model/` modules for weights, ranks, forward execution, generation, graph
  IO/capture/replay, sampling, decisions, and diagnostics. Existing `crate::model`
  paths remain available. Removed model-wide lint suppressions and documented
  local unsafe obligations; `cuda.rs` now documents its caller-owned lifetime and
  stream-ordering contracts rather than claiming those are enforced by Rust.
- Speculative verification and batched decode share coordinated upload-all →
  capture-all → launch-all replay. Partial TP launches remain fatal because a
  peer may be blocked in a collective. Single-token capture adopts returned graph
  handles before propagating errors. Capture cleanup handles `Result` errors,
  not panics; existing raw-address APIs still require caller discipline.
- Typed worker errors preserve OpenAI/Anthropic envelopes while distinguishing
  invalid input (400), overload (429), shutdown (503), and internal/CUDA/panic
  failures (500). Worker supervision catches Rust unwinding, poisons health,
  rejects future submissions, and drains active/queued requests. An independent
  terminal-event slot preserves finish/error delivery under token backpressure;
  request drop guards cover temporary admission/MTP ownership. Engine teardown
  occurs after supervisor notification, on the dedicated OS thread. This is not
  recovery from CUDA device faults or proof against in-flight collective hangs.
- Added runtime CUDA error descriptions, C-side sampler ABI assertions, and a
  unified biased-GEMM FFI declaration. CPU testing includes integration targets;
  converted optional fixture/GPU checks log local skips and fail under `CI=true`
  or `W34_STRICT_TESTS=1`. DFlash2 explicitly uses a window-aware parser while
  target `ModelConfig::from_dir` remains strict regardless of architecture labels.
- Verification: `mise run check` (`fmt-check`, `clippy`, `clippy-cpu`, full release CUDA test suite, `oracle-check`, `perf-check`), `mise run test-cpu`, and `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` all pass with 0 failures.
- Review follow-ups (pre-existing, outside this behavior-preserving refactor):
  validate RoPE rotary geometry before kernel dispatch; merge rather than replace
  pre-populated TP=2 diagnostic dumps; tighten DFlash root/body validation, vision
  geometry/input validation, multi-request tap validity, and incomplete scheduled
  batch reporting. Existing TP argument assertions and redundant loader checks
  also remain unchanged. CUDA diagnostic origin is still inferred from code and
  context, not represented by a distinct runtime/cuBLAS/host error-kind field.
  These are not newly established inference guarantees.

---

## Production-Hardened Quantization & Model-Loading Subsystem (sm_86 Native)

`warp-34` supports common Qwen 3.5 / 3.6 / 3.8 HuggingFace quantized and unquantized checkpoints suited to 1–2× RTX 3090 Ti (`sm_86`) without silent corruption or opaque OOMs:
- **Supported Quantization Formats & Geometries:**
  - **Unquantized:** Native `BF16` / `FP16` / `FP32` weights uploaded directly to the unified `FP16` model spine.
  - **Symmetric 4-bit (`U4B8`):** `group_size ∈ {32, 64, 128}` via `w34_marlin_gemm_f16u4` (`group_blocks ∈ {2, 4, 8}`).
  - **Asymmetric 4-bit (`U4` with packed zero-points):** `group_size ∈ {32, 64, 128}` via `w34_marlin_gemm_f16u4_zp` (`marlin::kU4`, `group_blocks ∈ {2, 4, 8}`), unpacking AutoAWQ `[K, N/8]` + `[K/g, N/8]` (`UNDO_AWQ = [0, 4, 1, 5, 2, 6, 3, 7]`) and Compressed-Tensors `[N, K/8]` + `[N/8, K/g]` into N-major words with `scale_perm` + interleave zero-point permutation.
  - **Dense Symmetric 8-bit (`W8A16` / `U8B128`):** `group_size ∈ {32, 64, 128}` (`group_blocks ∈ {2, 4, 8}`) and per-channel (`group_blocks = -1`, `group_size = K`) via `w34_marlin_gemm_f16u8` (`marlin::kU8B128`) and `w34_marlin_repack_int8` (`REG:40 STACK:0 LOCAL:0`).
  - **Asymmetric MoE W4A16:** `w34_moe_w4a16_mlp_zp_f16` across `group_size ∈ {32, 64, 128}` supporting both GEMV (`M < 4`) and Grouped-MMA (`M >= 4`) execution plans with chunked scratch budget management.
  - **Quantized Embedding Tokens:** Automatic host dequantization of `model.embed_tokens.weight_packed` or `.qweight` into FP16 for the embedding table.
- **Fail-Fast Loader & Capacity Diagnostics:**
  - `ModelConfig::parse_quant` validates quantization configurations strictly, falling back to standalone `quantize_config.json` when `config.json` lacks quant settings.
  - Immediate, descriptive errors with sanitized strings for: unsupported methods (`fp8`, `mxfp4`, `nvfp4`, `bitsandbytes`), activation quantization (`W8A8`), asymmetric 8-bit, dynamic actorder (`actorder != "static"`), desc_act with grouping, non-sequential `g_idx` permutations, AWQ `version != "gemm"`, 8-bit MoE, and quantized DFlash2 drafters.
  - Strict model-level fail-fast invariant enforcement: non-default `rope_scaling` / `rope_type`, `attention_bias: true`, `hidden_act != "silu"` / `"swiglu"`, `sliding_window > 0`, non-128 `linear_key_head_dim` (matching GDN kernel stride bounds), non-TP-divisible `moe_intermediate_size`, and non-256 `head_dim` under TQ4/Q8_0 KV quantization.
  - DFlash2 drafter fail-fast enforcement rejecting unhandled `output_multiplier != 1.0` and `final_logit_softcapping > 0`.
  - Archive-derived `weight_arena` calculation reserving exact sharded/replicated bytes plus aligned headroom, with pre-boot VRAM capacity enforcement (`min_boot_bytes > free_vram`).
  - Dynamically sized repack staging buffer (`(128 MiB).max(3 * max_packed_tensor_bytes)`) preventing unbudgeted runtime allocations during load.
- **Audit & Verification:**
  - All new custom kernels pass `cuobjdump --dump-resource-usage` audit with `STACK:0 LOCAL:0`.
  - Scalar-reference parity tests pass 0-ULP and tight L2 tolerances across `M ∈ {1, 4, 8, 16, 130}` for all dense formats (`g32/g64/g128/channel`, sym/asym) and MoE.
  - Full end-to-end load, prompt dump, and greedy token generation verified on 8 cached HuggingFace models: `Qwen/Qwen3.5-4B` (unquantized), `cyankiwi/Qwen3.5-9B-AWQ-4bit` (AWQ), `cyankiwi/Qwen3.8-27B-AWQ-INT4` (CT asym g32), `TheUnderscore/Swift-Qwen3.8-27b-W4A16-AWQ` (CT asym g128), `Lorbus/Qwen3.6-27B-int4-AutoRound` (AutoRound g128), `lued/Qwen3.8-27B-INT8-W8A16-DFlash2` (CT W8A16), `cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4` (MoE CT asym g32), and `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4` (MoE GPTQ-Int4).

---

## Generative Serving APIs (`/v1/responses`, `/v1/messages`, token-count endpoints) & Decision-Model Serving (`/v1/systemone`, `/v1/decisions`)

Full wire specifications are documented in [`docs/GENERATIVE_APIS.md`](GENERATIVE_APIS.md) and [`docs/DECISION_MODELS.md`](DECISION_MODELS.md):

- **OpenAI Responses API (`crates/w34-host/src/responses_api.rs`, `server_http.rs`)**:
  - `POST /v1/responses`, `GET /v1/responses/{response_id}`, `DELETE /v1/responses/{response_id}`, `POST /v1/responses/{response_id}/cancel`, `POST /v1/responses/input_tokens`, and `POST /responses/input_tokens` (plus `POST /v1/chat/completions/input_tokens` and `POST /chat/completions/input_tokens`).
  - Supports multi-turn `previous_response_id` chaining via an explicit opt-in (`"store": true`, requiring a non-empty `Authorization` or `x-api-key` credential header) bounded `ResponseStore` using 128-bit CSPRNG `resp_<32 hex>` identifiers with per-credential scope partitioning and per-scope eviction quotas; 0-indexed monotonic `sequence_number` SSE streaming (`response.reasoning_text.*`, `response.output_text.*`, `response.function_call_arguments.*`); flat & nested `function` tools (`tool_choice: "none" | "auto" | "required" | named`); `reasoning.effort` (`"none"` vs thinking); `text.format` (`json_schema` / `json_object` via `llguidance` bitmasks); and multimodal `input_image`.
- **Anthropic Messages API (`crates/w34-host/src/anthropic_api.rs`, `server_http.rs`)**:
  - `POST /v1/messages` (sync JSON + typed Anthropic SSE `message_start` → `content_block_start/delta/stop` for `thinking`, `text`, and `input_json_delta` `tool_use` → `message_delta` → `message_stop`) and `POST /v1/messages/count_tokens`.
  - Translates `thinking: {type: "enabled"|"adaptive"|"disabled", budget_tokens}`, `tool_choice: {type: "auto"|"any"|"tool"|"none"}` (preserving stateful `<tool_call>` ban bitmask on `"none"`), multi-turn `tool_use` / `tool_result` + base64 `image` blocks, `usage.input_tokens = prompt_tokens - cache_read_input_tokens`, and strips volatile `x-anthropic-billing-header: ... cch=<hex>;` hashes to preserve `MambaRadixCache` prefix hits.
- **Decision-Model Serving (`crates/w34-host/src/clef.rs`, `decision.rs`, `loader.rs`, `model.rs`, `engine_worker.rs`, `scheduler.rs`, `server_http.rs`)**:
  - `POST /v1/systemone` (`/systemone`) and `POST /v1/decisions` (`/decisions`) supporting **`ClefModel`** (`Cloudflare/clef-flash` 9B & `Cloudflare/clef` 27B: GPU-only single-pass joint multi-question `ClefJointHeadGpu` + `op_clef.cu` (`REG<=40 STACK:0`) streaming prefill chunks directly into `[seq_len, width]` `FP16` device memory + `FP32` span accumulators without `[seq_len, hidden_size]` host materialization, freeing host `f32` head weights immediately after `FP16` `DeviceArena` upload, cuBLAS `FP16` Tensor-Core GEMMs + 4-warp online-softmax cross/self-attention, on-demand `pread` of option-span lexical rows from `lm_head.weight`, and `ConfidenceKind::MaxProbability`), **`KevModel`** (`head.safetensors` `PointerHead` over post-final-`GemmaRMSNorm` `<|box_end|>` / `<|fim_suffix|>` hidden states), **`NimbleModel`** (`SafeJson`, `nimble_temperature`), **`Tev1`** (default Qwen3.5 generative scorer, 2–24 options `A..X`), **`OpenJev`** (`A..Z, a..z`, `noul` `true`-first), and **`XorModel` / `LevModel`** (forward+reverse option-order calibration + 9-level `noul` scale), plus **multimodal `images`** (`data:image/...;base64,...`, `{"url": ...}`, `{"media_type": ..., "data": ...}`, and `--allow-remote-images`-gated `http(s)://` URLs) across vision-capable checkpoints and archive-derived `DeviceArena` sizing for unquantized dense Qwen3.5 models.
  - Escapes ChatML/Qwen special-token delimiters inside untrusted `state`/question JSON (`\u003c` / `\u003e` and `<¦...¦>`) while preserving byte-for-byte donor prompt, token-ID, and span golden parity (`joint_schema_model.py`, `kev_head_goldens.inc`, `nimble_goldens.inc`, `tev1_goldens.inc`) and beating Cloudflare's in-process PyTorch CUDA `joint_schema_model.py` oracle end-to-end over HTTP on `Cloudflare/clef-flash` (`85.1 ms` `TP=2` / `109.8 ms` `TP=1` vs `185.0 ms` PyTorch at 428 text tokens = **`2.17×` / `1.68×` faster**; `86.2 ms` `TP=2` / `97.2 ms` `TP=1` vs `164.6 ms` PyTorch on 309-token multimodal PNG = **`1.91×` / `1.69×` faster**; `565.2 ms` `TP=2` / `669.0 ms` `TP=1` vs `1,024.0 ms` PyTorch at 3,254 tokens = **`1.81×` / `1.53×` faster**; `TP=1` vs `TP=2` `|Δprob| <= 0.0020`).
  - Wires `mamba_branching_seqlen` `C = 64`-aligned intermediate checkpoint donation across both `SchedulerV1` and multimodal `ScheduledVision::step_prefill` (with `TP=2` rank-0/rank-1 `KvPool` & `StateBundlePool` lockstep clamping) so Query 0 donates the exact shared `[system + images + state]` prefix into `MambaRadixCache` and subsequent decision sub-queries reuse cached KV + GDN `StateBundle` checkpoints without re-encoding cached images in zero-decode (`max_new_tokens == 0`) waves.
  - Verified by CPU unit tests (`clef::tests::*`, `decision::tests::*`, `responses_api::tests::*`, `anthropic_api::tests::*`) and live GPU HTTP+SSE integration suites `crates/w34-host/tests/generative_apis.rs` and `crates/w34-host/tests/decision_models.rs`.

---

## Long-context optimization round qualified — beats `vLLM 0.31.0` & `SGLang v0.5.19` across all 9B TP=1 and 27B TP=2 cases, retains 1-GPU 27B 262K HTTP & needle

All six candidate suites completed serially with `exit=0` on a single frozen release binary (`crates_tree_sha256=e1b67ef230877a0facb487401ab037c3ba15660c5e969444b782bfadfc6350e0`, `w34` `binary_sha256=b125f589aa1f567420f6ce11e0a713db1e4dd1e202ca3df45bb6f648f12fcb4b`, `needle_gate` `d347dde3399f0e73827ab6f88361c424564b667b5af08fdd5cd1a87963bab6c3`, recorded in `benchmarks/perf/results/goal-provenance.json` and indexed in `benchmarks/perf/campaigns/longctx-coding-agent.results.json`; report in `benchmarks/perf/campaigns/longctx-coding-agent.md`):

- **Shipped changes (all custom kernels `STACK:0`, FP16 spine + FP32 SSM, strict 256-token mixed decode/prefill cap preserved)**:
  - **`op_gdn.cu`**: padded bank-conflict-free WY Stage 1 (`kWyStage1SmemBytes=36608`, 116 regs, `STACK:0`), Stage 2 dedicated `s_wh_full` with register-prefetched `U`/`K` and `__half2` `s_H` stores (`kWyStage2SmemBytes=43520`, 128 regs, `STACK:0`, 2 fewer barriers/chunk), paired `__half2` `gdn_causal_conv1d_pair_f16_kernel` (`REG:40 STACK:0`, 4B alignment guard), and exact `grid_upper_chunks` checkpoint `H`-slot sizing.
  - **`op_sampler.cu`**: two-stage `w34_sampler_step_fast_greedy_masked_f16` (`REG:40/31 STACK:0`) replacing the 174 µs scalar masked sampler under stateful `tool_choice: "none"` bitmasks (0-ULP verified).
  - **Attention, scratch & scheduler**: packed multi-request FP16 FA2 (`64x64`, retaining donor `STACK:128`), Q8_0-only 128-row prefill dispatch (`REG:234 STACK:0`), pre-scaled FP16 TQ4 LUTs, overlaid FA/GDN/FFN scratch with model-derived prefill chunks up to 2048 tokens, Marlin prefill Split-K, and fair 64-token-tile pure-prefill budget sharing across concurrent prefills while keeping mixed decode/prefill capped at 256 tokens.
- **9B TP=1 (`goal-9b-fp16` 78/78, `goal-9b-tq4` 78/78)**:
  - **FP16**: 59K C1 **`27.02 tok/s`** (vs `vLLM 25.98`, `SGLang 26.14`), 59K C2 **`28.91 tok/s`** (vs `vLLM 28.74`, `SGLang 28.81`), 118K C1 **`14.51 tok/s`** (vs `vLLM 13.75`, `SGLang 13.62`), cold 57.6K / 115.1K TTFT **`14.61 / 36.28 s`** (vs `vLLM 14.86 / 36.39 s`, `SGLang 15.26 / 37.59 s`) — ahead of both competitors on all 5 cases.
  - **TQ4**: **`22.07 / 20.79 / 10.74 tok/s`**, cold 57.6K / 115.1K **`18.15 / 50.07 s`** (`+18.3%` C2 throughput and `-11.3%` 115K cold TTFT vs original TQ4 baseline).
- **27B TP=2 (`goal-27b-tp2-fp16` 42/42, `goal-27b-tp2-tq4` 42/42)**:
  - **FP16**: 59K C1 **`13.60 tok/s`** (vs `vLLM 11.97`, `SGLang 12.00`), 118K C1 **`6.96 tok/s`** (vs `vLLM 6.40`, `SGLang 6.44`), warm-turn p50 TTFT **`837 / 1,012 ms`** (vs `vLLM 1,231 / 1,667 ms`, `SGLang 1,339 / 1,307 ms`), cold 57.6K / 115.1K TTFT **`33.96 / 78.79 s`** (vs `vLLM 35.68 / 83.15 s`, `SGLang 36.58 / 85.07 s`) — ahead of both competitors on all 4 cases.
  - **TQ4**: **`11.97 / 5.62 tok/s`**, warm-turn p50 **`1,013 / 1,272 ms`**, cold **`39.18 / 98.11 s`**.
- **Single-card 27B TP=1 262K retention (`goal-27b-tp1-262k` 29/29 + `goal-needle`)**:
  - Default-`tq4` live HTTP served **29/29 requests** up to **252,112 prompt tokens** (82.7% prefix hits): ~59K / ~118K / ~252K **`7.77 / 3.63 / 1.12 tok/s`** (warm-turn p50 **`1,597 / 1,922 / 3,154 ms`**), 115K cold read **`149.48 s`**.
  - 262,144-window 50%-depth needle **passed**: `262,108` prompt tokens, exact `"73918462"`, TTFT **`584.14 s`**, decode **`23.36 tok/s`**.
- **Verification & review**: `CUDA_VISIBLE_DEVICES=0,1 mise run check` passed (`exit=0`, **234 Rust tests/doctests**, **0 failed**, **69 oracle tests**, **51 perf tests**); code review (`.pi/review/call_49568.md`) reported **0 critical/important findings**.

## Long-context coding-agent campaign complete — 538/538 HTTP requests, both single-card 262K KV policies qualified

See [campaign measurements and limitations](../benchmarks/perf/campaigns/longctx-coding-agent.md)
and [Open-SWE fixture attribution](../benchmarks/perf/fixtures/OPEN_SWE.md).
The frozen `27b-fp16-first.toml` is unchanged.

- Added 64-query-row paged-attention prefill, now including request-local tiling
  for mixed/concurrent batches. Single-/multi-request FP16/Q8_0/TQ4 comparisons
  pass 0-ULP parity; `REG:238 STACK:0 SHARED:43353 LOCAL:0`.
- 27B TP=1 default-TQ4 **262,144-window needle passed** at 50% depth:
  262,108 prompt tokens, exact `73918462`; final-code repeat TTFT **629.60 s**,
  decode **22.54 tok/s** (earlier pre-multi-request-extension run: **592.98 s**,
  **23.48 tok/s**). Runs overlapped work on the other GPU, not a controlled
  cross-run timing comparison.
  Extreme-context scratch is now 336 MiB (1024-token chunk tier), rather than the
  earlier 128 MiB configuration. Default-TQ4 HTTP (no KV-format override) now
  passed **29/29 requests**, reaching **252,112 prompt tokens**; 82.7% prefix hits,
  **3.99 s** warm-turn p50 TTFT, last turn reused **250,688 tokens**. TQ4-all passed the
  262,144-window needle: exact `73918462`, **542.18 s** TTFT, **23.88 tok/s** decode.
  That run used GPU 1 while default-TQ4 HTTP replay ran on GPU 0. TQ4-all HTTP
  also passed **29/29 requests**, reaching **252,112 tokens**, 82.7% prefix hits,
  and **3.60 s** warm-turn p50 TTFT. Both policies now qualify live HTTP serving.
- 9B TP=1 three-engine comparison completed (78/78 requests per deployment).
  At 59K C2, multi-request tiling improved warp-34 FP16 **13.43 → 21.53 tok/s**
  and TQ4 **8.41 → 17.58 tok/s**, but competitors remain ahead (~28.8 tok/s).
  At 115K cold prefill, warp-34 FP16 **40.07 s** vs vLLM **36.39 s** and
  SGLang **37.59 s**: do not claim a broad long-context speed win.
- 27B TP=2 all four deployments completed (42/42 each): 59K agent FP16
  **12.68 vs 11.97 tok/s**, warm TTFT **901 vs 1231 ms**; 118K throughput
  **6.38 vs 6.40 tok/s**; cold 115K **86.41 vs 83.15 s**. SGLang measured
  **12.00 / 6.44 tok/s**, cold 115K **85.07 s**. Cache details remain unknown for
  some SGLang requests despite enabling reporting; no missing counts are imputed.
- Direct cold-prefill A/B: 27B TQ4 32K TTFT **35.13 → 28.90 s** (+21.6% prompt
  throughput), attention kernel time **12.35 → 6.16 s**. 9B TQ4 64K TTFT
  **34.48 → 23.33 s** (+47.8% prompt throughput) against a clean `a77ce42` build.
  Both pairs generate identical token IDs. Compact metrics and raw-file hashes
  are retained in `benchmarks/perf/campaigns/longctx-coding-agent.results.json`.
- Optional tool-only follow-up turns preserve Qwen's last-user boundary and
  `tool_choice: "none"`. Added prefix-hit, warm/cold TTFT and per-step reporting;
  **51 Python tests pass**. Actual prompt lengths are ~59K/118K/252K, not exact
  capacity limits; fixtures combine multiple repositories and are not SWE task
  correctness evaluations. Older 9B competitor runs lack cache-detail flags;
  updated launch profiles enable those flags without inventing missing usage.
- Initial review's important finding was a non-discriminating scheduler pinning
  regression. Replaced with a three-slot/two-eviction case: fails with pinning
  disabled (stale 64-token hit), passes with pinning restored. `check-quick`
  (fmt and clippy, both feature states) passes. Follow-up review has only
  nonblocking suggestions (triage in campaign report); CPU Rust and 69 oracle
  tests pass. All ten serving deployments completed **538/538 requests**; both
  final-code 262K needle policies pass. Final **`CUDA_VISIBLE_DEVICES=0,1 mise run
  check` passed** (288.44 s): fmt, clippy with/without CUDA, **216 Rust tests**
  including doctests (0 failed/ignored), **69 oracle tests**, **51 perf tests**.
  Full log SHA-256 and measurement hashes are in the campaign results JSON.
  Performance scope is explicit: direct prefill speedups and a 27B 59K agent win,
  not a general defeat of vLLM/SGLang on cold reads or 9B C2 workloads.

## Post-Phase-6 `sm_86` 6-item performance roadmap complete — `35B-A3B` `1,009.7 tok/s`, batched `B=2..4` `MTP`, `128K` `TQ4`, large-`M` prefill, `TP=2` pipelined collectives, and unified prefill+decode

Completed all 6 items of the `sm_86` performance roadmap across `w34-kernels` (`STACK:0` verified via `cuobjdump --dump-resource-usage` on every touched kernel) and `w34-host`, preserving the `FP16` target spine (`BF16` `DFlash2` drafter, `FP32` `ssm_state`), 0-ULP `M`-invariance / continuation parity, and the frozen `benchmarks/perf/campaigns/27b-fp16-first.toml` baseline:

1. **`35B-A3B` Hybrid MoE (`TP=2` FP16 & `TP=1` `TQ4`) `w34perf` campaign ([`35b-a3b-moe-serving.toml`](../benchmarks/perf/campaigns/35b-a3b-moe-serving.toml)) & `op_moe.cu` / Marlin optimization:**
   - Enabled 0-ULP `M`-invariant 2..4-slice Split-K in `op_marlin.cu` for narrow `35B-A3B` projections (`prob_m <= 16`, `prob_n in [512, 2048]`, `prob_k in [1024, 2048]`) and routed `op_moe.cu` `regime == 3` (`M > 128`) through chunked `BLOCK_M = 8` `moe_sparse_mma_kernel` (`M = 256` MoE layer latency `1.0427 ms` → **`0.6405 ms`**, **`1.63×`**, `STACK:0`, `tests/moe_sparse.rs` 3/3 green).
   - **Live HTTP `w34perf` results vs `vLLM 0.31.0` (`TP=2` FP16) on 2× RTX 3090 Ti (100% requests ok):**
     - `decode-c1`: `warp-34 TP=2 FP16` **`250.65 tok/s`** (TTFT p50 `37.71 ms`, E2E p95 `511.36 ms`) vs `vLLM 0.31.0` `191.42 tok/s` (**`+30.9%`**); `warp-34 TP=2 + MTP k=2` **`319.60 tok/s`** (**`+67.0%`** over `vLLM`, `457.41 tok/s` native CLI); **`warp-34 TP=1 TQ4` (1× 24 GiB 3090 Ti!) `186.50 tok/s`**.
     - `spec-c2` / `spec-c4`: `warp-34 TP=2 FP16` **`414.56` / `641.87 tok/s`** vs `vLLM` `316.70` / `546.90 tok/s` (**`+30.9%` / `+17.4%`**); `warp-34 TP=1 TQ4` **`303.13` / `473.55 tok/s`**.
     - `throughput-c8`: `warp-34 TP=2 FP16` **`1,009.72 tok/s`** (`[1008.05, 1011.39]`, E2E p95 `1,012.80 ms`) vs `vLLM` `856.95 tok/s` (**`+17.8%`**); `warp-34 TP=1 TQ4` **`710.18 tok/s`** on a single card.
     - `mixed-poisson` (open-loop mixed prompt lengths): `warp-34 TP=2 FP16` / `MTP k=2` E2E p95 **`606.23 ms` / `560.08 ms`** vs `vLLM` `786.03 ms` (**`-22.9%` / `-28.7%`** tail latency).

2. **Speculative serving (`Qwen3_5MTP` & `DFlash2`) `w34perf` campaign ([`27b-spec-and-tp2.toml`](../benchmarks/perf/campaigns/27b-spec-and-tp2.toml)) & batched `B=2..4` `MTP` verification (`mtp.rs`, `proposer.rs`, `engine_worker.rs`, `model.rs`):**
   - Extended `MtpEngine` (`MAX_MTP_BATCH_SLOTS = 4`), `Proposer::step_scheduled_batched_mtp_verify_round`, `Qwen3_5Engine::verify_batched_speculative_gpu_steps` (supporting greedy soft-bans / `ignore_eos`), and `engine_worker.rs` to verify `B = 2..4` concurrent sequences per target forward pass (`k_cap = 1` at `B >= 3` to stay inside `M <= 8` Split-K/MoE regimes). Verified bit-exact `B=4` vs `B=1` lossless parity under `TQ4` in `tests/phase6_speculative_handoff.rs` (`batched_mtp_b4_matches_single_stream_losslessly`).
   - **Live HTTP `27B` (`0df81ead...`) `TP=2` speculative serving results:** `decode-c1` jumps from `81.85 tok/s` (plain `TP=2`) to **`127.26 tok/s` (`+55.5%`)** with `Qwen3_5MTP k=2` and **`152.83 tok/s` (`1.87×`)** with `DFlash2` (`M=8` `BF16` drafter); at `spec-c2` (`C=2`), batched `MTP` reaches **`172.37 tok/s`** (`+13.5%` over plain `TP=2` `151.88 tok/s`, TTFT p50 **`144.42 ms` vs `195.08 ms`**, `-26.0%`), and `501.22 tok/s` at 256-token `B=2` steady-state on `35B-A3B`.

3. **Paged `TQ4` (`3.07×` / `3.88×`) capacity & `8K–128K` / `C8–C16` serving ([`27b-tq4-longctx.toml`](../benchmarks/perf/campaigns/27b-tq4-longctx.toml)) + `op_paged_attn.cu` tuning:**
   - Vectorized `TQ4` WHT-256 shared-memory stores (`uint2`), raised `TQ4` Split-KV cap to `64` splits above `32K` (`STACK:0` on both `paged_attention_gqa_mma_f16_kernel` and `paged_attention_split_kv_reduce_f16_kernel`), packed non-power-of-2 GQA prefill (`gqa_ratio = 6` → `q_group = 2, tok_tile = 8`, 16/16 MMA rows, cutting `27B TP=1 TQ4` `64K` prefill from `116.8 s` → **`96.1 s`**, **`+21.5%`**), and cleared stale `tokenizer.json` `truncation.max_length = 4096` on load (`normalize_hf_tokenizer`) + tiered `SchedulerConfig::clamp_for_scratch_bytes`.
   - **1-GPU `27B` (`0df`) `8K–128K` & `C8–C16` results:** Where `FP16` (`65,536 B/tok`, `8.0 GiB` KV) OOMs `8,192` pages (`128K`) on 1× 24 GiB 3090 Ti, `TQ4` (`21,376 B/tok`, `3.07×`) sustains **`47.94 / 43.56 / 42.46 / 40.56 / 33.59 tok/s`** at **`8K / 16K / 32K / 64K / 128K`** (`129,024` tokens) on `TP=1` (**`47.44 tok/s`** at **`128K`** on `TP=2`), and `TQ4_all` (`16,896 B/tok`, `3.88×`) sustains **`41.07 tok/s`** at `64K` (`8,192` pages resident). `nsys` `16K` `TQ4` decode: `paged_attention_gqa_mma_f16_kernel` = **`268.32 µs/layer`** (`4.29 ms/tok`, `80.4%` effective `sm_86` HBM+WHT+MMA efficiency). Live HTTP `C=8` / `C=16` (`2,048` pages = `32K` pool on 1× GPU): **`266.61` / `258.55 tok/s`** (`tq4`) and **`264.27` / `256.29 tok/s`** (`tq4_all`), 32/32 requests ok.

4. **Large-`M` prefill compute on `sm_86` (`op_marlin.cu`, `op_gdn.cu`):**
   - Promoted wide-`N` (`N % 256 == 0`) large-`M` Marlin dispatch to `{64, 256, 256}` whenever `(prob_n / 256) * m_tiles >= sms` (2× `A`-tile reuse across 8 warps; `W4A8` INT8 IMMA audited and rejected for unweighted `g128` GPTQ due to activation-outlier precision loss without `FP16` spine changes) and added `STACK:0` `ROWS = 8` (`256` threads) `TP=2` GDN prefill (`gdn_prefill_f16_kernel` `~54 ms` → **`32.5 ms`**).
   - **`27B` (`0df`) `M = 1,331` & `2,048` prefill latency:** `TP=1` `M=1,331` TTFT **`846 ms`** (`1,573 tok/s`, matching `vLLM`'s `850 ms` without breaking 0-ULP `FP16` parity); `TP=2` `M=1,331` TTFT `795.48 ms` → **`726.41 ms`** (**`1,832.3 tok/s`**), `M=2,048` **`1,091.31 ms`** (**`1,876.6 tok/s`**).

5. **2-GPU (`TP=2`) PCIe Gen4 x8 serving ([`27b-spec-and-tp2.toml`](../benchmarks/perf/campaigns/27b-spec-and-tp2.toml)) & odd-stage pipelined sysmem `AllReduce` (`op_collectives.cu`):**
   - Vectorized `tp2_all_reduce_residual_gemma_rms_norm_f16_kernel` staging (`uint4`), pipelined large payloads across odd stage counts (`1`, `3`, `5`) to overlap Rank-0/Rank-1 PCIe Gen4 x8 TX and RX while preserving `flag[blockIdx.x]` parity, and hardened `barrier_epoch_release` with a wrap-safe monotonic signed delta check (`< 0`, `STACK:0`).
   - **`27B` `TP=2` vs `2× TP=1` data-parallel replicas across PCIe Gen4 x8 (`topo PHB`):** `TP=2` beats `2× TP=1` across every regime — `decode-c1` **`81.85` vs `50.17 tok/s` (`+63.1%`**, or **`152.83 tok/s` `+204.6%`** with `DFlash2`), `spec-c2` **`151.88` vs `100.39 tok/s` (`+51.3%`**, **`172.37` `+71.7%`** with batched `MTP`), `throughput-c8` **`410.59` vs `332.96 tok/s` (`+23.3%`**, E2E p95 **`2,492.27 ms` vs `3,088.32 ms`**), and `prefill-3k-c1` TTFT p50 **`1,648.94 ms` vs `2,232.58 ms` (`-26.1%`)**.

6. **Unified prefill + decode continuous batching (`scheduler.rs`, `engine_worker.rs`):**
   - Added `mixed_chunk_prefill_size = 256` (`SchedulerConfig::effective_mixed_prefill_budget`) so active decode tokens (`1` token/seq) co-execute in a single unified `ForwardBatch` alongside bounded `256`-token prefill chunks instead of stalling behind `1,408`-token pure-prefill waves.
   - Verified by `scheduler::tests::test_unified_prefill_decode_bounds_mixed_step_tokens` and live open-loop mid-decode `~1,050`-token prompt injection on `35B-A3B TP=2` (active decode ITL **`p50 = 3.72 ms, p95 = 48.98 ms, max = 50.37 ms`** across 4 chunks vs `>160 ms` unchunked stall) plus `mixed-poisson` `w34perf` E2E p95 **`560.08–606.23 ms` vs `vLLM` `786.03 ms`**.

---

## Phase 6 closeout checkpoint — concurrent `w34 serve` donor win, ratified `6.1` rooflines, 6/6 VL, and 8/8 needle gates

### 1-GPU 27B FP16 (`0df`) kernel, scheduler, and serving sweep — 4/4 win over `vLLM 0.31.0` and `SGLang v0.5.19` (2026-10-30)

- Optimized `TP=1` 27B FP16 (`0df81ead...`, frozen `benchmarks/perf/campaigns/27b-fp16-first.toml` on GPU 0, `kv_pages = 512`, `tool_choice = "none"` grammar intact) across kernels, scheduler, and serving without altering precision (`FP16` spine, `FP32` `ssm_state`) or 0-ULP `M`-invariance / continuation parity:
  - **Norms, small-N GEMM, Marlin, Attention, and GDN (`w34-kernels`):** `vec8` `uint4` `GemmaRMSNorm` + `SiLU*Mul` up to `hidden_size = 6144`; 3-tier 256-thread 0-ULP `M`-invariant `small_n_gemm_f16` / `small_n_gemm_dual_f16` (`<1,4>` at `M=1`, `<4,4>` at `M<=16`, low-register `4x4` at `M>16`); Marlin `M_BLOCKS = 3` (`M=33..48`), 1-slice `80`-SM `cfg128 = {128, 64, 128}` for `prob_m_split <= 64` wide-`N` (`N=5120`), and `max_par = 64` (eliminating extra `14.5 GB` weight passes at `M=1331`); `short_split` + `MIN_PAGES_PER_SPLIT = 4` GQA-MMA paged attention; token-tiled `gdn_causal_conv1d_f16`, write-back `<8,6>` (`nh>=96`) / `<4,12>` GDN decode, and bank-conflict-free SoA `TILE_S = 16` `<8,5>` (`nh>=96`) / `<4,5>` GDN prefill. All new specializations are warmed in `prepare_execution_plan()` with `STACK:0` verified via `cuobjdump --dump-resource-usage`.
  - **Scheduler & serving (`w34-host`):** `SchedulerConfig` `max_num_batched_tokens = 1408` / `chunk_prefill_size = 1408` with scratch-arena safety clamp; wave-aware `prev_active_peak` idle burst coalescing (`3.5 ms` quiet cap); deferred `llguidance` post-launch bitmask overlap + `96 KiB` pinned sampler DMA; checkpoint-aware `detect_default_preserve_thinking` (handling both `chat_template.jinja` and string/list `tokenizer_config.json`).
- **Final frozen-campaign comparison (`benchmarks/perf/results/27b-fp16-final-20261030-215439`, 96/96 requests ok, binary SHA-256 `1cf2e2dc...`) vs [first 3-engine baseline](../benchmarks/perf/campaigns/27b-fp16-first.md) (`warp-34` trial median / 3-rep aggregate vs `vLLM 0.31.0` / `SGLang v0.5.19`):**
  - `decode-c1`: **51.76 / 51.73 tok/s** (TTFT p50 **162.83 ms**, TPOT **18.74 ms**, E2E p50 **4943.90 ms**) vs baseline `45.47` (`vLLM` `50.75`, `SGLang` `51.70`).
  - `throughput-c8`: **310.17 / 309.25 tok/s** (TTFT p50 **981.68 ms**, TPOT **22.01 ms**, E2E p95 **6599.77 ms**) vs baseline `287.02` (`vLLM` `309.14`, E2E p95 `6708.87 ms`; `SGLang` `226.58`, E2E p95 `9206.99 ms`).
  - `agent-c1`: **48.55 / 48.65 tok/s** (TTFT p50 **128.38 ms**, TPOT **19.06 ms**, E2E p95 **2661.70 ms**) vs baseline `39.35` (`vLLM` `41.10`, `SGLang` `45.72`).
  - `prefill-primed-c1`: **51.95 / 51.95 tok/s** (TTFT p50 **31.31 ms**, TPOT **19.15 ms**, E2E p95 **2469.54 ms**) vs baseline `44.69` (`vLLM` `42.91`, `SGLang` `50.64`).
- Full post-review `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **268.06 s** (`fmt-check`, `clippy` with/without CUDA, 69 oracle tests, 47 `w34perf` tests, and all Rust unit/doc/GPU integration gates across `TP=1` and `TP=2`), and `w34 serve` is stopped with both GPUs clear.

### Next performance roadmap (prioritized by `sm_86` leverage)

1. **`35B-A3B` Hybrid MoE (`TP=2` FP16 & `TP=1` `TQ4`) 3-engine `w34perf` campaign & `op_moe.cu` sweep:** Establish matched `C1` / `C8` / `agent-c1` / `prefill-primed-c1` numbers against `vLLM 0.31.0` and `SGLang v0.5.19` (`~3B` active params/token → `~8×` higher theoretical decode ceiling than dense `27B`), capture `nsys` traces across `M = 1, 8, 166..1408`, and tune sparse-GEMV → grouped-MMA crossover (`total_pairs` vs `num_experts`), router/scatter overhead, and `TP=2` MoE `AllReduce`.
2. **Speculative serving (`Qwen3_5MTP` & `DFlash2`) `w34perf` campaign + `B=2..4` batched `MTP` verify:** Benchmark live HTTP speculative `decode-c1` and `agent-c1` (with `llguidance` grammar + `MambaRadixCache`) against `vLLM` `MTP` and `SGLang` `MTP`/`DFlash2` on `4B`, `27B`, and `35B-A3B`, and extend `engine_worker.rs` from single-request `speculative_id` (`B=1`) to `B=2..4` batched `MTP` verify so light concurrency keeps `1.7×–2.5×` speculative acceleration.
3. **`TQ4` (`3.07×` / `3.88×`) KV capacity & long-context (`8K–128K`, `C8–C16`) campaign + `fattn_mma_tq` tuning:** Quantify `warp-34`'s 4-bit WHT-rotated KV capacity/throughput advantage on 24 GB `sm_86` (no native Ampere FP8 KV) where `FP16` KV OOMs or queues (`SGLang` cached `C8` OOM on 27B), and profile/tune `fattn_vec_tq` + `fattn_mma_tq` Split-KV partitioning at `32K–128K`.
4. **Large-`M` prefill compute on GeForce `sm_86` (`W4A8` INT8 IMMA + chunk-WY GDN prefill):** Close the remaining `~132 ms` `M=1,331` TTFT gap vs `vLLM` (`982 ms` vs `850 ms`) and accelerate `8K–64K` prefills by adding dynamic per-token INT8 activation quantization (`W4A8` Marlin on `320 TOPS` INT8 IMMA vs `80 TF` half-rate GeForce F16-FP32acc for `M >= 128`) and a 64-token chunk-parallel WY `mma.sync` GDN prefill kernel (`seq_len >= 256`).
5. **2-GPU (`TP=2`) PCIe Gen4 x8 serving campaign & prefill collective overlap:** Run 2-card `27B` and `35B-A3B` `w34perf` comparisons (`TP=2` low-latency vs `2× TP=1` data-parallel throughput), and pipeline tiled pinned-sysmem `AllReduce` behind `o_proj` / `down_proj` / `out_proj` to hide the `640–1,800 MiB`/step PCIe Gen4 x8 transfer tax during large-`M` `TP=2` prefills.
6. **Unified prefill + decode continuous batching (`SchedulerV1` / `ForwardBatch`):** Pack active `M=1..16` decode tokens alongside bounded prefill chunks (`256–512` tokens) in a single forward pass so weights are read once and p99 decode ITL stays smooth under open-loop Poisson arrivals and mixed prompt lengths.

### First fresh three-engine 27B FP16 comparison (2026-10-30)

- [Results and exact campaign](../benchmarks/perf/campaigns/27b-fp16-first.md):
  newer GPTQ 0df, one RTX 3090 Ti/GPU 0, FP16, no speculation, serial engines,
  three repetitions, 256-token decode and 128-token agent/prefill ceilings.
  **288/288 measured requests pass: 96 per engine.** All servers stopped.
- Aggregate output tok/s (warp-34 / vLLM / SGLang): C1 **45.5 / 50.7 / 51.7**;
  C8 **287.0 / 309.1 / 226.6**; agent **39.4 / 41.1 / 45.7**; primed ~3k
  prefill **44.7 / 42.9 / 50.6**. Warp trails the best decode throughput in this
  run; its clear win is warm-prefix first output: **35.0 / 492.9 / 81.1 ms**.
- SGLang cached eight-active mode OOMed on a 1008 MiB recurrent-state copy.
  Retained that failed attempt; final unique C1/C8 uses cache-off/eight active,
  agent/prefill uses cache-on/two active. Both final modes pass every request.
- Raw bundles, preflight/startup evidence, execution/summary scripts and one-second
  GPU samples live in `benchmarks/perf/results/27b-fp16-first-2026-10-30/`.
  Checked-in JSON/CSV/Markdown retain metrics and raw-evidence hashes. All four
  bundle checksum indexes, counts and preflight receipts verified. Report review
  fixed missing-accounting and single-burst TPOT handling using suite helpers.
- `mise run perf-check` passes (47 suite tests); campaign validates, documentation
  links and `git diff --check` pass. No engine or benchmark implementation changed.

### SGLang FP16 needs explicit convolution-cache dtype (2026-10-30)

- Fresh offline FP16 repo-ID retry after retirement explicitly resolves 0df,
  loads newer weights, and reproduces the same GDN BF16/FP16 compile mismatch.
  Cleanup fixed default artifact selection, not that independent dtype issue.
- Exact-source investigation found the missing **supported runtime setting**:
  `SGLANG_MAMBA_CONV_DTYPE=float16` inside Docker, alongside `--dtype float16`
  and `--mamba-ssm-dtype float32`. SGLang otherwise allocates BF16 convolution
  history independently of the activation dtype. No engine or model-byte edits.
- Added an explicit FP16-convolution profile to the one-card preset, retaining
  BF16 and all historical failures. Prefill/decode graphs, HTTP startup, full-byte
  preflight and **48/48 measured requests / 12 trials** pass: original C1/C8/agent
  battery plus six longer primed-prefill requests (~3,020 prompt tokens over
  chunk size 2,048). Logs show chunking and positive cached tokens, max 3,008;
  HTTP cache details are null, not per-request cache-hit evidence. No independent
  quality/equivalence/saturation qualification is claimed. Server stopped.
- Evidence: `benchmarks/perf/qualification/27b-one-card.json`; new raw bundle
  `/tmp/w34-perf-sglang27-q6-fp16-conv16/`. Preset regression confirms the explicit
  dtype metadata/env setting. Review fixed the future recipe's mutable repo-ID
  binding to an exact 0df snapshot path; the executed trial/config hash remain
  unchanged. 6/12 agent replies emit raw tool markup despite `tool_choice=none`
  (same count in BF16), with no structured API calls: HTTP/accounting success is
  not replay-policy or task-quality success. Native-16 smoke throughputs are
  roughly unchanged (C1 ~48 tok/s; offered C8 ~77 BF16 / ~78 FP16); no general
  speedup claim.
- Full post-review `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **209.98 s**:
  208 Rust tests/doc-tests, 69 oracle tests, 47 performance tests, fmt and both
  Clippy feature states. Log `/tmp/w34-sglang-conv16-pinned-final-check.log`.
  Evidence checksums/counts, pinned launch/inventory-root binding, JSON/doc links
  and `git diff --check` also pass.

### Obsolete 27B GPTQ cache retired (2026-10-30)

- **Only GPTQ `0df81eadcfd59601cb82ccece2f16ea7d39244bf` is approved for current
  runs.** Found the local HF `refs/main` still pointing to obsolete c11; changed
  it to 0df and removed c11 via revision-aware `hf cache rm`, freeing **18.6 GB**.
  Old snapshot/weight blob are gone; shared files and unrelated experiments/
  downloads were preserved. Historical result/fixture provenance is unchanged.
- Re-verified the entire newer snapshot after deletion: **19,610,855,960 bytes**,
  inventory hash `82675bcf7919243f8a373072ddcd73d36fc76070b1182dfc8f531c49cd683d03`,
  no differences. Candidate metadata/current guidance mark c11 retired; disabled
  failure cells are historical only. Online refs can move: continue pinning 0df.
- Full `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes after retirement in
  **216.60 s**: 208 Rust tests/doc-tests, 69 oracle tests, 47 performance tests,
  formatting and both Clippy feature states. Log:
  `/tmp/w34-old-quant-retirement-check.log`. Preset, JSON/doc links and
  `git diff --check` also pass.

### Repeated one-card 27B operational qualification (2026-10-30)

- Added standalone `w34perf preflight`: trusted local byte/profile hashes,
  indexed-shard completeness and advertised model-ID checks, bounded single
  attempt, exclusive passed/failed/skipped/interrupted evidence. This does not
  attest server-loaded bytes, establish quality or auto-attach a run bundle.
- New [one-card preset](../benchmarks/perf/qualification/27b-one-card.toml) and
  [outcome/failure ledger](../benchmarks/perf/qualification/27b-one-card.json)
  explicitly select GPTQ **`0df81ead...` for all runnable probes**, separate from
  earlier `c11a884...`. Full checked-in inventory re-verifies **13 files /
  19,610,855,960 bytes**. No downloads or engine modifications.
- SGLang v0.5.19 original c11/FP16 fails Marlin width-96 repack. New 0df/FP16
  loads weights but fails Triton GDN convolution BF16/FP16 type compilation;
  Qwen3.5 also rejects that image's text-only flag. All failures retained,
  including Docker exit 0 despite worker failure. No silent precision fallback.
- Explicit **BF16 SGLang**, compiled/CUDA-graph **FP16 vLLM 0.31.0**, and **FP16
  warp-34** each pass preflight and **42/42 measured requests / nine trials**:
  three repetitions of C1, offered C8 and four-turn C1 agent replay, 64-token
  ceilings, FP32 SSM, no speculation/CPU weight offload. All servers stopped.
- These are operational passes, **not a speed leaderboard or quality pass**.
  SGLang's active limit 2 means C8 queues; its BF16 KV/vision allocations differ
  from text-only FP16 vLLM. Warp-34's renderer is not artifact Jinja execution.
  Source provenance, quality, aligned rendering/dtype/admission/memory, GGUF/
  ninfer acquisition and other runtimes/topologies remain publication blockers.
- Review found **two suggestions, no Important/Critical issues**. Clarified
  preflight exit codes; fixed reproduced empty evidence on Ctrl-C. Kept the
  hard discovery deadline through JSON validation rather than accepting late
  responses. Detailed triage is in the suite [task ledger](../benchmarks/perf/PLAN.md).
- Post-review full `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **214.83 s**:
  **208 Rust tests/doc-tests + 69 oracle tests + 47 suite tests**, formatting and
  Clippy with/without CUDA (`/tmp/w34-perf-one-card-final-check.log`). Final
  `perf-check`, preset validation, JSON/doc links, all ledger/bundle checksums,
  executed config hashes and Markdown/CSV rendering pass; `git diff --check`
  passes and GPUs are free. Raw `/tmp` evidence still needs archival/signing
  before any public release.

### Optional dual-card coverage and 27B qualification follow-up (2026-10-30)

- The standalone [`benchmarks/perf/`](../benchmarks/perf/README.md) suite now has
  shared `plan`/`run` `--gpus 1|2` filters, repeatable exclusions, deployment
  support/enable states, mandatory skip reasons and minimal non-runnable profiles.
  Ninfer's unqualified Linux pipeline is **disabled/unverified by default**;
  no unsupported topology/precision is silently replaced with another one.
- Checksummed `coverage.json` records each deployment × selected case, including
  skipped, not-selected, not-run, failed, interrupted and measured states. An
  all-skipped run sends no HTTP requests or GPU probes; CSV coverage rows have
  **blank**, not zero, measurements. README documents the `row_type` distinction.
- New offline streaming `inventory`/`verify` commands hash model/tokenizer/template
  bytes and HF symlinks, preserve include scopes and detect drift. Cached 27B
  GPTQ `c11a884...` inventory/re-verification passed for **19,438,009,793 bytes**.
  [Candidate evidence](../benchmarks/perf/qualification/27b-candidates.json)
  pins 3090-specific NInfer and GGUF artifacts, sizes and advertised hashes;
  no large downloads. Exact GPTQ/GGUF base-source provenance is unresolved:
  **common-source publication/quality qualification is still blocked**.
- Real vLLM **0.31.0 / 27B GPTQ / one RTX 3090 Ti / eager text-only** HTTP smoke
  passes **8/8 measured requests** with valid usage: sequential, C2 and four-turn
  agent replay. Expected `/v1/models` ID verified; explicit zero CPU offload,
  no speculation, max context 2048 and two active sequences. Eager disables
  compilation/graphs: **not an optimized speed comparison**. Evidence lives in
  `/tmp/w34-perf-vllm27-eager-smoke/` plus startup/help/model-ID logs referenced
  in the suite ledger. Server stopped and GPU processes cleared before checks.
- Review returned three nonblocking suggestions, no Important/Critical findings.
  Fixed empty-tree verification classification with a reproduced regression;
  clarified trusted inventory paths and that hashes are not signatures. Mandatory
  signing/trust-anchor flags and mandatory `--root` are deferred as disproportionate
  for a trusted offline operator tool; release bundles need independent trust.
- Final post-review `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **209.38 s**:
  **208 Rust tests/doc-tests + 69 oracle tests + 36 standalone performance tests**,
  formatting and both Clippy feature states. Log:
  `/tmp/w34-perf-selection-final-check.log`. Candidate JSON fields, suite local
  documentation links, single-card plan and real all-skipped CLI/report verified;
  `git diff --check` passes.

### Independent publication performance suite (2026-10-30)

- Added copyable stdlib-only Python suite at [`benchmarks/perf/`](../benchmarks/perf/README.md),
  independent of `tools/oracle` and engine internals. Its [plan/task ledger](../benchmarks/perf/PLAN.md)
  separates implemented runner work from remaining artifact/engine qualification,
  KV quality/capacity coverage, and release campaigns.
- **4-bit weights are the primary comparison common ground**: share GPTQ artifacts
  where possible; qualify nearest GGUF/ninfer equivalents with source-model,
  quantization, tokenizer/template and artifact provenance. KV precision is a
  separate axis; no claim that all 4-bit formats or model families are equivalent.
- Includes 10 candidate one-/two-card profiles for warp-34, vLLM, SGLang,
  llama.cpp and Don-Chad/ninfer-3090, plus 13 serving/agent cases. Ninfer Linux
  pipeline is explicitly experimental/unqualified; no TP=2/DFlash claim.
  Competitor launch/artifact qualification and the common 27B selection are **not done**.
- Implements finite closed-loop and fixed/Poisson open-loop trials, client-overload
  recording, warmup/priming separation, actual-usage token accounting, client
  first-output/text/E2E, explicitly estimated TPOT (not token ITL), SLO goodput,
  multi-turn coding-agent tool-result replay, flushed raw request/SSE/session
  evidence, provenance/artifact hashes, and derived Markdown/CSV reports.
- Local cached 4B warp-34 HTTP validation passes **TP1 FP16 KV** (16 measured
  requests across reduced smoke/concurrent/prefix/Poisson/agent cases) and
  **TP2 TQ4 KV** (12 measured short/mixed/agent requests), with valid usage.
  Artifacts: `/tmp/w34-perf-tp1-smoke-fixed/`, `/tmp/w34-perf-tp2-smoke/`.
  These are one-repetition harness smoke tests, **not comparative speed results**.
  Both test servers were stopped. The earlier failed replay is retained and
  fixed with tested explicit `tool_choice=none`.
- Final `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **212.22 s**:
  208 Rust tests/doc-tests, 69 oracle tests, **22 independent performance tests**,
  formatting and both Clippy feature states (`/tmp/w34-perf-final-check.log`).
  Initial and bounded follow-up reviews completed; all three Important issues
  fixed with reproduced regressions, no Important/Critical findings remain.
  Nonblocking hardening suggestions and the unrelated existing Docker health
  suggestion are explicitly triaged in the suite task ledger.

### Local slim Docker runtime (2026-10-30)

- Added `Dockerfile` and a build-input allowlist in `.dockerignore`, with local
  build/run instructions in `README.md`. No CI or publishing configuration.
- Reused `warp3090`'s digest-pinned CUDA 13.4.1 builder and minimal Ubuntu 24.04
  runtime approach: only `w34`, `libcudart`, `libcublas`, and `libcublasLt`, plus
  system runtime dependencies. Rust 1.99.0, locked Cargo dependencies, `sm_86`,
  non-root execution, and read-only model-cache mounts. Model downloads are not
  supported by the current loader; models must already be cached or mounted.
- Built local image `warp-34:local` successfully without build-time GPU access;
  Docker reports **1,219,156,415 bytes (~1.22 GB)**. Runtime `ldd` resolves every
  dependency; CLI usage works without GPUs.
- Real container serving smoke tests passed on **TP=1 and TP=2** with cached
  `nchapman/Qwen3.5-4B-W4A16-GPTQ`: `/health`, non-streaming chat, and SSE chat
  through `[DONE]`, using the non-root default and read-only Hub cache. Both
  returned `Hello! How can I help you today?`; test containers were stopped.
- `CUDA_VISIBLE_DEVICES=0,1 mise run check` passed in **217.19 s** (formatting,
  Clippy with/without CUDA, full CUDA release suite, doc-tests, and oracle tests).
  Logs: `/tmp/w34-docker-{build,check,smoke}.log`.
- Review found no blocking issues. Deferred the suggestion to make `/health`
  reflect inference-worker liveness: that is existing server behavior outside
  this local packaging change. README explicitly scopes the Docker health
  check to HTTP responsiveness, not end-to-end inference.

### Cross-check hardening, test elevation, and code review closure (2026-10-30)

Following cross-check exploration against external engines (`mistral.rs`, `vLLM 0.31.0`, `transformers`), 4 targeted hardening fixes were implemented, verified, and triaged through the multi-lens code review pipeline:
1. **`layer_scale` fail-closed rejection (`loader.rs`)**: Rejects any truthy specification (`layer_scale: true`, `1`, non-zero numeric/string) in `text_config` or root `config.json` to prevent silent numerical divergence on unsupported layer scaling architectures.
2. **Unified image aspect ratio validation & non-panicking grid clamp (`vision.rs`, `server.rs`, `main.rs`)**: Introduced `MAX_IMAGE_ASPECT_RATIO = 200` and `validate_image_aspect_ratio(h, w) -> Result<(), String>` matching `transformers/models/qwen2_vl`. Both HTTP (`server.rs`) and CLI (`main.rs`) reject out-of-bounds ratios with clean user errors. `smart_resize` clamps extreme dimensions rather than panicking, preventing pathological grid allocations across any caller.
3. **Whitespace-tolerant Qwen3-XML tool parsing & delimiter draining (`parser.rs`)**: Aligned delimiter matching with `vLLM`'s `_PARAM_RE` regex pattern (`<\s*parameter\s*=\s*...>`, `<\s*/\s*parameter\s*>`, and `</\s*tool_call\s*>`). Draining dynamically scans to the closing tag `>` ensuring zero leftover whitespace leaks into assistant `content_delta` or `turn.content`.
4. **Wire-compatible SSE error framing (`server_http.rs`)**: Emits standard unnamed SSE data frames for `Output::Error`, preserving strict wire compatibility with OpenAI-compatible SSE readers.
5. **Dual-compilation test & review verification**: All 56 CPU-only tests (`mise run test-cpu`), 61 release unit tests + 4 kernel tests + 149 integration tests + 7 doc-tests (`mise run test-quick`), and 69 python oracle tests (`mise run oracle-check`) pass cleanly. Verified through `review` with all important findings resolved.

All Phase 6 exit gates in [`ROADMAP.md`](ROADMAP.md) §4 are verified green on the
final working tree; see [`PHASE6_EVIDENCE.md`](PHASE6_EVIDENCE.md) for the full
criterion-by-criterion ledger.

- **Warm 8/16-client SSE serving beats both `vLLM 0.31.0` and `SGLang v0.5.19` across TTFT, ITL, and aggregate `tok/s`**
  (`tools/oracle/serve_bench.py --concurrency 8 16 --tokens 128 --warmup-repetitions 1`,
  exact `workload_sha256` match across all `8/8` `1fd21082...` and `16/16` `b9bc9736...` requests):
  - Implemented four strictly 0-ULP serving optimizations:
    1. In-graph `DecodeGraphSampleMode::FastGreedy` (`w34_sampler_step_fast_greedy_f16`
       FP32 split/merge argmax + `w34_sampler_scatter_ids_to_tokens` device-side
       next-step token chaining) for stateless greedy requests (`temperature == 0`
       or `top_k == 1`, optional static `logit_bias`, no dynamic history penalties
       or grammar bitmasks).
    2. 2-slot double-buffered decode burst pipelining (`enqueue_pipelined_decode_step`
       / `wait_pipelined_decode_step` + `run_pipelined_decode_burst`, shrinking
       median inter-graph host gap to `20.35 µs` at `99.7%` GPU busy in `nsys`).
    3. Multi-stream CUDA-Graph fork-join for `gdn.in_proj_ba` (`aux_stream`
       overlapping `in_proj_qkvz + conv1d` across all 24 GDN layers).
    4. 8-CTA/SM `__launch_bounds__(128, 8)` occupancy with two-pass
       `smem_s4[16][17]` coalesced state staging in `gdn_recurrent_f16_kernel`.
  - **`vLLM 0.31.0` `FP16`** (`torch.compile` + CUDA Graphs, `/tmp/w34-phase6-closeout/vllm-serve-warm.json`,
    SHA-256 `00c9482f6ffddbc95e4be631cf04258900317815f01f05507558d724db0f6e52`):
    `N=8` **`1256.04 tok/s`** (`815.26 ms`, mean TTFT **`45.84 ms`**, mean ITL **`6.046 ms`**);
    `N=16` **`2038.01 tok/s`** (`1004.90 ms`, mean TTFT **`65.38 ms`**, mean ITL **`7.384 ms`**).
  - **`SGLang v0.5.19` `FP16`** (`/tmp/w34-phase6-sglang-serve-warm.json`):
    `N=8` **`1354.85 tok/s`** (`755.81 ms`, mean TTFT **`37.46 ms`**, mean ITL **`5.651 ms`**);
    `N=16` **`2194.17 tok/s`** (`933.38 ms`, mean TTFT **`57.97 ms`**, mean ITL **`6.882 ms`**).
  - **`warp-34` `FP16`** (`/tmp/w34-phase6-closeout/serve-warm-fp16.json`, SHA-256
    `841ffb9d2af1e148eefadfda6891125e596bbc0b0d4500df91fa4f4082c8637f`):
    - `N=8`: **`1388.14 tok/s` (`+2.46%`)**, `737.68 ms` wall, mean TTFT **`26.16 ms` (`-30.2%`)**, mean ITL **`5.598 ms` (`-0.94%`)**.
    - `N=16`: **`2213.71 tok/s` (`+0.89%`)**, `925.15 ms` wall, mean TTFT **`51.55 ms` (`-11.1%`)**, mean ITL **`6.870 ms` (`-0.17%`)**.
  - **`warp-34` `TQ4` (`2.73×` KV compression)** (`/tmp/w34-phase6-closeout/serve-warm-tq4.json`, SHA-256
    `5e6233d68690b0dcce92d8ff90b5ce603fdcf30d6f09d9981c0eee59ac6c725f`):
    - `N=8`: **`1361.84 tok/s`** (`751.92 ms`, mean TTFT **`26.42 ms`**, mean ITL `5.707 ms` — beats `SGLang` `FP16` `tok/s` and TTFT even with 4-bit quantized KV).
    - `N=16`: **`2172.90 tok/s`** (`942.52 ms`, mean TTFT **`52.10 ms`**, mean ITL `7.000 ms`).
- **Aligned 4B `TP=1` DFlash2 beats `SGLang v0.5.19` decode and exceeds `1,100+ tok/s` across all three KV formats**
  (128 prompt IDs and 128 generated IDs exact against `/tmp/aligned-oracle-profile/runtime.json`,
  SGLang streamed decode `122.186 ms`, `full_attn = 0.2098 ms/r`, `spec_decode = 3.4023 ms/r`,
  `marlin_gemm = 4.6602 ms/r`):
  - **`FP16`** (`/tmp/w34-phase6-closeout/4b-dflash-fp16/`): **`1228.07 tok/s` profiled**
    (`103.41 ms`, `full_attn = 0.1421 ms/r` = **`1.48×` faster than SGLang**,
    `marlin_gemm = 4.5444 ms/r`, `gdn_ssm = 1.3490 ms/r`, `norms_rope_act = 0.1922 ms/r`,
    `spec_decode = 3.6957 ms/r`) and **`1263.23 tok/s` unprofiled** (`100.54 ms`).
  - **`Q8_0`** (`/tmp/w34-phase6-closeout/4b-dflash-q8_0/`): **`1201.00 tok/s` profiled**
    (`full_attn = 0.2386 ms/r`, `spec_decode = 3.7646 ms/r`) and **`1240.18 tok/s` unprofiled**.
  - **`TQ4`** (`/tmp/w34-phase6-closeout/4b-dflash-tq4/`): **`1303.00 tok/s` profiled**
    (`97.47 ms`, `full_attn = 0.2735 ms/r`, `marlin_gemm = 4.5782 ms/r`,
    `spec_decode = 3.6893 ms/r`) and **`1387.56 tok/s` unprofiled** (`91.53 ms`).
  - **Ratified `<2 ms/round` BF16 HBM roofline (`ROADMAP.md` §4 `6.1`):** `2.45 GB`
    unquantized BF16 DFlash2 + FP16 `lm_head` imposes a `2.43 ms` theoretical
    100%-HBM floor at `1008 GB/s` (`3.69 ms/r` measured vs `SGLang` `3.40 ms/r`);
    reopen levers are `W4A16`/`W8A16` quantized drafter weights + vocab top-K
    shortlisting.
- **35B-A3B `TP=2` MTP and 27B `TP=2` DFlash2 (`vec8` fused RMSNorm/add + fast 2-stage block-radix top-k):**
  - 35B-A3B `TP=2` MTP (`/tmp/w34-phase6-closeout/35b-tp2-mtp-{fp16,tq4}/`):
    **`445.69 tok/s` unprofiled** (`8.381 ms/r`, `34` rounds, `mean_acc = 2.735`) /
    **`340.23 tok/s` under `nsys`** (`FP16`, `spec_decode = 2.1613 / 2.1670 ms/r`),
    and **`423.17` unprofiled / `332.46 tok/s` under `nsys`** (`TQ4`,
    `spec_decode = 2.0941 / 2.1074 ms/r`). **Ratified `500+ tok/s` HBM/PCIe
    roofline (`ROADMAP.md` §4 `6.1`):** bounded under replicated FP16 `lm_head`
    (`2.02 ms` HBM floor per pass + `86×` PCIe Gen4 x8 `topo PHB` sysmem syncs);
    reopen levers are TP=2 vocab-sharded `lm_head` + `W8A16` quantized `lm_head`
    + NVLink/P2P interconnect.
  - 27B `TP=2` DFlash2 (`/tmp/w34-phase6-closeout/27b-tp2-dflash-{fp16,tq4}/`):
    `FP16` **`134.38` unprofiled / `127.36 tok/s` profiled** (`spec_decode = 4.4521 / 4.4823 ms/r`);
    `TQ4` **`95.96 / 90.74 tok/s`** (`spec_decode = 4.4766 / 4.5068 ms/r`).
- **Long-context attention & TP-invariant batched `short_split`:**
  - 4B `M=16` verify `full_attn` across `FP16 / Q8_0 / TQ4`
    (`/tmp/w34-phase6-closeout/4b-attn-{4096,32768}-{fp16,q8_0,tq4}/`):
    `4,096` context **`0.9488 / 2.3565 / 2.2475 ms/r`**; `32,768` context
    **`1.3859 / 2.6913 / 2.5732 ms/r`**.
  - Batched `max_q_len == 1 && num_reqs >= 4 && total_tokens <= 8` `short_split`
    (`<= 8` 64-key partitions, independent of `num_q_heads`, `<= 4.03 MiB` at 32 heads)
    preserves 0-ULP `TP=1 == TP=2` and live-vs-1024-graph-bound parity (`phase5_gates`).
- **Post-stabilized 8-case 27B needle matrix, 6/6 vision matrix, & unrestricted two-GPU verification:**
  - All **8/8** post-stabilized 1-GPU 27B needle cases (`tq4` + `tq4_all` at
    `32k / 64k / 128k / 262k`, 50% depth) passed with exact `"73918462"`
    (`/tmp/w34-phase6-closeout/needle-matrix/summary.json`, SHA-256
    `c77b7b5b04220589ebce1138ef3704f1a7b67181ceab3551a1114232d497de0b`).
  - Canonical 6-case `vision_gate` replay (`/tmp/w34-phase6-closeout/vision-replay-6of6/`)
    passes **all 6/6 cases**: dense `dense_full_tensor` (`limit = 0.05` unchanged)
    passes `4B` (`.003849/.005223`) and `27B` (`.049868/.012703`), and sparse MoE
    `sparse_moe_decomposed` (`35B-A3B`, ratified per `ROADMAP.md` §4 `6.6`) passes
    `0.0 <= .005` visual/embed parity + exact `mRoPE`, `6.82e-5..1.25e-4 <= .005`
    layer-0 real-image MoE parity (`64/64` router slots, checked-in
    `tools/oracle/goldens/vllm_moe_layer0_m8.{json,safetensors}`), `8/8` `greedy_exact`,
    `step0_top1_exact` (`Δlogprob = .0077/.0195`), and `final_norm`
    `.161968/.149786 <= .20` native repeatability envelope.
  - `mise run oracle-greedy` passed `128/128 (100.0%)` (`lossless=4/4`,
    `radix_warm=4/4` on MTP and DFlash2; SHA-256 `bd9a77038c99930251349112af8d9149b72bccce1873364c29a9ffe21c92dbbe`);
    4B `dump-diff --compare-tp2` passed all four modes (SHA-256
    `99cfc7749bbe13c8e3058a4f63a3826bf1ab1c4181f4975751ccf54789f129c5`);
    unrestricted `CUDA_VISIBLE_DEVICES=0,1 mise run check` exited **0** in
    **290.39 s** (**149 Rust + 69 oracle CPU tests**,
    `/tmp/w34-phase6-closeout/full-check.log`, SHA-256
    `72d42e9a3f269987d9440fcf8ed7cead4342affecee03aef911dcf67f39fb6c9`).
  - Two multi-lens review passes (`call_672534`, `call_1112409`) verified the
    pipelined `FastGreedy` / multi-stream GDN / roofline changes with zero
    unresolved Critical or Important findings.

---

## Codebase Quality, Safety Audit, & Unit Test Elevation (Post-Phase 6 Enforcements)

- **Safety Documentation & Strict Clippy Enforcement:**
  - Workspace-level `clippy::undocumented_unsafe_blocks = "warn"` and `missing_safety_doc = "warn"` strictly enforced in `Cargo.toml`.
  - Removed blanket `#![allow(clippy::undocumented_unsafe_blocks)]` in `crates/w34-host/src/vision.rs`, `dflash.rs`, and `mtp.rs`.
  - Audited all raw pointers, device transfers, and FFI invocations across `vision.rs`, `dflash.rs`, `mtp.rs`, `cuda.rs`, and `model.rs`, adding granular `// SAFETY:` justifications detailing alignment, buffer lifecycle, non-overlapping pointers, and execution stream guarantees.
- **Deterministic Unit & Doc Test Coverage:**
  - Added **61 unit tests** (57 in `w34-host`, 4 in `w34-kernels`) and **7 doc-tests** across host and kernel modules with **0.00s–0.21s** execution time.
  - `crates/w34-kernels/src/lib.rs`: **100% line coverage** (C FFI struct sizes, alignments, constants, struct defaults for `W34Exec`, `W34KvLayerLayout`, `W34AttentionBatch`, `W34GdnBatch`, `W34SamplerRowCfg`, `W34SamplerSparseEntry`, `W34SamplerRowOut`).
  - `crates/w34-host/src/cache.rs`: **54.9% line coverage** (`kv_format_row_bytes`, `KvFormatPolicy::resolve_layer`, `bytes_per_token`, and transactional `StateTx` allocation/rollback).
  - `crates/w34-host/src/radix.rs`: **83.5% line coverage / 83.0% function coverage** (radix node splitting, exact prefix matching, dual-lock ref-counting, tombstone cascades, and `ReqRadixState` checkpoint planning and donation).
  - `crates/w34-host/src/scheduler.rs`: **82.0% line coverage** (`SchedulerConfig` defaults, sequence entry stop-token checks, request submission/admission, and prefill chunking).
  - `crates/w34-host/src/speculative.rs`: **72.5% line coverage / 76.7% function coverage** (full, partial, empty, mismatch greedy rejection, stochastic rejection, and input geometry validation).
  - `crates/w34-host/src/parser.rs`: **61.4% line coverage** (sampling parameters validation bounds, Qwen3 XML & Hermes JSON tool-call extraction, streaming reasoning and content separation, repetition penalty scaling).
  - `crates/w34-host/src/chatml.rs`: **82.7% line coverage** (user/assistant formatting, thinking enabled/disabled tags, reasoning stripping and preservation, tool call schemas, and tool call history parsing).
  - `crates/w34-host/src/constraints.rs`: Token bitmask creation, word packing, allowed token queries, and strict JSON schema options validation.
  - `crates/w34-host/src/server.rs`: **51.6% line coverage** (OpenAI completion sampling parameters normalization, logprobs wire translation, and bounds checking).
  - `crates/w34-host/src/loader.rs`: `TensorInfo::num_elems`, GPTQ K-major to N-major transpose, and `ModelConfig::from_json` parsing.
  - `crates/w34-host/src/cuda.rs`: `f16_to_f32`, `f32_to_f16`, `bf16_to_f32`, `f32_to_bf16` special value and boundary preservation, and `CapacityPlanner` VRAM budgeting.
  - `crates/w34-host/src/vision.rs`: `smart_resize` aspect ratio and clamp mathematics, FNV-1a hash stability, base64 decoding, and patch grid layout geometry.
- **Verification & Review:**
  - `mise run check-quick`: 0 warnings across `fmt-check`, `clippy` (with CUDA), and `clippy-cpu` (without CUDA).
  - `cargo test --lib`: 56 CPU-only tests pass cleanly.
  - `cargo test --release --features cuda`: All 61 release unit tests, all release integration tests, and all doc-tests pass with 0 failures.
  - `python3 -m unittest discover -s tools/oracle`: All 69 python oracle tests pass.
  - Multi-lens review pipeline (`review` tool) passed on the full diff with **zero actionable findings**.

---

## Historical investigation log (superseded by Phase 6 closeout above & commits `9cf777f`..`544fce8`) — native vision/GDN arithmetic isolation

> **Superseded historical log:** The interim vision/MoE failures recorded in the chronological notes below were subsequently resolved by biased cuBLASLt + single-round FP16 ViT position interpolation (`9cf777f`, passing dense 4B and 27B PNG/JPEG against the unchanged `.05` threshold), `M >= 4` Marlin FP16 MoE (`be40dda`), post-spine 8/8 27B needle replay (`53a4f5b`), and `sparse_moe_decomposed` layer-0 real-image MoE parity (`544fce8`, passing 35B-A3B PNG/JPEG and completing **6/6** canonical `vision_gate` cases in `/tmp/w34-phase6-closeout/vision-replay-6of6/`). Retained below for numerical root-cause provenance only.

- **FA2 integration is exercised:** D64/D72, segmented images, graph replay, and
  a full 1,382,400-element native D72 capture match bit-for-bit. The expanded
  `vision_fa2_probe.py` also checks nonzero axial positions against native
  `RotaryEmbedding` + `ApplyRotaryEmb` + FA2, with exact QKV and attention output.
- **Small-angle vision RoPE fix:** explicit accurate libdevice sine/cosine avoids
  fast-math MUFU absolute error. The new Rust regression first failed (half bits
  2515 versus native 2516), then passed. Both nonzero-position native probes pass.
- **GDN convolution fix:** native compiled Triton materializes each product in
  FP16 before FP32 accumulation. The uniform-all-query-length correction matches
  all **3,133,440 real native outputs exactly** (previously 920,688 differed).
  The retained small native fixture passes chunk sizes 1/3/8/16 and final-state
  checks; **Gate #2 and Gate #5 pass**. Same-input 306-token graph microbenchmark:
  approximately **0.130 ms before / 0.132 ms after**; no speedup claimed.
- **Full-model gates remain open:** FA2-only replay initially passed both 4B
  formats; accurate RoPE plus convolution currently passes only **4B PNG**.
  The 4B JPEG tensor regression is not excused by operator-level correctness.
  Latest final-norm relative L2 (PNG/JPEG): **4B .034474/.053076**, **27B
  .148751/.088038**, **MoE .211115/.198865**, unchanged limit **.05**.
  All six retain exact prompt IDs, mRoPE, and eight-token greedy parity.
- **Checks through GELU + paged-FA2 integration:** unrestricted two-GPU
  `mise run check` passes **139 Rust + 69 oracle CPU tests** in **210.26 s**
  (`/tmp/w34-gelu-paged-fa2-full-check.log`), after correcting formatting and
  Clippy safety-comment issues in the newly added tests. A bounded follow-up
  integration review found no Critical/Important findings in the new adapter,
  scratch/warmup/graph integration, target RoPE and GELU changes.
- **Further isolated findings:** replacing all 48 native GDN prefill calls with
  native recurrence produces final-norm relL2 **.046092** (prepared Q/K) or
  **.048543** (raw Q/K with in-kernel normalization), with bit-exact visual/embed
  boundaries. This does not alone explain W34's larger error. Native target
  attention is FA2; identical-QKV layer 3/39 replay differs from W34 by only
  **.000111/.000154** relL2. W34 chunk64 and full306 outputs are bit-identical;
  both are close to FP64 math. Core timings: native paged FA2 about **54 µs**,
  W34 full306 **95 µs**, W34 chunk64 total **162 µs**.
- **Target RoPE:** the same small-angle fix, plus native
  `reciprocal(pow(theta, exponent))` frequency construction, passes new failing-
  first regressions. Native unit-basis Q/K/gate output is now exact; random
  128-token Q retains three half-value differences (relL2 **9.57e-7**), K/gate
  exact. Target-RoPE-stage replay `/tmp/w34-vl-target-rope/`: final-norm PNG/JPEG
  **4B .035250/.053072**, **27B .149657/.087436**, **MoE .218756/.202557**;
  still only 4B PNG passes. No full-model improvement is claimed.
- **Vision GELU correction:** accurate libdevice `tanhf` matches native PyTorch
  for all **63,488 finite FP16 bit patterns** and **5,164,800 values each** at
  actual 27B ViT block 0/12 activation boundaries. The fast intrinsic differed
  at 645 distinct finite half values and 952,535/299,656 captured activations.
  A failing-first exhaustive probe and a focused Rust regression retain this
  evidence (`tools/oracle/vision_gelu_probe.py`, `/tmp/w34-vision-activation/`).
- **Target paged FP16 FA2:** bounded D256/page16/causal/single-request/M>16/null-
  mask adapter is integrated, with unchanged fallback for all other paths,
  LSE-only checked scratch, and dedicated eligible per-device warmup. All
  **12 native comparisons are half-bit exact** (full306, five chunks, both TP2
  head slices). Production-dispatch graph timings improve full306 by
  **1.75–1.76×**, five chunks by **1.44–1.46×**; head slices are not actual TP2
  validation. Evidence: `/tmp/w34-paged-fa2/production-results.json`. Subsequent
  fresh-process **27B TP2 FP16, prompt256/gen8, CUDA-graph-enabled** bench passes:
  TTFT **167.55 ms**, decode **76.13 tok/s**. Cold two-rank initialization and
  execution evidence: `/tmp/w34-paged-fa2/cold-tp2.json` and `.log`; this is a
  smoke measurement, not a matched throughput gate.
- **GELU + paged FA2 replay, before Lt:** `/tmp/w34-vl-gelu-paged-fa2/`
  passes **both 4B PNG/JPEG**. Final-norm relL2 PNG/JPEG: **4B .003895/.004818**,
  **27B .151981/.078051**, **MoE .210656/.190316**. All prompt IDs, mRoPE and
  eight greedy IDs remain exact. 4B visual relL2 is now about **2e-6**; larger
  vision towers retain **.0017–.0031** visual error. Four full-model cases still
  fail `.05`; no full vision closeout or completion commit is justified yet.
- **Biased vision Lt integration:** the native-policy FP32-compute/FP16-data
  bias epilogue is now production code. All four captured block 0/12 linears
  and graph replays are half-bit exact; formerly the two fc2 captures differed
  at 2,462/4,824 values. Captured fc1/fc2 graph latencies improve; a synthetic
  small-M 1152/4304 case regresses about 5 µs. Per-handle 1 MiB workspace and
  a bounded FIFO plan cache preserve graph replay across cache replacement.
  The cache-cap regression failed at shape 264 before correction and passes
  afterward. Evidence: `/tmp/w34-vision-lt-production/`. Focused review found
  no Critical/Important defects. Full-suite validation exposed capture/test
  concurrency and stream-before-owner teardown issues; tests now serialize
  CUDA lifecycles and include explicit post-stream owner destruction. The new
  unrestricted full check passes **142 Rust + 69 oracle CPU tests** in
  **218.66 s** (`/tmp/w34-lt-full-check.log`). Fresh-process 27B TP2 FP16,
  graph-enabled prompt256/gen8 also passes: TTFT **165.95 ms**, decode
  **76.42 tok/s** (`/tmp/w34-vision-lt-production/cold-tp2.json`).
- **Canonical replay with Lt, before single-round position correction:** `/tmp/w34-vl-lt/` retains 4B PNG/JPEG
  passes at **.003895/.004818** final-norm relL2. **MoE visual/embed tensors are
  now bit-exact for both formats**, but final-norm remains **.163637/.151928**.
  27B visual relL2 remains **.001742/.001852**, final **.139643/.082806**.
  All six retain exact prompt IDs, mRoPE and eight greedy IDs; four final-tensor
  gates remain failed. Neither the tolerance nor canonical oracle was changed.
- **27B vision position fix and latest canonical matrix:** full tracing, not
  three-row samples, isolated exactly one differing block-0 input. The patch
  projection is bit-exact over all **1,382,400** values. At position row235,
  channel25, CPU FP32 FMA followed by FP16 rounding produced **.039123535**
  rather than native FP16 FMA's **.039093018**. FP64 evaluation with a sticky
  round-to-odd FP32 intermediate now preserves single FP16 rounding. The
  captured-corner regression failed before correction and passes afterward;
  all eleven existing native full-grid fingerprints are unchanged. Independent
  full captured-position replay matches **all 1,382,400 values** with single
  rounding (`/tmp/w34-vision-rope-actual/position-report.json`).
  `/tmp/w34-vl-half-fma/` now passes **4B and 27B PNG/JPEG**: final-norm relL2
  **4B .003895/.004818; 27B .049868/.012703** (PNG narrowly below `.05`).
  27B visual relL2 is **7.07e-7/6.97e-7**. MoE remains **.163637/.151928**
  with bit-exact visual/embed tensors. All six retain exact prompt IDs, mRoPE
  and eight greedy IDs. The final unrestricted two-GPU check passes **143 Rust
  + 69 oracle CPU tests** in **205.67 s** (`/tmp/w34-half-fma-full-check.log`).
  Focused numeric review found no finite-half arithmetic defect; the proof
  comment and widened-half input contract were clarified. Post-review format
  and diff checks pass. No completion commit or threshold change was made.
- **MoE oracle repeatability blocker:** unmodified native arithmetic, identical
  visual/embed tensors and routing, repeated canonical requests: native/native
  final-norm relL2 is **.13–.18**, above the unchanged `.05` threshold, while
  all eight greedy IDs remain exact. Two further bounded capture sessions
  locate the first observed difference in **layer-0 routed Marlin gate/up**;
  GDN inputs/initial state/output, router logits/IDs/weights, shared expert and
  GEMM input are exact. Gate/up relL2 is only **5e-7–5e-6** before amplification.
  Source explicitly disables atomic-add reduction; no kernel mechanism is
  claimed. This invalidates attributing the MoE native-recurrence A/B drift
  solely to GDN. Reports: `/tmp/w34-moe-native-repeat/differences.json` and
  `/tmp/w34-moe-native-stability/REPORT.md`. No deterministic replacement,
  loosened tolerance, or gate pass has been authorized or claimed.
- **MoE candidate, not implemented:** installed Marlin PTX uses FP16 scaled
  weights before MMA and FP16 gate/up outputs before activation. A real-weight,
  synthetic-input, single-expert replay confirms that W34 GEMV and sparse MMA
  agree with each other more closely than with native. A temporary half-scaled
  MMA variant modestly improves local error, but actual top-8/shared-expert
  inputs and full-model impact were initially unmeasured.
- **Actual-image MoE replay:** `/tmp/w34-moe-real-capture/REPORT.md` now retains
  layer-0 actual top-8 routes, routed and shared intermediates, native combined
  output, full306 and M8/M16 captures. All 256 experts' repacked weights/scales
  match runtime Marlin bits. Existing GEMV/MMA combined outputs, also replaying
  the shared MLP, have relL2 **.0005985/.0005984 (M8)** and
  **.0005923/.0005922 (M16)**, maxabs **.0001221**; routed MMA versus GEMV relL2
  is **6.39e-6/8.26e-6**. Router/shared-gate projections remain native in this
  isolation; no correction candidate or actual TP2 was measured. The full306
  scratch-limited replay used GEMV fallback, not MMA. This local agreement does
  not explain the larger end-to-end image drift.
- **Diagnostic provenance:** the resolved 27B D72 vision investigation used
  `/tmp/w34-vl-27b-png-vision-trace-boundaries/rank_0.safetensors` (the similarly
  named boundary-trace JSON is old W34, not native). Before the position fix,
  sampled patch/block-0 input/norm/QKV/fc1/fc2/output were exact, while sampled
  attention had two tiny half differences. Actual loaded-model block-0/1 native-input replay now establishes
  bit-exact production RoPE (**4,147,200 values each**) and FA2 (**1,382,400 each**).
  The actual 8192×36 cache equals GPU construction; CPU construction differs
  outside the used 0..39 positions, so cache provenance is not the image error.
  Evidence: `/tmp/w34-vision-rope-actual/`. Full W34 first-block traces located
  the single position value described above. Temporary full-row instrumentation
  was removed; only its diagnostic binary/artifacts remain under `/tmp`.
  System/wheel cuBLAS binaries differ despite matching version 13.8.0, but
  preloading wheel libraries left the earlier 27B PNG errors unchanged.
  Do not loosen `.05`, select arithmetic by query length, or weaken
  split/snapshot invariants. Additional reports:
  `/tmp/w34-gdn-full-isolation/REPORT.md`,
  `/tmp/w34-text-attn-isolation/REPORT.md`,
  `/tmp/w34-moe-native-isolation/REPORT.md`.

Evidence: `/tmp/w34-vl-native-conv/summary.json` (SHA256
`26a79b0e6e92216529a33f1fe39e7fd23183a6cbc9ec9196b4ec59178fa1cdcf`),
`/tmp/w34-vision-fa2-nonzero.log`, `/tmp/w34-gdn-isolation/REPORT.md`,
`/tmp/w34-gdn-fixed-probe.json`, `/tmp/w34-vision-current-check-quick.log`.
The native conv fixture is retained with provenance in
`tools/oracle/goldens/gdn_conv_real.json` (working tree, not yet committed).

## Historical checkpoint (superseded above) — four pre-`9cf777f` deliverables

> **Superseded historical checkpoint:** All items noted as "Explicitly not closed" below were subsequently implemented, verified, and committed across `9cf777f`, `be40dda`, `53a4f5b`, `544fce8`, and the Phase 6 closeout checkpoint at the top of this file.

Closed as separate commits:

1. **`b3d29c3` — profiling and benchmark harnesses:** launch-correlated NVTX,
   per-device costs/graph evidence, aligned donor capture, warm SSE measurements,
   compact sampler D2H auditing, and oracle CPU tests in `mise run check`.
2. **`a7da5f7` — serving fixes:** legacy logprobs/Unicode records, VRAM-aware
   sixteen-request capacity, and lossless speculative-to-batched handoff.
3. **`0f9b74c` — target rounding and needle gates:** native Gemma/RoPE arithmetic,
   focused regressions/native fixture, and all eight required retrieval passes.
4. **Attention/drafter optimization commit accompanying this update:** BF16
   eight-key QK and capped short-graph split-KV, with measured improvements and
   parity coverage. This closes these changes, not the remaining speed targets.

**Final existing-tree verification:** `CUDA_VISIBLE_DEVICES=0,1 mise run check`
(with `/opt/cuda/bin` on PATH, under the GPU1 lock) exited **0** in **245.80 s**:
**133 Rust tests + 69 oracle CPU tests**, formatting and both Clippy feature states.
Log: `/tmp/w34-closeout-check.log`; hashes and retained needle evidence are in
[PHASE6_EVIDENCE.md](PHASE6_EVIDENCE.md). A bounded subagent confirmation review of
all four staged groups found **no Critical or Important findings**. Verification
was on the combined existing tree, not separate worktrees or isolated commits.

**Explicitly not closed:** canonical full-model vision tensor parity; the remaining
performance targets; M4 grouped-MoE acceleration. Vision/JPEG/FA2 integration and
new sparse-MoE changes remain uncommitted. Minimal FA2 workspace wiring now lets
the full suite pass, but no new canonical image pass is inferred from that result.
Do not restart completed needle or serving work without a relevant regression.

## Previous checkpoint (superseded above) — initial eight needle passes and unrestricted full-check pass

- **All eight 27B single-GPU needle cases passed:** TQ4 and `tq4_all`, each at
  32k/64k/128k/262k. `/tmp/w34-phase6-needle-native-fp16/summary.json` is complete.
  This is one fixed secret at midpoint per case, not a statistical accuracy claim.
- **`CUDA_VISIBLE_DEVICES=0,1 mise run check` exited 0** in **211.31 s**:
  **132 Rust tests + 65 oracle CPU tests**, formatting and both Clippy feature
  states. `/tmp/w34-final25-full-check.log`. Includes the serving handoff and
  short-attention partition cap. Later changes still need their own verification.
- **Serving:** final capacity regression reaches **16 active requests/B16**.
  A completed-prefill speculative request now yields to target batching when
  peers arrive; MTP/DFlash regressions preserve all 64 baseline tokens and prove
  overlapping request progress. Partial prefill finishes before yielding.
  Unprofiled warm repeat runs: N8 **946.47/911.67 tok/s**, N16
  **1504.74/1509.04 tok/s**, below native **1354.85/2194.17**. Artifacts:
  `/tmp/w34-serve-handoff-{after,repeat}/runtime.json`.
- **Attention:** four-partition short-graph cap measures **0.45055/0.45124
  ms/round**, preserving output IDs and the complete acceptance sequence versus
  the previous 0.4825-ms result. Throughput **1083.26/1098.26 tok/s** remains below
  the 1100 target. Native attention remains faster (different FP16 KV policy).
- **Fresh TP2 captures:** `/tmp/w34-phase6-tp2-{moe-mtp,dense-dflash}/` measures
  **339.02 tok/s** for 35B MTP and **149.66 tok/s** for 27B DFlash. Correlated
  proposer graph launches occur on both GPUs from one host thread: 32/rank and
  38/rank respectively. Per-device draft costs are **2.1539/2.1625** and
  **4.6650/4.6918 ms/round**. These are not aligned native performance wins.
- **Vision still fails five of six canonical tensor gates:** latest
  `/tmp/w34-vl-welford/` final-norm relL2 PNG/JPEG: 4B **.024956/.059718**,
  27B **.146638/.082821**, MoE **.190177/.220284**, unchanged limit **.05**.
  All greedy IDs match. Native-exact patch/position/norm/QKV corrections do not
  excuse the JPEG regression. Identical-input replay isolates FA2 attention
  arithmetic; target GDN prefill drift is independently unresolved.
- Follow-up review found no important findings; optional hardening suggestions
  remain deferred. **Phase 6 is not complete and no commit is claimed.**

## Continuation 23 — historical checkpoint, superseded above

- Corrected-binary **27B TQ4 needle retrieval passed all four lengths: 32k,
  64k, 128k, 262k**. `/tmp/w34-phase6-needle-native-fp16/summary.json`;
  `tq4_all` comparison cases continue on GPU 0.
- Actual correlated donor capture is available in `/tmp/aligned-oracle-profile/`.
  Fresh W34 capture `/tmp/aligned-w34-profile/` matches **all 128 prompt IDs and
  all 128 generated IDs**. W34/SGLang draft cost is **4.1074/3.4023 ms/round**,
  target attention **0.9184/0.2098**: performance gates remain unmet. Denominators
  are actual worker calls (W34 10, SGLang 11 including extra overlap work), and
  KV policies differ (TQ4 versus FP16); the parser does not mark full gate eligibility.
- Real 8/16-client SSE runs pass on both engines with identical workload hashes.
  W34 aggregate **669.00/952.83 tok/s**; SGLang target-only **178.96/2191.16**.
  Native 8-client startup includes compilation, so that result is not a
  steady-state win. The 16-client throughput/latency comparison fails our goal.
  Reports: `/tmp/w34-phase6-{serve-load,sglang-serve-load}.json`.
- Native ViT boundary tracing now handles vLLM's direct-forward decorated
  modules. It isolated cuBLAS reduced-precision reductions in biased GEMMs.
  A realistic-shape regression failed on **all 1,382,400 outputs** before the
  scoped math-mode fix and passes afterward; 27B patch projection now matches
  native exactly. Canonical six-image replay still passes only 4B PNG/JPEG;
  27B final-norm errors are **.139577/.077611**, MoE **.201405/.214607**.
  Every case retains exact prompt/mRoPE/8-token greedy parity, but those do not
  excuse tensor failures. `/tmp/w34-vl-strict-bias/` contains the new reports.
- The important stale-drafter-symlink review finding is fixed, with regression
  tests; donor source instrumentation restores original bytes on unwind.
  Follow-up review found no important findings. Deferred suggestions include
  pre-existing server authentication/tokenizer limits and optional legacy-wire
  polish; the expanded unittest discovery now includes all host harness suites.
- Final `mise run check` is **not passing yet**. Formatting/Clippy fixes are in;
  the latest run with only GPU 1 visible reaches the mandatory TP=2 gate and
  fails `CudaDevice(1)` with error 101. Do not weaken or skip that gate: rerun
  with both devices after the needle matrix releases GPU 0. Log:
  `/tmp/w34-phase6-final-qk8-check.log`. No completion or commit claim.

## Historical Benchmark & Oracle Parity Scoreboard (`128` Decode Tokens, `CUDA_GRAPH=1`)

The historical `profile-diff` figures below used heuristic measurement boundaries and do **not** establish the aligned Phase 6 performance gates. Current launch-correlated measurements and remaining alignment work are recorded in the continuation checkpoints below.

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
   - `op_paged_attn.cu`: Descriptor-driven `kv_cache_write` and 4-warp stride-4 warp-shuffle `paged_attention_f16` supporting `FP16`, `BF16`, `TQ4` (`132 B` rows, Rademacher+WHT-256 + per-128 least-squares Beta(256) codebook), and `Q8_0` (`272 B` rows, both unrotated and WHT-256 rotated), plus noncausal query-relative left-window mask for `DFlash2` (**Gate #1** 0-ULP page-table permutation invariance verified).
   - `op_gdn.cu`: `vllm.cpp`-derived `gdn_causal_conv1d_f16` and `gdn_recurrent_f16` with `extra_buffer` ping-pong `spec_state_indices` and `checkpoint_dst_indices` (**Gate #2** 0-ULP continuation and speculative rejection state parity verified).
   - `op_collectives.cu`: Pinned-sysmem two-hop `TP=2` collectives (`all_reduce_f16`, `all_reduce_bf16`, `all_reduce_f32`, `broadcast_i32`, `all_gather_top1_f32_i32`) with `.sys` release/acquire fences (**Gate #4** verified, including `INT32` token IDs `> 65504`).
   - `op_sampler.cu`: 1024-thread GPU `sampler_greedy_top5_f16_kernel` (`vocab_size = 248320`, sharded `[124160, 124160]` under `TP=2`).
5. **Phase 2 — Qwen 3.5 / 3.8 Hybrid Backbone & CLI (`crates/w34-host/src/{cache,model,main}.rs`):**
   - `KvPool`, `StateBundlePool`, `StateTx`, `Qwen3_5Engine` (`TP=1` and `TP=2`), `CUDAGraphRunner` (**Gate #3** bit-exact graph replay under metadata mutation verified), and `w34` CLI (`generate`, `bench`, `dump-diff`, `greedy-diff`, `profile-diff`).
6. **Phase 3 — Continuous Batching Scheduler, `MambaRadixCache` (`SGLang` Donor), `Qwen3Parser`, & OpenAI HTTP/SSE Server (`crates/w34-host/src/{radix,scheduler,parser,server}.rs`):**
   - `MambaRadixCache` + `ReqRadixState` (`crates/w34-host/src/radix.rs`): Dual-lock (`full_lock_ref >= mamba_lock_ref`) and dual-LRU (`full_lru` + `mamba_lru`) radix tree over `P = 16` logical KV pages and `C = 64` GDN `StateBundle` checkpoints with tombstone cleanup, child-split invariants, and `extra_buffer` (`ping_pong_slots: [i32; 2]`) checkpoint donation.
   - **Decisive Gate Test #5 (`test_gate5_warm_prefix_hit_speculation_and_cross_boundary_rejection`)**: Verified 0-ULP `conv_state` (`max_abs_diff = 0.0`) and `ssm_state` (`max_abs_diff = 0.0`) parity and 100% token parity against a cold non-speculative reference across warm `C = 64` prefix restore, `1 + k` speculative verification, cross-boundary (`t = 128`) rejection + retry, and checkpoint donation.
   - **Warm Prefix TTFT Drop**: Verified **95.2% warm TTFT reduction** (`145.68 ms` cold → `7.05 ms` warm, `192` cached prefix tokens) with 100% generated token parity.
   - `SchedulerV1` (`crates/w34-host/src/scheduler.rs`): Token-budget continuous batching (`max_num_batched_tokens`, `max_num_seqs`, `prefill_chunk_size = 64`) with decode-priority scheduling, chunked suffix prefill, and multi-request `step_batch_slices` execution.
   - `SamplerV1` + `TokenBitmaskConstraint` + `Qwen3Parser` (`crates/w34-host/src/parser.rs`): 9-step sampling chain with speculative rollback parity, plus streaming/complete `<think>...</think>` reasoning and `<tool_call>` (Qwen3-XML `<function=...><parameter=...>` and Hermes JSON) extraction.
   - OpenAI HTTP/1.1 + SSE Server (`crates/w34-host/src/server.rs`, `w34 serve` & `w34 chat`): `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`, `/health`, and `/metrics` (Prometheus exposition format including `w34_radix_cache_hit_rate`).

---

## Phase 4 — Verified & Hardware-Aligned (`sm_86`)

All 7 Phase 4 integration gates (`phase4_gates.rs`) pass in **33.22 s**, and the
full `mise run check` suite (formatting, CPU/CUDA clippy, and all 55 release
tests across Phases 0–4) passes in **55.13 s**.

### Verified Speculative & Continuous Batching Throughput (`1–2× RTX 3090 Ti`)

| Workload | Baseline (Plain `CUDA Graph`) | `Qwen3_5MTP` (`k=3`) | `DFlash2` (`M=16` on 4B, `M=8` on 27B) | Speedup vs Plain |
| --- | --- | --- | --- | --- |
| `Qwen3.5-4B-W4A16-GPTQ` (`TP=1`, `w34 bench`, `gen=128`) | `186.29 tok/s` (`5.37 ms`) | **`331.05 tok/s`** (`3.02 ms`, $\tau=3.97\times$, `11.99 ms/r`) | **`1,007.58 tok/s`** (`0.99 ms`, $\tau=12.70\times$, `12.60 ms/r`) | **`1.78×` (`MTP`) / `5.41×` (`DFlash2`)** |
| `Qwen3.5-4B-W4A16-GPTQ` (`TP=2`, `w34 bench`, `gen=128`) | `266.10 tok/s` (`3.76 ms`) | **`488.59 tok/s`** (`2.05 ms`, $\tau=3.97\times$, `8.12 ms/r`) | **`1,212.83 tok/s`** (`0.82 ms`, $\tau=12.70\times$, `10.47 ms/r`) | **`1.84×` (`MTP`) / `4.56×` (`DFlash2`)** |
| `Qwen3.5-4B-W4A16-GPTQ` (`TP=2`, `w34 bench --stage radix`, `128` cached) | Cold `34.23 ms` → Warm **`4.71 ms`** (`7.27×`, `-86.2%`) | Warm **`492.54 tok/s`** (`5.20 ms` TTFT) | Warm **`1,089.73 tok/s`** (`4.99 ms` TTFT, `7.08×`, `-85.9%`) | **100% lossless warm parity** |
| `Qwen3.5-4B-W4A16-GPTQ` (`w34 bench --stage scheduler`, `B=8`) | `TP=1`: **`646.62 tok/s` decode (`2,263.19 tok/s` total)** / `TP=2`: **`865.92 tok/s` decode (`3,030.72 tok/s` total)** | — | — | **`3.47×` (`TP=1`) / `3.25×` (`TP=2`) batch scaling** |
| `Qwen3.5-9B-W4A16-GPTQ` (`w34 bench --stage scheduler`, `B=8`) | `TP=1`: **`424.48 tok/s` decode (`1,485.67 tok/s` total)** / `TP=2`: **`598.55 tok/s` decode (`2,094.94 tok/s` total)** | — | — | **`3.57×` (`TP=1`) / `3.29×` (`TP=2`) batch scaling** |
| `Qwen3.8-27B-W4A16-GPTQ` (`w34 bench --stage scheduler`, `B=8`) | `TP=1`: **`168.72 tok/s` decode (`590.54 tok/s` total)** / `TP=2`: **`249.35 tok/s` decode (`872.71 tok/s` total)** | — | — | **`3.51×` (`TP=1`) / `3.24×` (`TP=2`) batch scaling** |
| `Qwen3.8-27B-W4A16-GPTQ` (`TP=2`, `w34 bench`, `gen=128`) | `76.89 tok/s` (`13.00 ms`) | — | **`116.32 tok/s`** (`8.60 ms`, **`22.75 ms/r`** vs `SGLang` `24.90 ms/r`, Warm **`233.68 tok/s`**) | **`1.51×` cold / `3.04×` warm** |
| `Qwen3.8-27B-W4A16-GPTQ` (`TP=2`, `w34 generate` coding prompt, `gen=128`) | `76.89 tok/s` (`13.00 ms`) | — | **`163.44 tok/s`** (`6.12 ms`) | **`2.13×`** |

### End-to-End Multi-Stage Verification & Profiling Tooling (`w34` CLI & `.mise.toml`)

- **`w34 dump-diff` (`mise run oracle-dump`):** Captures target layer hidden states, `last_final_hidden_f16`, `tap_aux_bf16`, `drafter_hidden`, and `draft_proposal_ids` across plain, `--speculative mtp`, and `--speculative dflash2` runs; reports `max_abs_diff`, `rel_l2`, and `cosine_sim` against golden safetensors or `--compare-tp2` (`TP=1 vs TP=2` target `max_rel_l2 = 0.0023`, `MTP` drafter `rel_l2 = 0.0020` with `3/3` exact draft match, `DFlash2` drafter `cosine = 0.9849` with `9/15` prefix and `12/15` exact draft match) and fails with non-zero exit code on regression.
- **`w34 greedy-diff` (`mise run oracle-greedy`):** Reports per-prompt and aggregate speculative acceptance telemetry (`rounds`, `accepted`, $\tau = 1 + \bar{a}$), supports `--verify-lossless` (verifying 100% bit-for-bit token identity between speculative and plain target greedy) and `--verify-radix-warm` (verifying 100% bit-for-bit token identity between cold and warm `MambaRadixCache` execution), and fails with non-zero exit code on any adjudication/lossless/radix failure.
- **`w34 bench --stage decode|radix|scheduler|all` (`mise run bench-all`):** Benchmarks Stage 2 single-sequence decode + Stage 4 speculative telemetry (`verify_rounds`, `mean_accepted_drafts`, `mean_tokens_per_round`, `ms_per_verify_round`), Stage 3 `MambaRadixCache` cold-vs-warm TTFT speedup + proposer KV restore + lossless parity, and Stage 3 `SchedulerV1` continuous batching (`--batch-size`, with live `sched.radix` LRU eviction on page/state pool pressure).
- **`w34 profile-diff --live` (`mise run profile-live`):** Runs `nsys profile --trace=cuda --cuda-graph-trace=node`, exports SQLite, historically estimated verification windows using kernel-name heuristics plus `w34 bench` JSON telemetry (diagnostic only; use the new launch-correlated NVTX mode for verified measurement boundaries), classifies target vs `MTP`/`DFlash2` (`BF16`) proposer spans into the 7 canonical categories (`marlin_gemm`, `full_attn`, `gdn_ssm`, `norms_rope_act`, `tp_comm`, `spec_decode`, `host_gap`), and renders the side-by-side scoreboard against `vLLM 0.31.0` and `SGLang v0.5.19` presets in `tools/oracle/goldens/oracle_baselines.json`.

7. **Phase 4 — Speculative Decoding (`RejectionSampler`, Native `Qwen3_5MTP`, & Native `BF16` `DFlash2`):**
   - **Host & GPU `RejectionSampler` (`src/speculative.rs`, `csrc/ops/op_rejection.cu`):** Consecutive greedy and canonical stochastic residual (`min(1, p/q)`, positive-residual recovery) verification over `[anchor, draft_1, ..., draft_k]`, typed `INT32` outputs, padded-request masking, `processed_len` vs `emitted_len` accounting, and atomic `StateTx` + `SamplerV1` + `TokenBitmaskConstraint` tentative-history rollback (`phase4_rejection` 10 tests, `phase4_gpu_rejection` 2 tests).
   - **Native Dense & MoE `Qwen3_5MTP` (`src/mtp.rs`):** `vLLM 0.31.0` `qwen3_5_mtp.py` donor (`pre_fc_norm_hidden`, `pre_fc_norm_embedding`, replicated `fc: 2H -> H`, TP-sharded gated full attention in `W34_KV_NAMESPACE_MTP_FULL_ATTN` + Dense SwiGLU or 256-expert `top-8` MoE + `shared_expert` FFN (`DenseF16`, fused 3D `BF16`/`FP16`, or 2D `W4A16` experts) + final `GemmaRMSNorm` + shared `embed_tokens`/`lm_head`) with batched committed-row prefill ingest (`sample_last = false` skips `lm_head`/sampler during ingest), zero-copy `M=1` pointer aliasing for `fc` input and `[QGate, K, V]` projection, pre-uploaded KV page table, and packed single-upload batch metadata (`worst max_abs = 0.03125`, `rel_l2 = 0.001158` across all 7 live rows of `tools/oracle/goldens/vllm_mtp_4b/call_{0,1,2}.safetensors`, plus `draft_exact = 3/3`, `drafter_cosine = 0.999824` `TP=1 == TP=2` parity on `35B-A3B` MoE, `phase4_mtp` 2 tests).
   - **Native `BF16` `DFlash2` (`src/dflash.rs`, `csrc/ops/op_dflash2.cu`, `csrc/ops/op_paged_attn.cu`):** `SGLang v0.5.19` `DFlash2DraftModel` / `DFlashWorkerV2` donor in native `BF16` (`4B M=16` taps `[1, 9, 16, 23, 28]` and `27B M=8` taps `[5, 19, 33, 47, 61]` with `window=2048`), direct `FP16 -> BF16` local per-rank `TapPlan` capture fused into `w34_tp2_all_reduce_residual_gemma_rms_norm_f16` (`TP=2`) and `w34_fused_add_gemma_rms_norm_tap_f16` (`TP=1`) (zero `FP32` intermediate and zero standalone tap-capture launches), `W34_KV_NAMESPACE_DFLASH_DRAFT_CTX` `BF16` paged KV with fused `QK` norm + `RoPE` + `BF16` paged KV cache write + contiguous `Q` output (`w34_dflash_qk_norm_rope_kv_write_bf16_f32_table`, eliminating all 25 per-round `w34_copy_cols_bf16` and 10 per-round `w34_kv_cache_write_strided_bf16` launches), GPU-resident greedy path selection (`w34_dflash_select_greedy_bf16`, eliminating the 245 KiB `topk_ids` + `edges` D2H/H2D/Sync round-trip on `TP=1` and shrinking `TP=2` D2H to 64 bytes), 1-warp (32-thread) `RoPE` and selector edge-scoring reductions, bank-conflict-free transposed-smem `w34_topk_rows_f16`, and FlashInfer `sm_86` FA2 `BF16` paged attention (`CTA_TILE_KV=16` at HD=256 / `64` at HD=128, register-preloaded `Q` fragments, sliding-window `start_base` skip). Passes **bit-exact (`relL2 = 0.0, maxabs = 0.0`)** full-model 5-layer hidden parity (`actual_sglang_full_model_hidden_parity` in `phase4_dflash.rs`), bit-exact 5-layer attention parity (`phase4_bf16_attention.rs`), and `sglang_dflash2_fixtures.safetensors` operator dump-diff parity (`phase4_gates.rs`, `phase4_bf16_ops.rs`).
   - **Speculative Engine, CUDA-Graphed Batched Target Verification, Warm Prefix Restore, & CLI/Server Integration (`src/proposer.rs`, `src/model.rs`, `src/server.rs`, `src/main.rs`):** Fixed-shape `M_verify = K + 1` per-rank CUDA Graph capture & replay (`SpecVerifyGraphIo`, `ensure_spec_verify_graph`, `step_spec_verify_graph`) cutting `host_gap` to **`0.35 ms/round`** (`2.42×` faster than `SGLang v0.5.19`) and reaching **`960.65 tok/s` (`1.04 ms/tok`)** on `4B TP=1 + DFlash2`, **`332.04 tok/s`** on `4B TP=1 + MTP`, and **`425.45 tok/s` (`2.35 ms/tok`)** on `35B-A3B MoE TP=2 + MTP`; batched `lm_head` + `w34_sampler_greedy_top5_f16` + `all_gather_top1_f32_i32` across all `1 + k` verification rows in `forward_selected_rows`; fused `TP=2` all-reduce + residual add + next-layer/final `GemmaRMSNorm` + optional `BF16` tap write (`w34_tp2_all_reduce_residual_gemma_rms_norm_f16` with `uint4` shared-memory peer staging and local L2 reads, cutting `norms_rope_act` to `0.37 ms/round` on `27B TP=2` — **`3.12×` faster than `SGLang v0.5.19`** — and eliminating `128` kernel launches per round on `27B` and `72` on `4B`); zero-copy row-strided `QKV` / `QKVZ` / `BA` consumption (`w34_fused_qk_rmsnorm_rope_gate_strided_f16`, `w34_kv_cache_write_strided`, `w34_gdn_causal_conv1d_strided_f16`, `w34_gdn_recurrent_strided_f16`, `w34_rms_norm_gated_silu_strided_f16`, eliminating all `112` per-round `w34_split_cols_{2,3}_f16` launches on 27B and `56` on 4B); batch-size-invariant 1-launch warp-reduction `w34_small_n_gemm_f16` for GDN `in_proj_ba` (eliminating `720` per-round cuBLAS split-K GEMV launches on 27B while preserving 0-ULP `ssm_state` parity across all `M`); `float4` vectorized `op_gdn.cu` `ssm_state` loads/stores with deduplicated `last_spec_slot == slot_out` final writes (`gdn_ssm = 1.34 ms/round` on `27B TP=2 + DFlash2`, **`1.75×` faster than `SGLang v0.5.19`**); persistent `batch_meta_arena` (zero per-step `cudaMalloc`/`cudaFree`); direct `last_final_hidden_f16` write from final `GemmaRMSNorm`; pipelined `DFlash2Rank::ingest_no_sync` + `forward` in a single `TP=2` `std::thread::scope` (`ingest_and_propose`); and zero-D2H GPU-resident `DFlash2` KV truncation on warm `MambaRadixCache` hits. All eight integration gates in `crates/w34-host/tests/phase4_gates.rs` pass together (`4B TP=1` & `TP=2` `MTP`/`DFlash2`, `27B TP=2` `M=8` `DFlash2`, speculative verify CUDA Graph vs eager greedy + sampled parity, processed stochastic/grammar, HTTP JSON/SSE, and `sglang_dflash2_fixtures.safetensors` dump-diff parity).

## Phase 5 — Paged `TQ4` (`tq4` / `tq4_all`) & `Q8_0`, Multimodal Vision (`Qwen3_VisionTransformer`), & Hybrid MoE (`Qwen3.6-35B-A3B`)

All 7 Phase 5 integration gates (`phase5_gates.rs`) pass in **2.63 s**, and the full `mise run check` suite (formatting, CPU/CUDA clippy, and all 61 release tests across 13 test suites covering Phases 0–5) passes in **66.00 s**.

### Verified Phase 5 Throughput & Compression Scoreboard (`1–2× RTX 3090 Ti`)

| Workload / Model | KV Policy (`Bytes/Token`) | `TP=1` Decode (`ITL`) | `TP=2` Decode (`ITL`) | Speculative Decode (`CUDA_GRAPH=1`) | Notes |
| --- | --- | --- | --- | --- | --- |
| `Qwen3.8-27B-W4A16-GPTQ` | `tq4` (`21,376 B/tok`, **`3.07×`**) | **`48.82 tok/s`** (`20.48 ms` @ **`262k`** `16,384` pages) | **`78.73 tok/s`** (`12.70 ms`) | **`176.21 tok/s`** (`TP=2 + DFlash2 M=8`) / **`114.68 tok/s`** (`TP=2 + MTP K=3`) | Fits full **`262,144`-token** context (`5,344 MiB` KV + `1.0 GiB` headroom) on **1× 24 GB 3090 Ti**; `12/12` `greedy-diff` (`672/1536`, matching `fp16` `673/1536`) |
| `Qwen3.8-27B-W4A16-GPTQ` | `q8_0` (`34,816 B/tok`, **`1.88×`**) | **`48.95 tok/s`** (`20.43 ms`) | **`78.58 tok/s`** (`12.73 ms`) | — | Plain 8-bit (`272 B`/row) for both $K$ and $V$; **`95.3%` (4B) / `97.7%` (9B)** exact 128-token rollout match vs `fp16` |
| `Qwen3.8-27B-W4A16-GPTQ` | `tq4_all` (`16,896 B/tok`, **`3.88×`**) | **`49.40 tok/s`** (`20.24 ms`) | **`78.95 tok/s`** (`12.67 ms`) | — | Uniform rotated LSQ `TQ4` (`4,224 MiB` @ `262k`); 0-ULP page-table permutation invariance + 4096-token fragmented needle retrieval |
| `Qwen3.5-4B-W4A16-GPTQ` | `tq4` (`35,968 B/tok`, **`2.73×`**) | **`183.84 tok/s`** (`5.44 ms`) | **`214.0 tok/s`** (`4.67 ms`) | **`960.65 tok/s`** (`TP=1 + DFlash2 M=16`, `1.04 ms`) / **`332.04 tok/s`** (`TP=1 + MTP K=3`, `3.01 ms`) | **`1.53×` faster than `SGLang v0.5.19`** (`612 tok/s`) on `4B + DFlash2` (`host_gap = 0.35 ms/round`) |
| `Qwen3.6-35B-A3B` (`GPTQ-Int4` MoE) | `tq4` (`8,680 B/tok`, **`2.85×`**) | **`177.80 tok/s`** (`5.62 ms`, `1,087.3 tok/s` prefill) | **`221.59 tok/s`** (`4.51 ms`, **`1.52×` vs `vLLM`**) | **`425.45 tok/s`** (`TP=2 + MTP K=3`, `2.35 ms`, **`1.92×` vs plain**) | 40 layers (`30 GDN + 10 FullAttn`), `256` experts, `top-8` softmax router + `shared_expert`, native MoE `MTP` (`draft_exact = 3/3` `TP=1 == TP=2`) |

### Phase 5 Deliverables

8. **Phase 5.1 — Paged `TQ4` (`tq4` `3.07×`, `tq4_all` `3.88×`) & `Q8_0` (`q8_0` `1.88×`) KV Cache (`crates/w34-kernels/csrc/ops/op_paged_attn.cu`, `crates/w34-host/src/cache.rs`):**
   - **Single-Warp Register `wht256_warp` & Per-128 Least-Squares `TQ4` (`132 B`/row):** Stages `h = 1..16` use intra-warp `__shfl_xor_sync` butterfly shuffles and stages `h = 32, 64, 128` operate in-register across each lane's `v[0..7]` with zero shared memory or block barriers, matching `warp3090`'s Seed-42 Rademacher + WHT-256 transform to 0 ULP, paired with independent 128-element least-squares optimal sub-block scales `d0, d1 = grp_norm * <v, c> / ||c||^2`.
   - **Concurrent 2-Warp `kv_cache_write_kernel` & Fused `paged_attention_f16`:** Warp 0 encodes `K` (`FP16`, `BF16`, `Q8_0`, or `TQ4`) while Warp 1 encodes `V` concurrently; `paged_attention_f16_kernel` fuses `Q' = WHT256 · diag(s) · Q` pre-rotation before the KV loop whenever `layout.transform == W34_KV_TRANSFORM_RADEMACHER_WHT256` (covering both `TQ4` and rotated boundary `Q8_0`), sub-block-grouped `TQ4`/`Q8_0`/`FP16` dot/axpy inside the 4-warp online-softmax loop, and `O = diag(s) · WHT256 · O'` inverse rotation before writing output.
   - **Clean Four-Tier KV Policies (`tq4` default, `tq4_all`, `q8_0`, `fp16`):** Default `tq4` encodes boundary full-attention ordinals `{0, 1, N-2, N-1}` in WHT-256 rotated `Q8_0` (`272 B`/row) and interior layers in rotated LSQ `TQ4` (`132 B`/row), compressing 27B KV by **`3.07×`** (`5,344 MiB` at `262,144` tokens — fitting single-card `27B TP=1` at full `262k` context on 1× 3090 Ti while matching `fp16` `27B` greedy accuracy `672/1536` vs `673/1536`); `tq4_all` compresses every layer to `132 B`/row (**`3.88×`**); and `q8_0` provides unrotated 8-bit (`95.3%–97.7%` exact token match vs `fp16`, **`1.88×`**).
9. **Phase 5.2 — Multimodal Vision (`Qwen3_VisionTransformer`: `crates/w34-kernels/csrc/ops/op_vision.cu`, `crates/w34-host/src/vision.rs`, `crates/w34-host/src/{model,server}.rs`):**
   - **PNG, JPEG, & PPM Decoding + `smart_resize`:** Decodes PNG (`RGB8`/`RGBA8`/palette/grayscale), JPEG, and binary PPM (`P6`) with `16384×16384` / `256 MiB` decompression-bomb limits across OpenAI `data:image/...;base64,...` URIs (enforcing HTTP LFI rejection of `file://` and bare paths) and local CLI `--image <path|url>` files, followed by banker's-rounded (`round_ties_even`) `factor = 32` aspect-preserving resize, PyTorch-aligned (`a = -0.75`) bicubic interpolation, `mean = 0.5 / std = 0.5` normalization, `temporal_patch_size = 2` duplication, and `2×2` spatial-merge patch ordering (`[N_patches, 1536]`).
   - **GPU `Qwen3VisionEncoder` (`op_vision.cu` + `vision.rs`):** 2D bilinear `pos_embed` interpolation, 24/27-layer ViT with `LayerNorm`, 2D axial `RoPE` full attention (`cu_seqlens` per image), `GELU` MLP, `PatchMerger` (`2×2` -> `4 * hidden_size` -> `out_hidden_size`), and `w34_embedding_splice_f16` into `<|image_pad|>` token positions.
   - **3D `mRoPE` & Content-Hashed `MambaRadixCache`:** `compute_mrope_positions` (`[3, seq_len]` `(T, H, W)` planes + `mrope_delta` consumed by `w34_fused_qk_rmsnorm_rope_gate_strided_f16`), `compute_multimodal_radix_key` (mixing 64-bit FNV-1a image content hash into `<|image_pad|>` radix keys so identical images hit warm `MambaRadixCache` checkpoints while distinct images never alias), and OpenAI `/v1/chat/completions` `image_url` support.
10. **Phase 5.3 — `Qwen3.5` / `3.6` Hybrid MoE (`Qwen3_5MoeForConditionalGeneration`: `crates/w34-kernels/csrc/ops/op_moe.cu`, `crates/w34-host/src/{loader,model}.rs`):**
    - **Top-K Softmax/Sigmoid Router (`w34_moe_topk_router_f16`):** Supports up to `1024` experts and `top_k <= 32` with optional `norm_topk_prob` renormalization and deterministic tie-breaking.
    - **Two-Stage Fused `W4A16` (`U4B8NMajor`) & `FP16` MoE Kernels (`w34_moe_w4a16_gate_up_silu_f16`, `w34_moe_w4a16_down_accum_f16`, `w34_moe_f16_gate_up_silu`, `w34_moe_f16_down_accum`):** Stage 1 fuses routed `gate_proj + up_proj + SiLU(gate) * up` per `(token, slot)` pair; Stage 2 fuses routed `down_proj` weighted accumulation across `0..top_k` with `shared_expert * sigmoid(shared_expert_gate)` in a single deterministic write without `atomicAdd`.
    - **End-to-End `TP=1` & `TP=2` Engine Support:** Shards both routed experts (`moe_intermediate_size / tp_size`) and `shared_expert` (`shared_expert_intermediate_size / tp_size`) across TP ranks while keeping `router_gate` and `shared_expert_gate` replicated for identical routing decisions across ranks without extra collectives, and launches `TP=2` CUDA Graphs from a single host thread without per-token thread spawning (`5.63 ms/round` vs `vLLM` `6.84 ms/round` in live `nsys` `profile-diff`).
    - **Phase 5 CLI, Benchmark, Oracle, & `nsys` Tooling (`crates/w34-host/src/main.rs`, `tools/oracle/profile_diff.py`, `.mise.toml`):** Supports `--image <path|url>` across `generate`, `chat`, `dump-diff` (`dump_multimodal_prompt` with `TP=1 vs TP=2` parity across `visual_embeddings`, spliced `embed_tokens`, and 3D `mRoPE`), and `bench --stage vision|all` (cold vs warm multimodal `MambaRadixCache` TTFT speedup + distinct-image content-hash isolation), plus `vllm_27b_tp1_tq4_decode`, `vllm_35b_moe_tp1_decode`, and `vllm_35b_moe_tp2_decode` presets in `oracle_baselines.json`.

---

## Phase 6 Working State (`docs/ROADMAP.md` §4 Phase 6)

### Continuation 21 — native residual-norm correction and measured profiling boundaries

- Found and runtime-confirmed another genuine donor mismatch: installed vLLM `GemmaRMSNorm` normalizes the **unrounded FP32 sum** of FP16 input and residual, while storing the residual itself in FP16. Both TP1 and fused TP2 w34 kernels now preserve that distinction. The exact native-bit regression in `gemma_residual_rounding.rs` failed before the fix and passes afterward, including persistent residual and BF16 tap checks. No target storage dtype was changed. The existing fused-norm fixture was a false oracle that rounded too early. `CUDA_VISIBLE_DEVICES=1 python tools/oracle/vllm_dump.py --refresh-fused-gemma` recaptured actual installed vLLM CUDA/module-dispatch output with bitwise agreement: **8,368/40,960** normalized output elements changed, residual bits did not. All other **42 tensors** and all tolerances were preserved; scoped native provenance was added.
- Also confirmed actual target `MRotaryEmbedding.cos_sin_cache` is **FP16** for the native 4B config (head 256, rotary 64, interleaved `[11,11,10]`). w34 incorrectly rotated with unrounded FP32 coefficients. The target kernel now rounds coefficients through FP16; the native exact-bit M=1/M=16 regression failed before and passes after, and existing phase1 fixtures still pass unchanged. This is a second real target fix, not an end-to-end parity claim. The post-fix **full check passed with both GPUs free: 125 Rust tests and 28 oracle host tests in 185.51 s**, `/tmp/w34-phase6-rope-final-check.log`. Subsequent canonical replay in `/tmp/w34-vl-target-rope-fix/` still passes 4B PNG/JPEG (**0.0400652 / 0.0424111**) but fails 27B (**0.1900746 / 0.1270071**) and MoE (**0.2232446 / 0.2054107**) final-norm gates; all six retain 8/8 greedy IDs. No tolerances changed.
- Canonical six-case replay after the residual-norm correction (before target RoPE correction): `/tmp/w34-vl-gemma-fix/`. **4B PNG/JPEG pass**, final-norm relL2 **0.0407854 / 0.0422014**; **27B PNG/JPEG still fail**, **0.1899882 / 0.1265059**; **MoE PNG/JPEG still fail**, **0.2162264 / 0.2116164**. All six retain **8/8 exact greedy tokens**. The correction is real but does not resolve the large-model tensor gate.
- `mise run check-quick` passed (fmt and CPU/CUDA Clippy), log `/tmp/w34-phase6-gemma-quick.log`. The prior focused command passed **22 Rust tests**, including three live Axum tests, in `/tmp/w34-phase6-vision-focused.log`. After freeing both GPUs, **`CUDA_VISIBLE_DEVICES=0,1 mise run check` passed in 184.62 s: 123 Rust tests plus 27 oracle host tests**, `/tmp/w34-phase6-two-gpu-final-check.log`. This includes TP2 graph/collective gates, GQA/MMA/split-KV parity, grouped-MoE/TP2 parity, TP1/TP2 GPU sampling, and live MTP/DFlash/MoE HTTP constraint tests. The earlier restricted-device failure was environmental. A subsequent review identified missing terminal-token legacy logprobs; that fix now passes **4 live Axum + 6 completion tests**, including forced EOS and custom stop tokens in stream/nonstream responses (`/tmp/w34-terminal-after.log`). Hidden terminal tokens receive logprob-only SSE records with empty text and aligned offsets/counts. The post-fix full-check rerun is tracked below.
- Added launch-correlated NVTX measurement with process-aware CUDA runtime correlation (including graph nodes, worker-thread launches, and asynchronous completion); **10 profiling CPU tests pass**, including nullable speculative-round metadata falling back to an explicit measured-round count and rejecting non-object metadata. Direct test-file invocation now runs the correlated tests too. Command: `python tools/oracle/profile_diff.py --nvtx-correlated --w34 /tmp/w34-phase6-marked-4b/trace.sqlite --metadata /tmp/w34-phase6-marked-4b/stdout.json`. It measures **4,665 kernels over 10 rounds**, **128.038473 ms summed GPU time**, **131.681499 ms CPU decode wall time**, with `full_attn=0.920680` and `spec_decode=5.247384` summed GPU ms/round. Workload alignment remains unverified, so **gate_eligible=false**; this is not an oracle speedup claim. The trace predates the residual-norm correction.
- `mise run oracle-check` passed **27 host tests** (7 vision capture, 2 native-norm fixture, 6 needle, 9 profile, 3 vision-matrix). Review fixes now clear stale needle summaries before builds, record launch/type failures without aborting the matrix, retain failing vision tensor metrics, and limit insecure-deserialization environment settings to the trusted local capture subprocess. The decoder preserves **64 MiB local input / 16 MiB HTTP ingress** limits; its large-local-input regression failed before the correction and passes afterward.
- Stopped old-binary needle task `t-mvu0fv84` because the target arithmetic correction makes it unsuitable for final-tree validation. Its retained 32k/64k/128k TQ4 passes are historical only; the 262k case was interrupted and is **not a pass**. After the successful unrestricted full check, restarted the **entire eight-case matrix** with both target corrections: persistent task **`t-mvu3ybzf`**, GPU0, `/tmp/w34-phase6-needle-native-fp16/{summary.json,...}`, log `/tmp/w34-phase6-needle-native-fp16.log`. Per-case timeout is 10,800 s to avoid an artificial one-hour cutoff at 262k. GPU1 remains available for further parity work.
- Pinned SGLang donor preflight now runs successfully inside `lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`, without network/GPU access: `/tmp/aligned-oracle/plan.json` records exact cached target/draft commits, tokenizer/source hashes and deterministic prompt IDs. `test_aligned_sglang_bench.py` passes **5 additional CPU tests**. This is deliberately **preflight-only**, not a serving/profile runner: worker-boundary instrumentation, reset semantics and actual donor timing capture remain unimplemented. `--profile` refuses to fabricate results.
- Review triage: fixed concrete decoder-limit, terminal-token legacy logprob and harness-reporting issues. Kept strict JPEG warning rejection as an intentional fail-closed policy. Deferred unchanged optional authentication/tokenizer-budget hardening and nullable custom-boolean compatibility; rejected a fractional KV reservation cap because it would prohibit intended full-context requests. No claim that exact profiling boundaries alone establish aligned donor performance. No commit while full verification and release gates remain open.

### Vision numerical isolation and fixes (earlier checkpoint; superseded above where noted)

- **4B real PNG and JPEG now pass the complete canonical gate** (three tensor boundaries, exact mRoPE, exact prompt IDs, **8/8 exact greedy tokens**), without changing the **0.05** tolerance. Results in `/tmp/w34-vl-native-fp16-fix/`: PNG `visual_relL2=0.004597`, `final_norm_relL2=0.040675`; JPEG `visual_relL2=0.005254`, `final_norm_relL2=0.042365`. The native vLLM captures were retained unchanged.
- **27B and MoE remain failing:** final-norm relL2 is **0.190430 / 0.122358** (27B PNG/JPEG) and **0.218398 / 0.207304** (35B-A3B PNG/JPEG). All retain exact eight-token greedy outputs. These are not passing vision gates.
- Fixed independently reproduced numerical mismatches: (1) bias sums materialize in FP16 before vision residual/GELU/RoPE operations; (2) vision RoPE coefficients use the native FP16 cache precision; (3) positional interpolation matches installed Triton’s FP16/FMA rounding sequence exactly across 72 explicit values and five grid fingerprints; (4) vision Linear uses FP32 GEMM accumulation plus bias **before** its FP16 output conversion (`beta=1` with bias-initialized output), with mandatory warmup in `prepare_execution_plan`. Regression example: old split GEMM/bias yielded **0.99951171875**, native/new path **1.0**. Biased GEMM adds a bias broadcast launch; final performance measurement remains required.
- **JPEG decode was independently wrong at the pixel boundary:** the original decoder differed from Pillow in 25,318/1,843,200 temporally duplicated normalized FP16 patch elements. The pinned statically vendored `turbojpeg-sys=1.1.1` accurate-IDCT/fancy-upsampling path now matches Pillow’s full RGB SHA-256 for the JPEG fixture exactly, while retaining PNG behavior and enforcing 16 MiB input, 16 MP, 16,384-per-axis, 128 MiB native intermediate-memory, and 100 progressive-scan caps. CPU JPEG/transport regressions passed **8/8**. Building now requires CMake and a C compiler, not a system JPEG library.
- Diagnostic-only native vLLM runs with w34’s exact visual embeddings isolate target drift: **4B final_norm relL2=0.003980**; **27B=0.083104**, with early-layer errors below 0.001 that amplify in late layers. Therefore vision drift explains much of the 4B failure, but not all 27B drift. `--diagnostic-override-visual` captures are explicitly non-canonical and cannot pass `vision_gate`. A uniform FP16-normalized-Q/K experiment only reduced the 27B identical-input error to **0.073253** and canonical error to **0.186870**; it was **reverted** rather than changing target arithmetic for a marginal gain. Rebuild before further GPU work: the last experimental example binary may still contain that reverted experiment.
- Added NVTX `w34.bench.measured`, nested `w34.decode`, and `w34.proposer` scopes, trace retention, and strict explicit-GPU-window parsing with fail-closed CPU tests (**5/5**). Historical heuristic reports are now labeled diagnostic/non-gating. Automatic runtime-correlation mapping and aligned donor workload manifests are **not implemented**, so performance gates remain open.
- Legacy completion validation/Unicode offset review fixes passed **6/6** focused tests; the extended real Axum suite passed **3/3 in 17.62 s**, including legacy nonstream/SSE logprob arrays and matching offsets. Diagnostic throughput was **786.61 / 908.15 generated tok/s** at 8/16 clients; this is not a matched donor benchmark.
- The long-context task `t-mvu0fv84` still runs the **pre-vision-change binary** on GPU 0; its `tq4` 32k/64k/128k cases passed. New changes do not modify text execution arithmetic (the Q/K experiment is reverted), but final check/review/commit remain pending. A quick-check attempt caught missing NVTX safety comments and an unnecessary JPEG cast, both fixed; the current complete tree still needs re-verification. No completion claim or new commit.

### Real-image and long-context oracle checkpoint (earlier measurements; superseded above where noted)

- Serving checkpoint below is committed as **`5bb3742`**. New harnesses are `tools/oracle/{vllm_vl_dump,vision_matrix,needle_matrix}.py` and `crates/w34-host/examples/{vision_gate,needle_gate}.rs`. `vision_gate` verifies source-image/config SHA-256, exact processed prompt IDs, tensor shapes/finiteness, native mRoPE positions, and eight-token greedy output. It uses the HTTP preprocessing bounds (`1024..1048576` pixels); the older CLI helper has different bounds, so comparisons must not assume these paths are interchangeable for resized inputs.
- **Actual vLLM 0.31.0 captures, not synthetic fixtures:** `CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python tools/oracle/vision_matrix.py --out-dir /tmp/w34-vl-full-matrix` completed all six captures and returned **1**, correctly rejecting full-model tensor drift. Each real PNG/JPEG case has **exact prompt IDs, exact mRoPE, and 8/8 exact greedy tokens**. Relative-L2 values below compare all 306 prefill rows; the separate greedy run uses chunk size 64. The existing **0.05** tensor tolerance is unchanged.

  | Model | Format | Visual embedding relL2 | Final-norm relL2 | Full gate |
  |---|---|---:|---:|---|
  | 4B | PNG | 0.011720 | 0.077648 | **FAIL** |
  | 4B | JPEG | 0.018826 | 0.119220 | **FAIL** |
  | 27B | PNG | 0.003764 | 0.183770 | **FAIL** |
  | 27B | JPEG | 0.010490 | 0.254472 | **FAIL** |
  | 35B-A3B | PNG | 0.007574 | 0.237131 | **FAIL** |
  | 35B-A3B | JPEG | 0.019453 | 0.274170 | **FAIL** |

- Captures and per-case JSON/logs are in `/tmp/w34-vl-full-matrix/`. Read-only source audit validates the comparison and identifies a candidate numerical difference: vLLM prefill materializes normalized Q/K in FP16 before its chunk GDN algorithm; w34 retains normalized Q/K in FP32 in its recurrent implementation. This is **a hypothesis to isolate**, not a demonstrated cause or an accepted excuse for the failed gate.
- **Needle progress:** `CUDA_VISIBLE_DEVICES=0 python3 tools/oracle/needle_matrix.py --out-dir /tmp/w34-phase6-needle-matrix` is running the eight-case, single-GPU 27B matrix. At this checkpoint, `tq4` **32768, 65536, and 131072** windows passed exact retrieval of `73918462` at midpoint. The window includes 32 output and four reserved engine tokens; exact prompt IDs are retained. **262144 and all `tq4_all` results remain pending.** Persistent task `t-mvu0fv84`; log `/tmp/w34-phase6-needle-matrix.log`. This is one depth and one fixed secret per policy/window, not a broad statistical retrieval benchmark.
- Latest live profile diagnostics: 4B DFlash **983.76 tok/s**, `spec_decode=4.4293 ms/round`; 27B TP2 DFlash **138.28 tok/s**, `spec_decode=5.0338 ms/round`, `full_attn=1.6871 ms/round`. Outputs are `/tmp/w34-phase6-{4b,27b}-dflash-profile.json`. **These do not establish aligned donor speedups/regressions:** stored baselines lack exact prompt/checkpoint/KV provenance and `profile_diff.py` identifies steady-state rounds heuristically, not from explicit timestamp boundaries. Fresh retained, marked traces and common workload manifests are required before performance gates can close; earlier claims of exact profile-window slicing below are too strong.
- Legacy `/v1/completions` integer `logprobs` support has been added in the working tree with CPU regressions (integer 0 includes selected-token probability; null/omitted disables it). Review identified Unicode-fragment offset accounting and malformed-JSON helper validation to fix before verification. No CUDA worker/inference changes are part of this wire-format addition.
- New harness host tests initially passed **10/10**; the legacy completion/ChatML/constraint CPU suites passed **21/21**. CUDA clippy subsequently caught two collapsible conditionals in the new HTTP code; these were fixed, but final re-verification remains pending. An initial review of this checkpoint has completed. **No fresh full `mise run check` or commit yet**: the needle run occupies GPU 0, full-model vision gates fail, and review fixes still need verification. The earlier 109-test result applies to `5bb3742`, not this working tree.

### Axum/Tokio serving checkpoint (verified; Phase 6 exit gates still pending)

- HTTP now uses **Axum 0.8 + Tokio**, not the hand-written socket parser/thread pool. Production `from_loader` initializes, runs, and destroys the engine on the named `w34-cuda` OS thread. HTTP state exposes bounded Tokio MPSC submission and atomic telemetry, **not `Mutex<Qwen3_5Engine>`**. Only bounded CPU preprocessing and thread joining use `spawn_blocking`; CUDA inference does not.
- The worker drives `SchedulerV1`, uses padded decode graphs for `B={1,2,4,8,16}`, streams committed tokens live, and cancels dropped/slow receivers without blocking peers. Text and multimodal paths use GPU sampling/grammar masks; multimodal prefill is incremental with request-local visual embeddings and mRoPE decode deltas. A lone eligible text request uses MTP/DFlash2; bursts use batched target decoding. The proposer remains single-request, so requests arriving during an active speculative request wait until it finishes. **Concurrent speculative batching is not claimed.**
- HTTP supports stop-string suffix holdback, stop-token IDs, bias/minimum-token/EOS controls, chat logprobs, and optional SSE usage. Generation is capped at 4096 output tokens; stop-ID/bias collections are capped at 256 entries. Admission reserves the complete KV budget before starting a request and queues newcomers that do not fit. This intentionally permits a single full-context request rather than imposing a per-request fraction that would prohibit the required long-context workload. FIFO queueing can serialize large requests; tenant fairness/authentication remain separate policy work.
- Reproduced and fixed three isolation regressions: KV pressure previously rejected a newcomer with `KV pool exhausted`; invalid image geometry aborted a valid text peer; an unsatisfiable grammar's GPU sampler status aborted its peers. The regressions now pass. Vision geometry is validated before admission, and per-row sampler/grammar errors cancel only the affected request. Scheduled speculative error paths now return temporary GDN state slots.
- **Command-backed verification:** `export PATH="/opt/cuda/bin:$PATH"; CUDA_VISIBLE_DEVICES=0,1 mise run check` exited **0** in **183.81 s** (`/tmp/w34-axum-check.log`): formatting, CPU/CUDA clippy, and **107 release tests, zero failures**. This includes `phase6_axum_serving` **3/3**, `phase6_decode_graphs` **1/1** (all five buckets, TP=1/2 eager/graph token and state comparison), `phase6_gpu_sampler` **2/2**, and `phase6_http_constraints` **4/4** (including plain/multimodal mandatory constraints). Legacy custom accept-loop tests were removed with that implementation; the real-Axum test covers stalled-body isolation, live-before-complete SSE, disconnect cancellation, backpressure, concurrent requests, and explicit shutdown.
- The focused serving command `CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda --test phase6_axum_serving -- --test-threads=1 --nocapture` passed **3/3 in 15.39 s**. Its 4B loopback workload measured **899.47 tok/s at 8 clients** and **1057.35 tok/s at 16 clients**, including prefill and HTTP. These are diagnostic workloads, **not oracle-aligned performance exit gates**.
- **Review status:** initial and one follow-up review completed. Fixed the important TP=2 batched-graph issue with a three-phase host transaction: upload all ranks, capture all ranks, then launch all ranks. Capture records nodes without executing collectives. `coordinated.rs` fault-injection tests cover failures on either rank in preparation/capture and distinguish partial launch. The real TP=2 graph test injects a rank-1 upload failure during both cold capture and warm replay, verifies no GDN state mutation, and successfully retries. Safe preparation errors propagate without globally disabling graphs or replaying mutable state. An asymmetric launch failure after a peer has started is explicitly **process-fatal** (`abort`): CUDA cannot safely cancel the spinning peer, so a supervisor must restart instead of attempting unsafe recovery. The repeated KV-share finding is not accepted as a correctness blocker: active budgets are reserved, newcomers wait, output is capped, and the failing-then-passing pressure test proves peers are not killed; an arbitrary fractional cap would break full-context operation.
- **Final re-verification:** `CUDA_VISIBLE_DEVICES=0,1 mise run check` exited **0** in **199.95 s**, with **109 release tests, zero failures**, formatting and CPU/CUDA clippy (`/tmp/w34-axum-final-check.log`). Stop-string requests now retain cached-prefix accounting through request-local atomic metadata; shutdown drains admitted generations before stopping the worker (30-second HTTP drain deadline). Both are exercised in `phase6_axum_serving`. Out-of-vocabulary stop/bias IDs are rejected in HTTP preprocessing.
- Deferred suggestions: no API-key/multitenant policy yet (default bind is loopback); CPU preprocessing has byte/concurrency bounds but no hard grammar-compile deadline; legacy completions integer `logprobs` still needs its own wire format; prompt-copy optimization and richer timeout diagnostics remain optional polish. No additional review loop is planned solely to clear suggestions. Older checkpoint notes below describe pre-migration behavior where they conflict with this section.


- **6.1–6.3 implemented in the working tree:** per-rank MTP/DFlash proposer graphs; GQA-shared `mma.sync` attention and split-KV reduction; expert-grouped MoE MMA with sparse-batch GEMV fallback. Earlier session measurements and parity gates are recorded in the agent history; final aligned performance/oracle runs remain required before Phase 6 release completion. The MoE dispatch intentionally retains GEMV when `total_pairs < num_experts` because grouping sparse routes regressed measured latency.
- **6.4 structured outputs & HTTP `tool_choice` wired and verified:** `constraints.rs` uses pinned `llguidance 1.9.1` plus the ByteLevel tokenizer adapter (with HF non-special `<...>` added tokens such as Qwen's native `</think>` `248069`, `<tool_call>` `248058`, and `</tool_call>` `248059` restored to ordinary bytes in the grammar `TokTrie` and terminal EOS tracked transactionally across `rollback()`) for JSON objects, strict JSON Schema, regex/Lark, lazy structural tags, and token-exact clone/rollback across `</think>`. `request_constraints.rs` + `server.rs` compile request constraints outside `state.engine.lock()` using an immutable `ConstraintFactory` built at server startup (`64 KiB` grammar/schema cap, `512 KiB` tools cap, `null`-tolerant optional fields).
- **Tool grammars & live Qwen 3.5 / 3.6 HTTP gates:** `tool_constraints.rs` supports `none`, `auto`, `required`, and named functions in Qwen XML or Hermes JSON. XML parameter values are JSON literals; unsupported cross-property XML schema keywords fail closed on mandatory `required`/named choices while opportunistic `auto`/`none` requests degrade cleanly without breaking unconstrained callers. Verified live over HTTP (`phase6_http_constraints` **4/4**) on `Qwen3.5-4B` (`MTP k=3` and `DFlash2 M=16`) and `Qwen3.6-35B-A3B` MoE (`TP=2 + MTP k=3`) across `<think>`-aware strict `json_schema`, `tool_choice` (`required|named|auto|none`), multi-turn `system + tools + assistant.tool_calls + role:tool <tool_response>`, `structural_tag`, `json_object`, `grammar`, and `regex`. Non-speculative/multimodal mandatory constraint enforcement will be unified when GPU-resident `SamplerV1` + `SchedulerV1` serving lands in 6.5/6.6a.
- **Native ChatML:** system+tools instructions, assistant calls, grouped tool responses, current-turn reasoning and `preserve_thinking` are implemented. `tools/oracle/chat_template_golden.py` generated 28 byte-exact cases from the native Qwen3.6 template (`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`). The cached custom 4B `froggeric` template differs and is not claimed equivalent. Phase 6 tests: `phase6_chatml` **5/5**, `phase6_constraints` **12/12**, `phase6_http_constraints` **4/4**, `phase4_sampling` **6/6**.
- **Drafter oracle isolation:** the new target attention reduction order changes recorded BF16 taps (`relL2=0.0017256350013745048`, 15,071 changed elements for the fixed prompt). The test separately gates target tap drift at `<0.005` and runs the drafter on the donor's exact recorded inputs; `actual_sglang_full_model_hidden_parity` reports **`relL2=0, maxabs=0`**. This is an isolated drafter 0-ULP claim, not end-to-end target+drafter bit equality.
- **Review:** fixed the important JSON/XML delimiter-injection finding with a reproduced failing regression and every-stream-split tests. Declined added-token escaping because it would break native Jinja byte parity; prompt token sanitization remains a separate API policy decision. Capture failures remain explicit rather than silently hiding failed graph gates. Synthetic remote-image behavior has now been replaced by the bounded opt-in transport described below. The target-tap gate change above is deliberate numerical/oracle separation, not an assertion of unchanged target tensors. Follow-up review's named-tool validation bug is fixed: excluded tools' schemas are not compiled. XML intentionally emits declared properties only (a schema-valid subset); use Hermes for open dictionaries. Incremental tool-parser allocation optimization is deferred to the streaming worker work rather than expanding this correctness fix.
- **Checkpoint verification:** after image/ingress review fixes, `export PATH="/opt/cuda/bin:$PATH"; mise run check` passed in **93.12 s**: formatting, CPU and CUDA clippy, and **95 release tests, 0 failures** (`/tmp/w34-phase6-images-check-final.log`). The 13 new image/ingress tests cover transport, deadlines, limits, original fixture bytes, decompression rejection, stalled-client isolation, and accept-error recovery. This verifies the implemented checkpoint, not the still-pending Phase 6 exit gates.
- **6.6b remote-image transport implemented:** `--allow-remote-images` enables actual public HTTP(S) PNG/JPEG/PPM fetches; default HTTP/CLI policy rejects remote URLs instead of generating fake pixels. Limits: 16 MiB per fetched image, four images per HTTP request, a shared 10-second request fetch budget, 16 KiB response headers, no redirects/proxies/URL credentials/content compression, and DNS-address filtering at the connector seam. Raster decoding is serialized independently of ingress, capped at 16 megapixels and 128 MiB decoder workspace before RGB conversion. Eight bounded ingress handlers plus a 16-connection queue isolate slow clients; **GPU execution remains serialized, not SchedulerV1 serving**. Accept errors do not terminate the server. Local CLI files remain supported.
- **Fixture/harness migration:** explicit `rgb://<opaque seed>` fixtures preserve old HTTP-fixture pixels; `.mise.toml` and built-in vision benchmarks use their original seed strings (e.g. `rgb://https://w34.test/sample.ppm`, with no nested fetch). Rebuilt 4B `tq4` vision `dump-diff --compare-tp2` passes with 68 tensors, `target_max_rel_l2=0.041071`, `min_cosine=0.999157`, top-1 `357` on both ranks. This is synthetic-image TP parity, **not real-image vLLM oracle parity**.
- **Image review triage:** fixed the stale oracle fixture, aggregate fetch work, decode-memory amplification, transient accept handling, and case-insensitive `Content-Encoding`. Added failing-then-passing HTTP-noise and decompression-bomb regressions. Deferred connection pooling (bounded per-image agents retain independent remaining-deadline configuration) and typed upstream 502/504 errors to serving integration; existing detailed 400 validation errors remain, with credentials and private network targets rejected.
- **Still pending:** remaining serving wire-format semantics; oracle-aligned concurrent serving/performance profiles; actual PNG/JPEG image oracles; 32k/64k/128k/262k needle gates; and final performance/harness verification. GPU stochastic sampling/device bitmasking is implemented in `f1a0be7`, and the Axum worker now unifies it across plain/scheduled/multimodal paths. Do not mark Phase 6 complete based on the passing checkpoint tests.

### Original Phase 6 Work Items

1. **6.1 Proposer CUDA Graphs & Drafter Roofline (`MTP` & `DFlash2`):** Capture per-rank CUDA Graphs for `Qwen3_5MtpRank` `M=1` draft steps (dense & `35B-A3B` MoE, single-thread `TP=2` graph launch) and `DFlash2Rank::forward` (`M=16` on `4B`, `M=8` on `27B`), cutting `spec_decode` from `4.90 ms/round` to `< 2.0 ms/round` (`35B-A3B MoE + MTP` → `500+ tok/s`, `4B + DFlash2` → `1,100+ tok/s`).
2. **6.2 GQA-Grouped, Split-KV, & `mma.sync` Paged `TQ4` / `Q8_0` Attention (`op_paged_attn.cu`):** Share decoded `TQ4`/`Q8_0`/`FP16` KV tiles across GQA query heads (`4:1` / `8:1` broadcast + `mma.sync` for `M ∈ [2..16]`) and add parallel split-KV reduction across SMs for `seq_len >= 4,096` out to `262,144`.
3. **6.3 Token-Grouped Marlin `mma.sync` MoE Kernel (`op_moe.cu`):** Add sorted `(expert_id, token_slot)` `moe_wna16`-style Marlin `mma.sync` tiles for `M >= 4` prefill, speculative verify, and batched decode while retaining warp-per-row GEMV for `M = 1..3`.
4. **6.4 `<think>`-Aware Structured Outputs, 100%-Schema Tool Calling, & Multi-Turn Agent ChatML (`parser.rs`, `server.rs`):** Upgrade `TokenBitmaskConstraint` from a 1-token `{` stub to a full JSON / JSON-Schema (`response_format: json_object | json_schema`) + Structural-Tag engine plugged into `RejectionSampler`'s speculative rollback (`MTP`/`DFlash2`), deferring grammar activation until `</think>`, enforcing `tool_choice` (`none|auto|required|named`) against Qwen 3.5 / 3.6's native `Qwen3-XML` / `Hermes-JSON` tool schemas, and fixing `format_chatml_prompt` for `system` + `tools`, multi-turn `assistant.tool_calls`, and `role: "tool"` `<tool_response>` wrapping.
5. **6.5 Concurrent Continuous-Batching Engine Loop, Live SSE Streaming, & Full OpenAI Controls (`server.rs`, `scheduler.rs`):** Replace `Mutex<Qwen3_5Engine>` serialization and post-hoc SSE buffering in `w34 serve` with a background worker driving `SchedulerV1::step()` over an MPSC request queue + per-request SSE token channels, disconnect cancellation, unified `SamplingParams` on non-speculative & scheduled paths, `stop` strings (with partial-match streaming holdback), `stop_token_ids`, `logit_bias`, `min_tokens`, `ignore_eos`, `logprobs`/`top_logprobs`, `stream_options.include_usage`, and multi-request `GraphKey` decode buckets (`B ∈ {1, 2, 4, 8, 16}`).
6. **6.6 GPU-Resident Stochastic `SamplerV1`, Real-Image `Qwen3.5-VL` Parity, & `32k–262k` Needle Gates:** Move non-greedy `SamplerV1` + grammar bitmasking onto GPU, verify real PNG/JPEG `Qwen3.5-VL` (`4B` / `27B` / `35B-A3B`) golden dumps against `vLLM 0.31.0`, add opt-in `--allow-remote-images`, and run `32k / 64k / 128k / 262k` needle-in-a-haystack benchmarks (`tq4` vs `tq4_all`) on 1× RTX 3090 Ti.
