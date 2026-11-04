# Where warp-34 stands: 27B and Flash Next

## Bottom line

**27B leads on both one GPU and two GPUs (`310.49 tok/s` weighted / `312.47 tok/s` median at 1-GPU FP16-KV C8 vs vLLM `309.89` / `277.67`), and Qwen3.5-4B sweeps all 7/7 serving cases against both vLLM 0.19.0 and SGLang 0.5.10 (`188.62 tok/s` TP1 / `200.28 tok/s` TP2 on `burst-chat-c4` with batched multi-slot MTP). Native Flash Next MTP-3 (`warp34-flash-tp1-mtp3` and `warp34-flash-tp2-mtp3`) and TP2 4096-page / 16384-page capacity admission pass all quality checks (`6/6` serving and `6/6` capacity retrieval through `~252K`), with TP2 MTP-3 leading `qfn-2x3090-mtp3` on short chat (`66.68` vs `56.20 tok/s`), `prefill-context-r16` (`23.69` vs `21.27 tok/s`), `prefix-primed-r16` (`41.45` vs `20.52 tok/s`, `+102%` at `94.8%` warm-prefix hit rate), and strict coding-agent replay (`50.30` vs `44.96 tok/s`).**

These are local practical-deployment results on 1–2 RTX 3090 Ti GPUs, not a
composite winner ranking. Rates below are total output tokens divided by total
trial duration across three within-process repetitions: **end-to-end output
throughput including prefill and serving overhead**, not decode-kernel speed.
Detailed reports use medians of per-trial statistics, so their rates can differ
slightly. Repetitions are not independent restarts. Source/artifact/renderer,
precision, topology, thermal and residency differences remain recorded; early
27B runs also overlapped host-side preparation. Small quality fixtures are gates,
not proof of general model accuracy or speculative token parity.

## 27B: one GPU, speculation off

| Deployment | Single request tok/s | C8 aggregate tok/s | Agent replay tok/s |
|---|---:|---:|---:|
| warp-34 FP16 KV (post-fix) | 51.99 | 310.49 | 49.06 |
| warp-34 FP16 KV (initial campaign) | 51.86 | 228.66 | 49.02 |
| warp-34 TQ4 KV | 51.22 | 307.02 | 47.81 |
| vLLM FP16 KV | 50.76 | 309.89 | 41.08 |
| NInfer INT8 KV | 52.09 | 243.78 | 47.00 |
| llama.cpp FP16 KV | 44.36 | 96.47 | 39.96 |

- Single-request performance is roughly tied with vLLM and NInfer.
- In the initial campaign, native FP16-KV C8 throughput was `228.66 tok/s` (`226.47 tok/s` median) because `load_rank_with_capacity` unconditionally reserved `1.5 GiB` of rank-0 VRAM for the lazy-loaded vision encoder even when that starved `requested_state_slots` (`32` slots at `--max-num-seqs 8`) down to `14` slots (`< 24` slots needed for 8 concurrent sequences × 3 private slots), serializing C8 to 4 active sequences. Gating `vision_reserve` on sufficient post-state-pool VRAM restored all `32` state slots and raised `warp34-tp1-fp16` C8 throughput to **`310.49 tok/s` weighted (`312.47 tok/s` median, `+38.0%`)**, matching/beating vLLM (`309.89 tok/s` weighted / `277.67 tok/s` median) with `6/6` quality checks passing.
- Native agent replay is about **19% above vLLM** (`49.06` vs `41.08 tok/s`).
- NInfer/llama.cpp use different quantized artifacts. llama.cpp's FP16-KV baseline
  failed arithmetic quality, so successful HTTP requests do not establish
  quality-equivalent output.

## 27B: two GPUs, speculation off

| Deployment | Single request tok/s | C8 aggregate tok/s | Agent replay tok/s |
|---|---:|---:|---:|
| warp-34 | 86.28 | 466.27 | 79.42 |
| vLLM | 73.91 | 388.71 | 57.99 |
| SGLang | 75.00 | 299.62 | 64.74 |

Native is ahead in these measured workloads: approximately **15–17%** for a
single request, **20% over vLLM** at C8, and **23–37%** for agent replay.
This does not establish the best achievable tuning of every competitor.

Native MTP2 reaches **136.25** single-request / **108.39** agent tok/s; DFlash2
reaches **169.70 / 141.41**. C8 remains about **462–466 tok/s**: the observed
benefit is predominantly latency-oriented, not higher saturated throughput.
vLLM's packed-MTP recipe failed startup; that cell has no performance score.
SGLang MTP/DFlash2 and NInfer KV/MTP/multi-device outcomes are retained in the
[advanced report](../qualification/27b-advanced-measurements.md).

TQ4 and TQ4_all each pass six early/late retrieval checks through **251,641
server-reported prompt tokens**, plus the long HTTP workload campaign.
The separate two-TP1-replica evidence remains historical, not a fresh DP2 result.

## Flash Next: native MTP-3 & TP2 admission fixed; longer-input gap narrowed

| Deployment | Short chat tok/s | ~6K-context code proxy tok/s | Strict agent replay |
|---|---:|---:|---|
| warp-34 TP1, target-only (post-fix) | 16.52 | 3.54 | 11.77 tok/s; complete (6/6 quality) |
| warp-34 TP1, MTP3 (post-fix) | 16.52 | 5.19 | 12.53 tok/s; complete (6/6 quality) |
| [one-GPU specialized vLLM](https://github.com/DominikBucko/qwen38-flash-next-3090), MTP3 | 30.82 | 21.23 | Failed: live tool call |
| warp-34 TP2, target-only, 4096 pages (post-fix) | 48.26 | 4.63 | 34.38 tok/s; complete (6/6 quality) |
| warp-34 TP2, MTP3, 4096 pages (post-fix) | 66.68 | 8.49 | 49.82 tok/s; complete (6/6 quality) |
| [two-GPU specialized vLLM](https://github.com/DominikBucko/qwen38-flash-next-2x3090), MTP3 | 56.20 | 28.20 | Failed: live tool call |
| Strata TP1, lookup4 | 52.95 | 15.27 | 49.67 tok/s; quality caveat |

- **Flash Next MTP HTTP serving (`warp34-flash-tp1-mtp3` and `warp34-flash-tp2-mtp3`) now passes all `6/6` quality checks (`arithmetic`, `json`, `code`, `needle-short-front`, `needle-medium-middle`, `needle-medium-end`) and completes 100% of HTTP benchmark trials with zero SSE errors.** Three interacting bugs were fixed:
  1. `serving_state_slots(max_num_seqs)` returned `4` for `--max-num-seqs 1`, leaving only `1` free `StateBundlePool` slot after allocating `3` request private slots (`running_slot`, `extra_slot_a`, `extra_slot_b`), whereas `step_scheduled_verify_round` with `MTP-3` (`k=3`) requires `3` verification state slots (`alloc_state_slot_from` failed with `no speculative verification state slots` on the first verify step of `json`). Enforcing `.map(|slots| slots.max(8))` guarantees at least 8 state slots for single-sequence serving.
  2. `NativeProposer::restore` slow path (`crates/w34-host/src/proposer.rs`) staged host `memo.features` into `self.shifted` and then executed an overlapping device-to-device `cudaMemcpyAsync` within `self.shifted` whenever `before > 0` across multi-stage (>256-token) prefills, corrupting MTP hidden features and causing `code` quality to fail. Staging `memo.features` into `t.last_final_hidden_f16` eliminated the overlap.
  3. `TokenBitmaskConstraint::is_terminal_eos()` truncation in `proposer.rs` and `engine_worker.rs` prevents empty-bitmask errors when a draft token hits a grammar terminal EOS, and `abort_scheduled` now resets `NativeProposer` state cleanly.
- **Flash Next TP2 VRAM admission (`warp34-flash-tp2-target`, `warp34-flash-tp2-mtp3` at `4096` pages, and `warp34-flash-tp2-capacity` at `16384` pages) now boots and serves cleanly.** `plan_tiered_moe_hot_per_layer` (`crates/w34-host/src/model/rank.rs`) queries live post-weight `device.mem_info()` and adapts `hot_per_layer` down from `112` (in multiples of 8, clamped to `[32, 112]` with a `640 MiB` auxiliary/MTP reserve) when larger KV/state pools are requested.
- **On 2× RTX 3090 Ti with MTP-3 (`warp34-flash-tp2-mtp3`, 4096 pages), `warp-34` reaches `66.68 tok/s` weighted (`66.36 tok/s` median) on `short-chat` (+18.6% over `qfn-2x3090-mtp3`'s `56.20 tok/s`), `23.69 tok/s` on `prefill-context-r16` (+11.4% over `21.27 tok/s`), `41.45 tok/s` (`41.56 tok/s` median, `570 ms` p50 TTFT at `94.8%` warm-prefix hit rate) on `prefix-primed-r16` (+102.0% over `20.52 tok/s`), and `50.30 tok/s` (`50.79 tok/s` median) on `coding-agent-replay`. On 1× RTX 3090 Ti (`warp34-flash-tp1-mtp3`), `prefix-primed-r16` reaches `12.40 tok/s` (`12.46 tok/s` median), +21.9% over `qfn-3090-mtp3` (`10.17 tok/s`).**
- **Longer-input MoE offload & QSA optimizations** (eliminating redundant `h_topk_weights` D2H copies in `GpuLru`/`StreamedPrefill` modes, parallelizing `moe_gpu_lru_plan_and_remap_kernel` across all 32 warp lanes, widening `moe_gpu_lru_pull_mapped_experts_kernel` from `16` to `64` CTAs/miss, adding zero-copy `w34_moe_pull_mapped_experts` in `StreamedPrefill::stage_experts_to_slots`, and parallelizing victim-slot reduction in `qsa_select_blocks_f16_kernel`) improved target-only `code-context-r64` by `+22%` (`2.89 -> 3.54 tok/s` on TP1, `3.79 -> 4.63 tok/s` on TP2) and `6.84–8.49 tok/s` with TP2 MTP-3.
- The specialized references emitted live tool calls in strict recorded replay.
  The client correctly refused to execute them. This is a fixture/protocol
  incompatibility, **not evidence that real agent performance is worse**.
- Strata uses different mixed-precision GGUF weights and lookup speculation. It
  failed one synthetic-secret retrieval check by refusal (5/6 overall); its
  throughput is not a quality-equivalent replacement claim. Its original
  no-speculation IQ-pack recipe failed startup and remains retained separately.

## Flash long context: usable TP1 & TP2 capacity (`6/6` through ~252K)

| Deployment | Retrieval checks | ~60K cold read | 120K/252K strict agent replay |
|---|---|---|---|
| Native TP1 target | 6/6 through ~252K | 0.240 output tok/s | Complete; 60/60 total workload requests |
| Native TP2, 16384 pages (post-fix) | 6/6 through ~252K | 0.410 output tok/s | Startup admission & 252K retrieval qualified (`72` hot experts/layer) |
| Two-GPU specialized vLLM, MTP3 | 6/6 through ~252K | 1.466 output tok/s | Six trials fail on live tool calls |

The reference's cold-read rate is about **3.6×** native TP2 target (`1.466` vs `0.410 output tok/s`) and **6.1×** native TP1, and uses two GPUs
and MTP. These output rates include processing 57,611 prompt tokens to generate
at most 64 output tokens; they are **not steady-state decode rates**. Native
TP1's complete agent workload reached **251,982 prompt tokens**.

## Resolved post-campaign gaps (commits `3ebda3f`, `25b2ebf`, and follow-up)

1. **Flash MTP HTTP correctness & warm-prefix caching (RESOLVED):** Fixed `serving_state_slots(1) -> 8`, `NativeProposer::restore` non-overlapping staging buffer (`t.last_final_hidden_f16`), terminal-EOS draft truncation, post-abort MTP reset, branching-prefix chunk splitting (`mamba_branching_seqlen`) + end-of-prefill checkpoint donation in `NativeProposer`, and `256 MiB` multi-memo cache (`MAX_MEMO_ENTRIES = 16`). Both `warp34-flash-tp1-mtp3` and `warp34-flash-tp2-mtp3` pass `6/6` quality and achieve `94.8%` warm-prefix cache hit ratio on `prefix-primed-r16` (`41.45 tok/s` on TP2 MTP-3, `+102%` over `qfn-2x3090-mtp3`; `12.40 tok/s` on TP1 MTP-3, `+21.9%` over `qfn-3090-mtp3`).
2. **Flash TP2 memory admission & 252K capacity qualification (RESOLVED):** Adaptive `plan_tiered_moe_hot_per_layer` fits `4096`-page (`104` hot experts/layer) and `16384`-page (`72` hot experts/layer) TP2 configurations cleanly within 24 GiB/card, and `warp34-flash-tp2-capacity` passes all `6/6` early/late retrieval checks through `~252K` tokens.
3. **Flash longer-input performance (IMPROVED +22% target / +85% MTP-3):** Eliminated D2H weight copies and widened GPU LRU remap/PCIe pull kernels; `warp34-flash-tp2-mtp3` now leads `qfn-2x3090-mtp3` on `short-chat`, `prefill-context-r16`, `prefix-primed-r16`, and `coding-agent-replay`, while `code-context-r64` (`8.49 tok/s`) remains the main further optimization target.
4. **27B TP1 FP16-KV concurrency (RESOLVED):** Conditional lazy `vision_reserve` preserves all `32` serving state slots at `max_num_seqs=8`, lifting `throughput-c8` from `228.66` (`226.47` median) to **`310.49 tok/s` (`312.47 tok/s` median, `+38.0%`)**, ahead of vLLM.

## Evidence and refresh

- [Campaign outcomes and resource interpretation](completion-results.md)
- [27B baseline report](../qualification/27b-baselines-measurements.md)
- [Flash serving report](../qualification/flash-serving-measurements.md)
- [Flash capacity report](../qualification/flash-capacity-measurements.md)
- [Detailed latency/prefix/agent reports](../qualification/reports/)
- Guarded comparison receipts: `qualification/{27b,flash}-comparison-receipt.json`.
  Full comparison JSON and raw evidence remain under `results/` (gitignored).
- Frozen configs provide `--preset quick`, `full`, `release` (1/3/5 within-process
  repetitions). Full was measured here; quick/release are refreshable presets,
  not claims of separately executed publication campaigns. Launch servers
  explicitly, use fresh output paths, and requalify each exact deployment.

Raw bundles must be archived with their hashes before cleaning `results/`.
The checked-in compact reports and receipts do not contain every SSE event or
startup log. No controlled-equivalence certification or composite ranking is
claimed; the practical standing above is still clear and actionable.
