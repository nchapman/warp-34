# Where warp-34 stands: 27B and Flash Next

## Bottom line

**27B is competitive on one GPU and strong on two. Flash Next is behind the
specialized references, especially on longer inputs, and its native MTP HTTP
path failed qualification.** Long-context retrieval works on native Flash TP1;
large-capacity native TP2 serving did not fit the requested configuration.

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
| warp-34 FP16 KV | 51.86 | 228.66 | 49.02 |
| warp-34 TQ4 KV | 51.22 | 307.02 | 47.81 |
| vLLM FP16 KV | 50.76 | 309.89 | 41.08 |
| NInfer INT8 KV | 52.09 | 243.78 | 47.00 |
| llama.cpp FP16 KV | 44.36 | 96.47 | 39.96 |

- Single-request performance is roughly tied with vLLM and NInfer.
- Native FP16-KV C8 throughput is about **26% below vLLM**. Compressed native KV
  reaches 304–307 tok/s, roughly matching this vLLM configuration, with different
  KV precision. This is a practical configuration result, not an isolated causal
  demonstration that KV compression alone explains the difference.
- Native agent replay is about **19% above vLLM** in the baseline campaign.
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

## Flash Next: a substantial longer-input performance gap

| Deployment | Short chat tok/s | ~6K-context code proxy tok/s | Strict agent replay |
|---|---:|---:|---|
| warp-34 TP1, target-only | 16.49 | 3.53 | 11.77 tok/s; complete |
| [one-GPU specialized vLLM](https://github.com/DominikBucko/qwen38-flash-next-3090), MTP3 | 30.82 | 21.23 | Failed: live tool call |
| warp-34 TP2, target-only, 8K allocation | 46.26 | 4.63 | 34.29 tok/s; complete |
| [two-GPU specialized vLLM](https://github.com/DominikBucko/qwen38-flash-next-2x3090), MTP3 | 56.20 | 28.20 | Failed: live tool call |
| Strata TP1, lookup4 | 52.95 | 15.27 | 49.67 tok/s; quality caveat |

- Native short chat delivers about **54%** of the corresponding one-GPU reference
  and **82%** of the two-GPU reference throughput.
- On the longer code-context proxy, native delivers roughly **one-sixth** of the
  corresponding specialized reference throughput. This is the largest measured
  performance gap. The fixture is technical prose with a coding prompt, not a
  full source repository.
- The references use speculation; the working native configurations are
  target-only. Native Flash MTP reached HTTP readiness in TP1 and reduced-capacity
  TP2, but **failed quality and warmup with SSE errors**. No valid native MTP
  speedup is available from these runs.
- Native TP2 with 4096 pages failed serving-state memory admission. Reducing it
  to 512 pages/8K made target-only serving work, without clearing the larger
  allocation failure. TP1 target uses 4096 pages, so these are not equal-capacity
  topology comparisons.
- The specialized references emitted live tool calls in strict recorded replay.
  The client correctly refused to execute them. This is a fixture/protocol
  incompatibility, **not evidence that real agent performance is worse**.
- Strata uses different mixed-precision GGUF weights and lookup speculation. It
  failed one synthetic-secret retrieval check by refusal (5/6 overall); its
  throughput is not a quality-equivalent replacement claim. Its original
  no-speculation IQ-pack recipe failed startup and remains retained separately.

## Flash long context: usable TP1 capacity, costly end-to-end latency

| Deployment | Retrieval checks | ~60K cold read | 120K/252K strict agent replay |
|---|---|---|---|
| Native TP1 target | 6/6 through ~252K | 0.240 output tok/s | Complete; 60/60 total workload requests |
| Native TP2, 16384 pages | Not run: startup failed | No measurement | No measurement |
| Two-GPU specialized vLLM, MTP3 | 6/6 through ~252K | 1.466 output tok/s | Six trials fail on live tool calls |

The reference's cold-read rate is about **6.1×** native TP1, but uses two GPUs
and MTP. These output rates include processing 57,611 prompt tokens to generate
at most 64 output tokens; they are **not steady-state decode rates**. Native
TP1's complete agent workload reached **251,982 prompt tokens**. The reference
passes long retrieval, but its strict agent replay does not qualify. Do not
replace the failed agent measurements with zero throughput.

## What to prioritize next — not fixes implemented by this benchmark task

1. **Flash longer-input performance:** profile prefill, expert movement/cache
   behavior and decode after longer prompts. The measurements identify the gap,
   not its root cause.
2. **Flash MTP HTTP correctness:** prerequisite to a valid speculative comparison.
3. **Flash TP2 memory admission/capacity:** explain and improve the discrepancy
   between smaller working allocations and rejected larger allocations.
4. **27B TP1 FP16-KV concurrency:** investigate the deficit versus vLLM and the
   different result with compressed KV.

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
