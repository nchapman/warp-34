# Inference serving performance

Run status: **failed**. Finite trials include fill/drain; not steady-state capacity.
4-bit deployment equivalents, not bit-identical weights. No automatic winner ranking.
First output includes visible reasoning/tool deltas. TPOT is a usage/span estimate, not token ITL.

Values below are medians of per-trial statistics; JSON retains every trial and distribution.

| Deployment | Case | GPUs | Weights / KV | Reps | Req OK/total | tok/s median [min, max] | First output p50/p95 ms | E2E p95 ms |
|---|---|---:|---|---:|---:|---|---|---:|
| qfn-2x3090-mtp3 | short-chat | 2 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / bfloat16 | 3 | 12/12 | 55.61 [55.23, 57.80] | 867.71/993.68 | 4737.66 |
| qfn-2x3090-mtp3 | prefill-context-r16 | 2 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / bfloat16 | 3 | 12/12 | 21.26 [21.19, 21.35] | 2124.04/2160.64 | 3120.90 |
| qfn-2x3090-mtp3 | code-context-r64 | 2 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / bfloat16 | 3 | 12/12 | 28.27 [27.94, 28.39] | 5478.51/5563.98 | 9339.41 |
| qfn-2x3090-mtp3 | prefix-primed-r16 | 2 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / bfloat16 | 3 | 12/12 | 20.54 [20.32, 20.70] | 2150.71/2195.17 | 3196.68 |

## Long-context & prefix-cache breakdown

Cold bursts have >= 4,096 uncached prompt tokens; warm turns have cached_tokens > 0 and < 4,096 uncached tokens.

| Deployment | Case | Peak prompt tok | Cache hit % | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 ms | Cold prefill tok/s p50 | TPOT p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| qfn-2x3090-mtp3 | short-chat | 177 | n/a | n/a | n/a | n/a | 14.42 |
| qfn-2x3090-mtp3 | prefill-context-r16 | 1561 | n/a | n/a | n/a | n/a | 14.42 |
| qfn-2x3090-mtp3 | code-context-r64 | 5976 | n/a | n/a | 5478.51 | 1088.35 | 14.58 |
| qfn-2x3090-mtp3 | prefix-primed-r16 | 1562 | n/a | n/a | n/a | n/a | 15.48 |

## Campaign coverage

Unsupported/disabled/filtered entries are not failures or zero-throughput measurements.

| Deployment | Case | GPUs | Status | Repetitions completed | Reason |
|---|---|---:|---|---:|---|
| warp34-flash-tp1-target | short-chat | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-target | prefill-context-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-target | code-context-r64 | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-target | prefix-primed-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-target | coding-agent-replay | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-mtp3 | short-chat | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-mtp3 | prefill-context-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-mtp3 | code-context-r64 | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-mtp3 | prefix-primed-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-mtp3 | coding-agent-replay | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target | short-chat | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target | prefill-context-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target | code-context-r64 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target | prefix-primed-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target | coding-agent-replay | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3 | short-chat | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3 | prefill-context-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3 | code-context-r64 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3 | prefix-primed-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3 | coding-agent-replay | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | short-chat | 2 | measured | 3/3 |  |
| qfn-2x3090-mtp3 | prefill-context-r16 | 2 | measured | 3/3 |  |
| qfn-2x3090-mtp3 | code-context-r64 | 2 | measured | 3/3 |  |
| qfn-2x3090-mtp3 | prefix-primed-r16 | 2 | measured | 3/3 |  |
| qfn-2x3090-mtp3 | coding-agent-replay | 2 | failed | 0/3 | MeasurementError; inspect requests.jsonl for evidence |
| qfn-3090-mtp3 | short-chat | 1 | not_selected | 0/3 | not requested by engine selection |
| qfn-3090-mtp3 | prefill-context-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| qfn-3090-mtp3 | code-context-r64 | 1 | not_selected | 0/3 | not requested by engine selection |
| qfn-3090-mtp3 | prefix-primed-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| qfn-3090-mtp3 | coding-agent-replay | 1 | not_selected | 0/3 | not requested by engine selection |
| ninfer-flash-unverified | short-chat | 2 | disabled | 0/3 | Known pinned native artifact is IQ2_XS-derived (WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090), not the primary approximately 4-bit cohort; Windows-preview runtime is separately unqualified on Linux. Excluded from this cohort, not an unsupported-architecture claim. |
| ninfer-flash-unverified | prefill-context-r16 | 2 | disabled | 0/3 | Known pinned native artifact is IQ2_XS-derived (WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090), not the primary approximately 4-bit cohort; Windows-preview runtime is separately unqualified on Linux. Excluded from this cohort, not an unsupported-architecture claim. |
| ninfer-flash-unverified | code-context-r64 | 2 | disabled | 0/3 | Known pinned native artifact is IQ2_XS-derived (WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090), not the primary approximately 4-bit cohort; Windows-preview runtime is separately unqualified on Linux. Excluded from this cohort, not an unsupported-architecture claim. |
| ninfer-flash-unverified | prefix-primed-r16 | 2 | disabled | 0/3 | Known pinned native artifact is IQ2_XS-derived (WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090), not the primary approximately 4-bit cohort; Windows-preview runtime is separately unqualified on Linux. Excluded from this cohort, not an unsupported-architecture claim. |
| ninfer-flash-unverified | coding-agent-replay | 2 | disabled | 0/3 | Known pinned native artifact is IQ2_XS-derived (WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090), not the primary approximately 4-bit cohort; Windows-preview runtime is separately unqualified on Linux. Excluded from this cohort, not an unsupported-architecture claim. |
| strata-flash-tp1-target | short-chat | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-target | prefill-context-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-target | code-context-r64 | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-target | prefix-primed-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-target | coding-agent-replay | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-lookup4 | short-chat | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-lookup4 | prefill-context-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-lookup4 | code-context-r64 | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-lookup4 | prefix-primed-r16 | 1 | not_selected | 0/3 | not requested by engine selection |
| strata-flash-tp1-lookup4 | coding-agent-replay | 1 | not_selected | 0/3 | not requested by engine selection |

## Publication checklist / limitations

- **qfn-2x3090-mtp3 / short-chat:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-2x3090-mtp3 / prefill-context-r16:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-2x3090-mtp3 / code-context-r64:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-2x3090-mtp3 / prefix-primed-r16:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
