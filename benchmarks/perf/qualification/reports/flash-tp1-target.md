# Inference serving performance

Run status: **complete**. Finite trials include fill/drain; not steady-state capacity.
4-bit deployment equivalents, not bit-identical weights. No automatic winner ranking.
First output includes visible reasoning/tool deltas. TPOT is a usage/span estimate, not token ITL.

Values below are medians of per-trial statistics; JSON retains every trial and distribution.

| Deployment | Case | GPUs | Weights / KV | Reps | Req OK/total | tok/s median [min, max] | First output p50/p95 ms | E2E p95 ms |
|---|---|---:|---|---:|---:|---|---|---:|
| warp34-flash-tp1-target | short-chat | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / tq4 | 3 | 12/12 | 16.42 [16.40, 16.66] | 2116.39/2301.84 | 16163.68 |
| warp34-flash-tp1-target | prefill-context-r16 | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / tq4 | 3 | 12/12 | 7.66 [7.66, 7.67] | 4782.01/4817.22 | 8594.18 |
| warp34-flash-tp1-target | code-context-r64 | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / tq4 | 3 | 12/12 | 3.53 [3.52, 3.53] | 18498.48/18763.00 | 73490.96 |
| warp34-flash-tp1-target | prefix-primed-r16 | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / tq4 | 3 | 12/12 | 11.49 [11.37, 11.55] | 1990.82/2153.56 | 5802.63 |
| warp34-flash-tp1-target | coding-agent-replay | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / tq4 | 3 | 24/24 | 11.77 [11.69, 11.86] | 3230.44/3633.55 | 11435.64 |

## Long-context & prefix-cache breakdown

Cold bursts have >= 4,096 uncached prompt tokens; warm turns have cached_tokens > 0 and < 4,096 uncached tokens.

| Deployment | Case | Peak prompt tok | Cache hit % | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 ms | Cold prefill tok/s p50 | TPOT p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| warp34-flash-tp1-target | short-chat | 177 | 0.0% | n/a | n/a | n/a | 52.96 |
| warp34-flash-tp1-target | prefill-context-r16 | 1561 | 0.0% | n/a | n/a | n/a | 57.68 |
| warp34-flash-tp1-target | code-context-r64 | 5976 | 0.0% | n/a | 18498.48 | 322.40 | 210.99 |
| warp34-flash-tp1-target | prefix-primed-r16 | 1562 | 94.8% | 1990.82 | n/a | n/a | 57.11 |
| warp34-flash-tp1-target | coding-agent-replay | 1578 | 67.8% | 3230.44 | n/a | n/a | 60.63 |

## Agent replay per-step latency

Step numbers are zero-based. Token counts and TTFT are medians of per-trial p50s; missing cache usage stays unknown.

| Deployment | Case | Step | Req OK/total | Prompt tok | Cached tok | Uncached tok | TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| warp34-flash-tp1-target | coding-agent-replay | 0 | 6/6 | 294.00 | 0.00 | 294.00 | 3276.12 |
| warp34-flash-tp1-target | coding-agent-replay | 1 | 6/6 | 842.50 | 384.00 | 458.50 | 3436.12 |
| warp34-flash-tp1-target | coding-agent-replay | 2 | 6/6 | 1157.50 | 960.00 | 197.50 | 2465.74 |
| warp34-flash-tp1-target | coding-agent-replay | 3 | 6/6 | 1572.00 | 1280.00 | 293.50 | 3230.44 |

## Campaign coverage

Unsupported/disabled/filtered entries are not failures or zero-throughput measurements.

| Deployment | Case | GPUs | Status | Repetitions completed | Reason |
|---|---|---:|---|---:|---|
| warp34-flash-tp1-target | short-chat | 1 | measured | 3/3 |  |
| warp34-flash-tp1-target | prefill-context-r16 | 1 | measured | 3/3 |  |
| warp34-flash-tp1-target | code-context-r64 | 1 | measured | 3/3 |  |
| warp34-flash-tp1-target | prefix-primed-r16 | 1 | measured | 3/3 |  |
| warp34-flash-tp1-target | coding-agent-replay | 1 | measured | 3/3 |  |
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
| qfn-2x3090-mtp3 | short-chat | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | prefill-context-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | code-context-r64 | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | prefix-primed-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | coding-agent-replay | 2 | not_selected | 0/3 | not requested by engine selection |
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

- **warp34-flash-tp1-target / short-chat:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **warp34-flash-tp1-target / prefill-context-r16:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **warp34-flash-tp1-target / code-context-r64:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **warp34-flash-tp1-target / prefix-primed-r16:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **warp34-flash-tp1-target / coding-agent-replay:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
