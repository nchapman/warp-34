# Inference serving performance

Run status: **failed**. Finite trials include fill/drain; not steady-state capacity.
4-bit deployment equivalents, not bit-identical weights. No automatic winner ranking.
First output includes visible reasoning/tool deltas. TPOT is a usage/span estimate, not token ITL.

Values below are medians of per-trial statistics; JSON retains every trial and distribution.

| Deployment | Case | GPUs | Weights / KV | Reps | Req OK/total | tok/s median [min, max] | First output p50/p95 ms | E2E p95 ms |
|---|---|---:|---|---:|---:|---|---|---:|
| qfn-3090-mtp3 | short-chat | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / int8_per_token_head | 3 | 12/12 | 30.60 [30.41, 31.46] | 1173.16/1210.29 | 9294.49 |
| qfn-3090-mtp3 | prefill-context-r16 | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / int8_per_token_head | 3 | 12/12 | 10.25 [10.18, 10.27] | 4612.64/4631.13 | 6504.38 |
| qfn-3090-mtp3 | code-context-r64 | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / int8_per_token_head | 3 | 12/12 | 21.21 [20.91, 21.57] | 4763.56/4794.81 | 12912.97 |
| qfn-3090-mtp3 | prefix-primed-r16 | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE; optional MTP-INT4-g32 / int8_per_token_head | 3 | 12/12 | 10.20 [10.11, 10.20] | 4620.40/4635.50 | 6489.31 |

## Long-context & prefix-cache breakdown

Cold bursts have >= 4,096 uncached prompt tokens; warm turns have cached_tokens > 0 and < 4,096 uncached tokens.

| Deployment | Case | Peak prompt tok | Cache hit % | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 ms | Cold prefill tok/s p50 | TPOT p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| qfn-3090-mtp3 | short-chat | 177 | n/a | n/a | n/a | n/a | 28.36 |
| qfn-3090-mtp3 | prefill-context-r16 | 1561 | n/a | n/a | n/a | n/a | 27.44 |
| qfn-3090-mtp3 | code-context-r64 | 5976 | n/a | n/a | 4763.56 | 1252.51 | 29.13 |
| qfn-3090-mtp3 | prefix-primed-r16 | 1562 | n/a | n/a | n/a | n/a | 27.82 |

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
| qfn-2x3090-mtp3 | short-chat | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | prefill-context-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | code-context-r64 | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | prefix-primed-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-mtp3 | coding-agent-replay | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-3090-mtp3 | short-chat | 1 | measured | 3/3 |  |
| qfn-3090-mtp3 | prefill-context-r16 | 1 | measured | 3/3 |  |
| qfn-3090-mtp3 | code-context-r64 | 1 | measured | 3/3 |  |
| qfn-3090-mtp3 | prefix-primed-r16 | 1 | measured | 3/3 |  |
| qfn-3090-mtp3 | coding-agent-replay | 1 | failed | 0/3 | MeasurementError; inspect requests.jsonl for evidence |
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

- **qfn-3090-mtp3 / short-chat:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-3090-mtp3 / prefill-context-r16:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-3090-mtp3 / code-context-r64:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-3090-mtp3 / prefix-primed-r16:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified quantization_details; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
