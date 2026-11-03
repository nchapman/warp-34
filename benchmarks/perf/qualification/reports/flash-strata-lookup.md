# Inference serving performance

Run status: **complete**. Finite trials include fill/drain; not steady-state capacity.
4-bit deployment equivalents, not bit-identical weights. No automatic winner ranking.
First output includes visible reasoning/tool deltas. TPOT is a usage/span estimate, not token ITL.

Values below are medians of per-trial statistics; JSON retains every trial and distribution.

| Deployment | Case | GPUs | Weights / KV | Reps | Req OK/total | tok/s median [min, max] | First output p50/p95 ms | E2E p95 ms |
|---|---|---:|---|---:|---:|---|---|---:|
| strata-flash-tp1-lookup4 | short-chat | 1 | GGUF UD-IQ4_XS; mixed IQ3_S/IQ4_XS/IQ4_NL/Q8_0 experts / fp16 | 3 | 12/12 | 52.53 [50.69, 55.88] | 624.93/676.31 | 5416.23 |
| strata-flash-tp1-lookup4 | prefill-context-r16 | 1 | GGUF UD-IQ4_XS; mixed IQ3_S/IQ4_XS/IQ4_NL/Q8_0 experts / fp16 | 3 | 12/12 | 16.10 [16.06, 16.22] | 2899.16/2944.52 | 4202.52 |
| strata-flash-tp1-lookup4 | code-context-r64 | 1 | GGUF UD-IQ4_XS; mixed IQ3_S/IQ4_XS/IQ4_NL/Q8_0 experts / fp16 | 3 | 12/12 | 15.27 [15.13, 15.41] | 12236.36/12970.11 | 17894.52 |
| strata-flash-tp1-lookup4 | prefix-primed-r16 | 1 | GGUF UD-IQ4_XS; mixed IQ3_S/IQ4_XS/IQ4_NL/Q8_0 experts / fp16 | 3 | 12/12 | 53.63 [53.41, 54.61] | 89.06/99.52 | 1445.84 |
| strata-flash-tp1-lookup4 | coding-agent-replay | 1 | GGUF UD-IQ4_XS; mixed IQ3_S/IQ4_XS/IQ4_NL/Q8_0 experts / fp16 | 3 | 24/24 | 49.68 [49.43, 49.91] | 549.04/1020.97 | 3016.99 |

## Long-context & prefix-cache breakdown

Cold bursts have >= 4,096 uncached prompt tokens; warm turns have cached_tokens > 0 and < 4,096 uncached tokens.

| Deployment | Case | Peak prompt tok | Cache hit % | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 ms | Cold prefill tok/s p50 | TPOT p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| strata-flash-tp1-lookup4 | short-chat | 177 | 0.0% | n/a | n/a | n/a | 16.59 |
| strata-flash-tp1-lookup4 | prefill-context-r16 | 1561 | 0.0% | n/a | n/a | n/a | 16.75 |
| strata-flash-tp1-lookup4 | code-context-r64 | 5976 | 0.0% | n/a | 12236.36 | 487.58 | 17.12 |
| strata-flash-tp1-lookup4 | prefix-primed-r16 | 1562 | 99.5% | 89.06 | n/a | n/a | 16.28 |
| strata-flash-tp1-lookup4 | coding-agent-replay | 1380 | 80.0% | 368.89 | n/a | n/a | 11.36 |

## Agent replay per-step latency

Step numbers are zero-based. Token counts and TTFT are medians of per-trial p50s; missing cache usage stays unknown.

| Deployment | Case | Step | Req OK/total | Prompt tok | Cached tok | Uncached tok | TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| strata-flash-tp1-lookup4 | coding-agent-replay | 0 | 6/6 | 294.00 | 143.50 | 150.50 | 677.96 |
| strata-flash-tp1-lookup4 | coding-agent-replay | 1 | 6/6 | 749.00 | 533.00 | 216.00 | 510.43 |
| strata-flash-tp1-lookup4 | coding-agent-replay | 2 | 6/6 | 965.00 | 868.00 | 97.00 | 537.82 |
| strata-flash-tp1-lookup4 | coding-agent-replay | 3 | 6/6 | 1380.00 | 1165.50 | 214.50 | 594.43 |

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
| strata-flash-tp1-lookup4 | short-chat | 1 | measured | 3/3 |  |
| strata-flash-tp1-lookup4 | prefill-context-r16 | 1 | measured | 3/3 |  |
| strata-flash-tp1-lookup4 | code-context-r64 | 1 | measured | 3/3 |  |
| strata-flash-tp1-lookup4 | prefix-primed-r16 | 1 | measured | 3/3 |  |
| strata-flash-tp1-lookup4 | coding-agent-replay | 1 | measured | 3/3 |  |
| warp34-flash-tp2-target-8k | short-chat | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target-8k | prefill-context-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target-8k | code-context-r64 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target-8k | prefix-primed-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-target-8k | coding-agent-replay | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3-8k | short-chat | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3-8k | prefill-context-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3-8k | code-context-r64 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3-8k | prefix-primed-r16 | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-mtp3-8k | coding-agent-replay | 2 | not_selected | 0/3 | not requested by engine selection |

## Publication checklist / limitations

- **strata-flash-tp1-lookup4 / short-chat:** attached independent quality checks failed for this deployment; deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified chat_template_sha256; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **strata-flash-tp1-lookup4 / prefill-context-r16:** attached independent quality checks failed for this deployment; deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified chat_template_sha256; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **strata-flash-tp1-lookup4 / code-context-r64:** attached independent quality checks failed for this deployment; deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified chat_template_sha256; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **strata-flash-tp1-lookup4 / prefix-primed-r16:** attached independent quality checks failed for this deployment; deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified chat_template_sha256; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **strata-flash-tp1-lookup4 / coding-agent-replay:** attached independent quality checks failed for this deployment; deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified chat_template_sha256; unverified source_model_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
