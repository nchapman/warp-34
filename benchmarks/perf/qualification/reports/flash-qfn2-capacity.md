# Inference serving performance

Run status: **complete**. Finite trials include fill/drain; not steady-state capacity.
4-bit deployment equivalents, not bit-identical weights. No automatic winner ranking.
First output includes visible reasoning/tool deltas. TPOT is a usage/span estimate, not token ITL.

Values below are medians of per-trial statistics; JSON retains every trial and distribution.

| Deployment | Case | GPUs | Weights / KV | Reps | Req OK/total | tok/s median [min, max] | First output p50/p95 ms | E2E p95 ms |
|---|---|---:|---|---:|---:|---|---|---:|
| qfn-2x3090-capacity | cold-read-60k | 2 | AutoRound-W4A16-g128; FP8-E4M3-PLE / bfloat16 | 3 | 3/3 | 1.49 [1.42, 1.49] | 42309.09/42309.09 | 42986.96 |
| qfn-2x3090-capacity | agent-120k | 2 | AutoRound-W4A16-g128; FP8-E4M3-PLE / bfloat16 | 3 | 0/3 | n/a | n/a/n/a | n/a |
| qfn-2x3090-capacity | agent-252k | 2 | AutoRound-W4A16-g128; FP8-E4M3-PLE / bfloat16 | 3 | 0/3 | n/a | n/a/n/a | n/a |

## Long-context & prefix-cache breakdown

Cold bursts have >= 4,096 uncached prompt tokens; warm turns have cached_tokens > 0 and < 4,096 uncached tokens.

| Deployment | Case | Peak prompt tok | Cache hit % | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 ms | Cold prefill tok/s p50 | TPOT p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| qfn-2x3090-capacity | cold-read-60k | 57611 | n/a | n/a | 42309.09 | 1361.67 | 13.12 |
| qfn-2x3090-capacity | agent-120k | n/a | n/a | n/a | n/a | n/a | n/a |
| qfn-2x3090-capacity | agent-252k | n/a | n/a | n/a | n/a | n/a | n/a |

## Agent replay per-step latency

Step numbers are zero-based. Token counts and TTFT are medians of per-trial p50s; missing cache usage stays unknown.

| Deployment | Case | Step | Req OK/total | Prompt tok | Cached tok | Uncached tok | TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| qfn-2x3090-capacity | agent-120k | 0 | 0/3 | n/a | n/a | n/a | n/a |
| qfn-2x3090-capacity | agent-252k | 0 | 0/3 | n/a | n/a | n/a | n/a |

## Campaign coverage

Unsupported/disabled/filtered entries are not failures or zero-throughput measurements.

| Deployment | Case | GPUs | Status | Repetitions completed | Reason |
|---|---|---:|---|---:|---|
| warp34-flash-tp1-capacity | cold-read-60k | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-capacity | agent-120k | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp1-capacity | agent-252k | 1 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-capacity | cold-read-60k | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-capacity | agent-120k | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-capacity | agent-252k | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-capacity | cold-read-60k | 2 | measured | 3/3 |  |
| qfn-2x3090-capacity | agent-120k | 2 | measurement_failed | 3/3 | request/session or usage-accounting failure; see summary.json |
| qfn-2x3090-capacity | agent-252k | 2 | measurement_failed | 3/3 | request/session or usage-accounting failure; see summary.json |

## Publication checklist / limitations

- **qfn-2x3090-capacity / cold-read-60k:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified artifact_sha256; unverified chat_template_sha256; unverified gpu_ids; unverified quantization_details; unverified source_model_revision; unverified tokenizer_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-2x3090-capacity / agent-120k:** deployment support has not been qualified; failed requests or incomplete agent sessions; publish failures alongside metrics; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; missing/invalid server usage; no token-throughput claim; unverified artifact_sha256; unverified chat_template_sha256; unverified gpu_ids; unverified quantization_details; unverified source_model_revision; unverified tokenizer_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **qfn-2x3090-capacity / agent-252k:** deployment support has not been qualified; failed requests or incomplete agent sessions; publish failures alongside metrics; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; missing/invalid server usage; no token-throughput claim; unverified artifact_sha256; unverified chat_template_sha256; unverified gpu_ids; unverified quantization_details; unverified source_model_revision; unverified tokenizer_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
