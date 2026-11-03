# Inference serving performance

Run status: **complete**. Finite trials include fill/drain; not steady-state capacity.
4-bit deployment equivalents, not bit-identical weights. No automatic winner ranking.
First output includes visible reasoning/tool deltas. TPOT is a usage/span estimate, not token ITL.

Values below are medians of per-trial statistics; JSON retains every trial and distribution.

| Deployment | Case | GPUs | Weights / KV | Reps | Req OK/total | tok/s median [min, max] | First output p50/p95 ms | E2E p95 ms |
|---|---|---:|---|---:|---:|---|---|---:|
| warp34-flash-tp1-capacity | cold-read-60k | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE / tq4 | 3 | 3/3 | 0.24 [0.24, 0.24] | 238639.09/238639.09 | 267413.85 |
| warp34-flash-tp1-capacity | agent-120k | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE / tq4 | 3 | 30/30 | 0.54 [0.49, 0.66] | 7560.50/129897.54 | 141404.24 |
| warp34-flash-tp1-capacity | agent-252k | 1 | AutoRound-W4A16-g128; FP8-E4M3-PLE / tq4 | 3 | 27/27 | 0.21 [0.19, 0.32] | 7913.03/308465.61 | 324302.22 |

## Long-context & prefix-cache breakdown

Cold bursts have >= 4,096 uncached prompt tokens; warm turns have cached_tokens > 0 and < 4,096 uncached tokens.

| Deployment | Case | Peak prompt tok | Cache hit % | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 ms | Cold prefill tok/s p50 | TPOT p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| warp34-flash-tp1-capacity | cold-read-60k | 57611 | 0.0% | n/a | 238639.09 | 241.41 | 445.76 |
| warp34-flash-tp1-capacity | agent-120k | 117891 | 82.6% | 6706.29 | 122007.54 | 224.60 | 388.09 |
| warp34-flash-tp1-capacity | agent-252k | 251854 | 82.7% | 7298.84 | 275610.83 | 221.64 | 424.99 |

## Agent replay per-step latency

Step numbers are zero-based. Token counts and TTFT are medians of per-trial p50s; missing cache usage stays unknown.

| Deployment | Case | Step | Req OK/total | Prompt tok | Cached tok | Uncached tok | TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| warp34-flash-tp1-capacity | agent-120k | 0 | 3/3 | 26699.00 | 0.00 | 26699.00 | 106353.35 |
| warp34-flash-tp1-capacity | agent-120k | 1 | 3/3 | 27588.00 | 26752.00 | 836.00 | 4991.07 |
| warp34-flash-tp1-capacity | agent-120k | 2 | 3/3 | 28648.00 | 27648.00 | 1000.00 | 5715.56 |
| warp34-flash-tp1-capacity | agent-120k | 3 | 3/3 | 57779.00 | 28672.00 | 29107.00 | 126871.27 |
| warp34-flash-tp1-capacity | agent-120k | 4 | 3/3 | 59435.00 | 57792.00 | 1643.00 | 7441.11 |
| warp34-flash-tp1-capacity | agent-120k | 5 | 3/3 | 60855.00 | 59456.00 | 1399.00 | 6988.75 |
| warp34-flash-tp1-capacity | agent-120k | 6 | 3/3 | 89932.00 | 60864.00 | 29068.00 | 132373.58 |
| warp34-flash-tp1-capacity | agent-120k | 7 | 3/3 | 90910.00 | 89920.00 | 983.00 | 6442.98 |
| warp34-flash-tp1-capacity | agent-120k | 8 | 3/3 | 116480.00 | 90880.00 | 25600.00 | 117143.81 |
| warp34-flash-tp1-capacity | agent-120k | 9 | 3/3 | 117891.00 | 116480.00 | 1436.00 | 7619.36 |
| warp34-flash-tp1-capacity | agent-252k | 0 | 3/3 | 54542.00 | 0.00 | 54542.00 | 223756.39 |
| warp34-flash-tp1-capacity | agent-252k | 1 | 3/3 | 55431.00 | 54592.00 | 839.00 | 5124.72 |
| warp34-flash-tp1-capacity | agent-252k | 2 | 3/3 | 116550.00 | 55488.00 | 61062.00 | 271454.19 |
| warp34-flash-tp1-capacity | agent-252k | 3 | 3/3 | 117903.00 | 116544.00 | 1359.00 | 6956.97 |
| warp34-flash-tp1-capacity | agent-252k | 4 | 3/3 | 178972.00 | 117888.00 | 61084.00 | 279767.48 |
| warp34-flash-tp1-capacity | agent-252k | 5 | 3/3 | 180296.00 | 178944.00 | 1338.00 | 7477.89 |
| warp34-flash-tp1-capacity | agent-252k | 6 | 3/3 | 249233.00 | 180288.00 | 68939.00 | 327597.70 |
| warp34-flash-tp1-capacity | agent-252k | 7 | 3/3 | 250458.00 | 249216.00 | 1242.00 | 7298.84 |
| warp34-flash-tp1-capacity | agent-252k | 8 | 3/3 | 251854.00 | 250432.00 | 1422.00 | 7913.03 |

## Campaign coverage

Unsupported/disabled/filtered entries are not failures or zero-throughput measurements.

| Deployment | Case | GPUs | Status | Repetitions completed | Reason |
|---|---|---:|---|---:|---|
| warp34-flash-tp1-capacity | cold-read-60k | 1 | measured | 3/3 |  |
| warp34-flash-tp1-capacity | agent-120k | 1 | measured | 3/3 |  |
| warp34-flash-tp1-capacity | agent-252k | 1 | measured | 3/3 |  |
| warp34-flash-tp2-capacity | cold-read-60k | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-capacity | agent-120k | 2 | not_selected | 0/3 | not requested by engine selection |
| warp34-flash-tp2-capacity | agent-252k | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-capacity | cold-read-60k | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-capacity | agent-120k | 2 | not_selected | 0/3 | not requested by engine selection |
| qfn-2x3090-capacity | agent-252k | 2 | not_selected | 0/3 | not requested by engine selection |

## Publication checklist / limitations

- **warp34-flash-tp1-capacity / cold-read-60k:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified artifact_sha256; unverified chat_template_sha256; unverified gpu_ids; unverified quantization_details; unverified source_model_revision; unverified tokenizer_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **warp34-flash-tp1-capacity / agent-120k:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified artifact_sha256; unverified chat_template_sha256; unverified gpu_ids; unverified quantization_details; unverified source_model_revision; unverified tokenizer_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
- **warp34-flash-tp1-capacity / agent-252k:** deployment support has not been qualified; manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings; unverified artifact_sha256; unverified chat_template_sha256; unverified gpu_ids; unverified quantization_details; unverified source_model_revision; unverified tokenizer_revision
  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.
