# Completion campaign measurements

**Qualification evidence, not a winner ranking or publication certification.**

| Deployment | Case | Trials | Successful / requested | Output tok/s |
|---|---|---:|---:|---:|
| llamacpp-tp1-fp16 | agent-c1 | 3 | 24 / 24 | 39.96 |
| llamacpp-tp1-fp16 | decode-c1 | 3 | 12 / 12 | 44.36 |
| llamacpp-tp1-fp16 | prefill-primed-c1 | 3 | 12 / 12 | 39.86 |
| llamacpp-tp1-fp16 | throughput-c8 | 3 | 48 / 48 | 96.47 |
| vllm-tp1-fp16 | agent-c1 | 3 | 24 / 24 | 41.08 |
| vllm-tp1-fp16 | decode-c1 | 3 | 12 / 12 | 50.76 |
| vllm-tp1-fp16 | prefill-primed-c1 | 3 | 12 / 12 | 42.96 |
| vllm-tp1-fp16 | throughput-c8 | 3 | 48 / 48 | 309.89 |
| warp34-tp1-fp16 | agent-c1 | 3 | 24 / 24 | 49.02 |
| warp34-tp1-fp16 | decode-c1 | 3 | 12 / 12 | 51.86 |
| warp34-tp1-fp16 | prefill-primed-c1 | 3 | 12 / 12 | 51.94 |
| warp34-tp1-fp16 | throughput-c8 | 3 | 48 / 48 | 228.66 |
| ninfer-tp1-int8 | agent-c1 | 3 | 24 / 24 | 47.00 |
| ninfer-tp1-int8 | decode-c1 | 3 | 12 / 12 | 52.09 |
| ninfer-tp1-int8 | prefill-primed-c1 | 3 | 12 / 12 | 35.90 |
| ninfer-tp1-int8 | throughput-c8 | 3 | 48 / 48 | 243.78 |

## Startup / qualification outcomes

- `llamacpp-tp1-fp16`: startup **ready**, quality **failed**, measurements **complete**.
- `vllm-tp1-fp16`: startup **ready**, quality **passed**, measurements **complete**.
- `warp34-tp1-fp16`: startup **ready**, quality **passed**, measurements **complete**.
- `ninfer-tp1-int8`: startup **startup_receipt_not_retained**, quality **passed**, measurements **complete**.

## Interpretation

- Qualification measurements; not a winner ranking.
- Different weight/KV/cohort/renderer/GPU settings are not controlled equivalents.
- Repetitions within one server lifetime are not independent restart trials.
- Baseline quality passes do not establish long-context quality.
- Resource peaks cover whole runs including warmup and priming.

Exact deployment settings, per-case blockers, resource summaries and input hashes are in the accompanying JSON.
Raw manifests, request events, quality evidence, startup logs and telemetry remain in the referenced result directories.
