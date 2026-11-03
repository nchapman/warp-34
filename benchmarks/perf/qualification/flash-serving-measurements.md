# Completion campaign measurements

**Qualification evidence, not a winner ranking or publication certification.**

| Deployment | Case | Trials | Successful / requested | Output tok/s |
|---|---|---:|---:|---:|
| qfn-2x3090-mtp3 | code-context-r64 | 3 | 12 / 12 | 28.20 |
| qfn-2x3090-mtp3 | prefill-context-r16 | 3 | 12 / 12 | 21.27 |
| qfn-2x3090-mtp3 | prefix-primed-r16 | 3 | 12 / 12 | 20.52 |
| qfn-2x3090-mtp3 | short-chat | 3 | 12 / 12 | 56.20 |
| warp34-flash-tp1-target | code-context-r64 | 3 | 12 / 12 | 3.53 |
| warp34-flash-tp1-target | coding-agent-replay | 3 | 24 / 24 | 11.77 |
| warp34-flash-tp1-target | prefill-context-r16 | 3 | 12 / 12 | 7.66 |
| warp34-flash-tp1-target | prefix-primed-r16 | 3 | 12 / 12 | 11.47 |
| warp34-flash-tp1-target | short-chat | 3 | 12 / 12 | 16.49 |
| qfn-3090-mtp3 | code-context-r64 | 3 | 12 / 12 | 21.23 |
| qfn-3090-mtp3 | prefill-context-r16 | 3 | 12 / 12 | 10.23 |
| qfn-3090-mtp3 | prefix-primed-r16 | 3 | 12 / 12 | 10.17 |
| qfn-3090-mtp3 | short-chat | 3 | 12 / 12 | 30.82 |
| strata-flash-tp1-lookup4 | code-context-r64 | 3 | 12 / 12 | 15.27 |
| strata-flash-tp1-lookup4 | coding-agent-replay | 3 | 24 / 24 | 49.67 |
| strata-flash-tp1-lookup4 | prefill-context-r16 | 3 | 12 / 12 | 16.13 |
| strata-flash-tp1-lookup4 | prefix-primed-r16 | 3 | 12 / 12 | 53.88 |
| strata-flash-tp1-lookup4 | short-chat | 3 | 12 / 12 | 52.95 |
| warp34-flash-tp2-target-8k | code-context-r64 | 3 | 12 / 12 | 4.63 |
| warp34-flash-tp2-target-8k | coding-agent-replay | 3 | 24 / 24 | 34.29 |
| warp34-flash-tp2-target-8k | prefill-context-r16 | 3 | 12 / 12 | 19.17 |
| warp34-flash-tp2-target-8k | prefix-primed-r16 | 3 | 12 / 12 | 29.32 |
| warp34-flash-tp2-target-8k | short-chat | 3 | 12 / 12 | 46.26 |

## Startup / qualification outcomes

- `qfn-2x3090-mtp3`: startup **ready**, quality **passed**, measurements **failed**.
- `warp34-flash-tp1-mtp3`: startup **ready**, quality **failed**, measurements **failed**.
- `warp34-flash-tp1-target`: startup **ready**, quality **passed**, measurements **complete**.
- `warp34-flash-tp2-mtp3`: startup **failed**, quality **None**, measurements **None**.
  - server exited before readiness: 1
- `warp34-flash-tp2-target`: startup **failed**, quality **None**, measurements **None**.
  - server exited before readiness: 1
- `qfn-3090-mtp3`: startup **ready**, quality **passed**, measurements **failed**.
- `strata-flash-tp1-target`: startup **failed**, quality **None**, measurements **None**.
  - server exited before readiness: 1
- `strata-flash-tp1-lookup4`: startup **ready**, quality **failed**, measurements **complete**.
- `warp34-flash-tp2-mtp3-8k`: startup **ready**, quality **failed**, measurements **failed**.
- `warp34-flash-tp2-target-8k`: startup **ready**, quality **passed**, measurements **complete**.

## Interpretation

- Qualification measurements; not a winner ranking.
- Different weight/KV/cohort/renderer/GPU settings are not controlled equivalents.
- Repetitions within one server lifetime are not independent restart trials.
- Baseline quality passes do not establish long-context quality.
- Resource peaks cover whole runs including warmup and priming.

Exact deployment settings, per-case blockers, resource summaries and input hashes are in the accompanying JSON.
Raw manifests, request events, quality evidence, startup logs and telemetry remain in the referenced result directories.
