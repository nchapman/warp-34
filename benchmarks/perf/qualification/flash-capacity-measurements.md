# Completion campaign measurements

**Qualification evidence, not a winner ranking or publication certification.**

| Deployment | Case | Trials | Successful / requested | Output tok/s |
|---|---|---:|---:|---:|
| qfn-2x3090-capacity | agent-120k | 3 | 0 / 3 | unknown |
| qfn-2x3090-capacity | agent-252k | 3 | 0 / 3 | unknown |
| qfn-2x3090-capacity | cold-read-60k | 3 | 3 / 3 | 1.47 |
| warp34-flash-tp1-capacity | agent-120k | 3 | 30 / 30 | 0.57 |
| warp34-flash-tp1-capacity | agent-252k | 3 | 27 / 27 | 0.24 |
| warp34-flash-tp1-capacity | cold-read-60k | 3 | 3 / 3 | 0.24 |

## Startup / qualification outcomes

- `qfn-2x3090-capacity`: startup **ready**, quality **passed**, measurements **failed**.
- `warp34-flash-tp1-capacity`: startup **ready**, quality **passed**, measurements **complete**.
- `warp34-flash-tp2-capacity`: startup **failed**, quality **None**, measurements **None**.
  - server exited before readiness: 1

## Interpretation

- Qualification measurements; not a winner ranking.
- Different weight/KV/cohort/renderer/GPU settings are not controlled equivalents.
- Repetitions within one server lifetime are not independent restart trials.
- Baseline quality passes do not establish long-context quality.
- Resource peaks cover whole runs including warmup and priming.

Exact deployment settings, per-case blockers, resource summaries and input hashes are in the accompanying JSON.
Raw manifests, request events, quality evidence, startup logs and telemetry remain in the referenced result directories.
