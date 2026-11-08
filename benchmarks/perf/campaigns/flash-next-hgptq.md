# Flash-Next HGPTQ standalone serving qualification

**Qualification only — no locked performance baseline.** Concurrent engine
performance work will be incorporated before baseline measurements. Recorded
rates are diagnostic observations of source `b7f5fdd`, not current standings.

## Scope and environment

This campaign refreshes Flash-Next only. It reuses the existing short-chat,
prefill-context-r16, code-context-r64, prefix-primed-r16 and coding-agent-replay
workloads. No new agentic workload, third-party quant selection, cross-engine
comparison or model-quality evaluation is included.

- Host: user-provided Vast pod; 2× RTX 3090 24 GiB, NV4 NVLink between GPUs,
  350 W configured power limits; AMD EPYC 7R13, AVX2, 96 visible logical CPUs.
- Container memory limit: 183,320,444,928 bytes (about 170.7 GiB); host-reported
  RAM is not the container allocation. Host memory is shared with other activity.
- Engine source: `b7f5fdd`, release CUDA build with `W34_ARCH=sm_86`, CUDA 13.2,
  Rust 1.99.1. Engine source was not modified for these measurements.
- Target: `nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ` at
  `6ee06d2936b111f19eb5eb331044cb8b8b28562b`.
- DFlash2 drafter: `nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ` at
  `24a62babba3b54387b3ea73f5599e5de83029dd8`, block size 8, BF16 activations.
  These are pinned prior integration revisions, not moving Hub `main`.
- Both inventories were computed on the pod; target inventory covers
  100,351,844,379 bytes, drafter inventory 556,432,104 bytes. Serialized artifact
  sizes are not loaded VRAM usage. Binary, target, drafter and template hashes
  are recorded in `flash-next-hgptq.toml` and retained run evidence.
- HTTP client and server run on pod localhost; TQ4 KV, 512 pages (8K target pool),
  greedy requests, thinking disabled, workload concurrency one, telemetry at 1s.

This differs from the local 3090 Ti PCIe/AVX-512 environment. Do not attribute
cross-host differences solely to GPU model or quantization. No matched
FP8-PLE-versus-HGPTQ run has been performed on this pod.

## Initial quick run: serving state-budget failure

`flash-hgptq-quick-001` used `--max-num-seqs 1`, one warmup per case and one
measured repetition of the quick preset. Plain and MTP completed both cases on
both GPU counts. DFlash2 failed during the first short-chat warmup on both.

| GPUs | Mode | Short chat tok/s | Agent replay tok/s |
|---:|---|---:|---:|
| 1 | Plain | 29.73 | 24.10 |
| 1 | MTP-3 | 32.67 | 27.30 |
| 1 | DFlash2 | Failed warmup | Not run |
| 2 | Plain | 51.71 | 47.06 |
| 2 | MTP-3 | 80.96 | 73.19 |
| 2 | DFlash2 | Failed warmup | Not run |

These are client-observed completion tokens per wall-clock second, including
prefill/HTTP overhead, not pure decode throughput. Short chat has four measured
requests; replay has two four-turn sessions. One repetition is preliminary.
DFlash2 failure has no throughput value; it is not zero or a successful trial.

A separate raw SSE reproduction retained the exact TP2 error:

```text
CUDA error -10 in no speculative verification state slots
```

The serving planner (`crates/w34-host/src/model/rank.rs`, `serving_state_slots`)
requests `max(8, 4 * max_num_seqs)` slots. Eight slots are insufficient for this
DFlash2 workload; `proposer/scheduled.rs` fails while allocating verification
slots after the first output token. This was a serving-capacity failure, not an
HTTP client timeout. The current client reports generic SSE errors; the raw SSE
reproduction is retained separately to preserve the precise diagnosis.

The refreshed campaign uses **`--max-num-seqs 4` for all six profiles**, requesting
16 serving state slots while leaving workload concurrency at one. This is a
capacity workaround, not an engine fix or a four-concurrent-request benchmark.
Do not pool seq1 results with the revised capacity configuration. Old manifests
retain the original launch commands and hashes.

## State-budget qualification

`flash-hgptq-state-budget-002` reran both DFlash2 profiles with max-num-seqs=4.
Both completed short chat (4/4 requests) and agent replay (8/8 requests):

| GPUs | DFlash2 short chat tok/s | DFlash2 agent replay tok/s |
|---:|---:|---:|
| 1 | 37.60 | 30.76 |
| 2 | 95.86 | 87.30 |

This is a one-repetition serving check, not a controlled speedup against the
seq1 plain/MTP results above. All six profiles must share the revised capacity
setting for the main comparison.

## Full qualification outcome

`flash-hgptq-full-003` completed in about 55 minutes with the revised common
capacity configuration: **6/6 profiles, 90/90 measured trials, 432/432 measured
requests**, zero request failures, and complete token accounting in every trial.
All warmup and priming requests also passed: **648/648 requests across phases**.

| Profile | Completed workloads | Measured trials | Successful measured requests |
|---|---:|---:|---:|
| TP1 plain | 5/5 | 15/15 | 72/72 |
| TP1 MTP-3 | 5/5 | 15/15 | 72/72 |
| TP1 DFlash2 | 5/5 | 15/15 | 72/72 |
| TP2 plain | 5/5 | 15/15 | 72/72 |
| TP2 MTP-3 | 5/5 | 15/15 | 72/72 |
| TP2 DFlash2 | 5/5 | 15/15 | 72/72 |

The primed-prefix cases report nonzero cache reuse on every repetition. DFlash2
reports 5,888 cached prompt tokens per four-request trial versus 6,144 for
plain/MTP (6,212 total prompt tokens in each); do not assume identical cache
reuse across modes merely because the input fixture is the same.

The run establishes that these profiles and existing workloads execute and
produce accounted measurements. The checked-in campaign now marks these pinned
pod deployments `supported`; original manifests retain their pre-qualification
`unverified` status and publication blockers. It does not establish correctness/quality
parity, multi-request concurrency, long-context capacity beyond these fixtures,
or independent-restart statistical confidence. No standings were updated.

[Machine-readable qualification results](flash-next-hgptq.results.json) retain
all three attempts, exact deployment metadata, raw input hashes and provisional
rates. Failed seq1 probes remain failed; they were not rewritten as passes.
The full campaign is the `flash-hgptq-full-003` subset. Regenerate the JSON with
`tools/perf_completion_report.py` from the local evidence root below.

Verification: `mise run perf-check` passed all 91 CPU-only tests and template
validation. The campaign test covers all six launch profiles, independent
checkpoint pins and unchanged quick/full/release workloads. All 126 checksummed
files across the 14 run bundles verified after copying locally. The engine
release CUDA build succeeded on the pod; the full project CUDA test/check suite
was not run (this change is benchmark configuration, tests and documentation).
All benchmark servers were stopped and both GPUs were idle at completion.

## Evidence and limitations

Raw bundles are copied locally under
`benchmarks/perf/results/flash-hgptq-pod-001/` (gitignored), including immutable
request journals, summaries, manifests, coverage, server logs, inventories,
build/hardware evidence and telemetry. Remote run directories live under
`/workspace/`; preserve the local copies before destroying the pod.

Server `/api/metrics` snapshots are cumulative and include warmups. They are not
per-measured-trial acceptance rates. Unique prompts do not establish cold expert
or OS caches. The client reports chunk timing and usage-span TPOT estimates,
not token-level ITL. Independent quality evidence and upstream source-model
revision attestation are not supplied by this performance run; publication
blockers remain explicit. No local GPU benchmarks were run.
