# 27B + MTP: single-3090-Ti capacity check

> **Historical pre-fix measurements; the default recommendation is withdrawn.**
> The single-client investigation found that embedding placement ignored requested
> target KV capacity. Correcting that decision allowed single-client 262k TQ4
> serving with MTP to start and complete near-262k requests. These earlier failures
> therefore do not establish a hardware context limit. MTP allocation budgeting
> and operating headroom are being qualified separately; concurrent latency/cache
> findings below remain observations, not a diagnosed cause.

## Earlier four-client pressure configuration (not the production default)

The initial **text-only** pressure run on one 24 GB RTX 3090 Ti used:

```sh
export PATH="/opt/cuda/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 CUDA_DISABLE_PTX_JIT=1 ./target/release/w34 serve \
  --model nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --revision ae51987e3339a2a70070702548571e10fe8bcbc2 --offline \
  --speculative mtp --mtp-k 3 --kv-format tq4 \
  --kv-pages 2560 --max-num-seqs 4 --max-model-len 8192 --port 18834
```

This allocates **40,960 shared target KV token slots**, admits at most **four
active sequences**, and caps each prompt plus requested output at **8192 tokens**.
Four maximum-length requests consume 32,768 logical token slots before additional
reservations/cache effects, leaving 8192 shared slots of nominal slack. Eight
clients can submit, but excess requests queue; this is not eight active GPU requests.
MTP is enabled, with the existing concurrency-dependent execution/fallback policy.

This is a **memory-tested starting point, not a guaranteed production-safe maximum
or an interactive-latency qualification**. Vision, sampling, other checkpoints,
longer output budgets and sustained churn require separate tests. Approximately
1 GiB free VRAM is a conservative improvement over the larger tested pools, not
proof against all future allocation paths. A longer-duration 27B soak has not run.

## Environment and capacity probes

- One RTX 3090 Ti, driver `615.71.09`; the second card unused.
- Runtime code from `1769e82`; harness branch `feat/extended-workload-soak`.
- Pinned 27B HGPTQ revision above, TQ4, MTP-3; no separate drafter.
- Fresh server for each configuration; no overlapping GPU jobs.

| Shared token slots (`--kv-pages`) | Requested active limit | Actual GDN pool | Observation |
|---|---:|---:|---|
| 262,144 (16,384) | 8 | Not initialized | Startup rejected: insufficient VRAM for serving state pool and headroom |
| 131,072 (8192) | 8 | 12 slots | Startup succeeds, only 191 MiB free; no load test, not recommended |
| 65,536 (4096) | 4 | 15 slots | 631 MiB free at startup; 26/26 pressure-test responses complete, GPU use rises another 44 MiB; too little margin to choose |
| **40,960 (2560)** | **4** | **16 slots** | **26/26 responses complete; minimum observed free VRAM approximately 1067 MiB** |

The requested active limit alone does not establish executable concurrency: serving
state slots are clamped to available memory, and requests need multiple state
slots. KV, GDN snapshots, weights, MTP and runtime workspaces compete for the same
VRAM. A historical long-context single-request result cannot be applied unchanged
to concurrent MTP serving.

## Pressure workload and results

Two sequential turns per session at client concurrency **1, then 4, then 8**.
All sessions in a group start together. Each has a unique prefix to prevent
cross-session cache hits, repeated synthetic service-design notes, and an actual
256-token generated answer carried into its follow-up. The follow-up should be
able to reuse its own history. `temperature=0`, thinking disabled, `ignore_eos=true`;
256 output tokens per response. This is a synthetic capacity/prefill stress probe,
**not an intact SWE trajectory or representative answer-quality task**.

The 64k-pool run used 11,054 initial / 11,336 follow-up prompt tokens. The chosen
40k-pool run used **7455 initial / 7737 follow-up prompt tokens**; including the
256-token output budget, both fit the 8192 per-request cap. These different prompt
lengths mean the configurations are not a matched performance comparison.

### Chosen 40k pool: observed client experience

| Concurrent clients | Turn | TTFT min / median / max (s) | Median E2E (s) | Total cached tokens |
|---|---|---:|---:|---:|
| 1 | Initial | 5.60 / 5.60 / 5.60 | 7.86 | 0 |
| 1 | Follow-up | 0.10 / 0.10 / 0.10 | 2.56 | 7680 |
| 4 | Initial | 5.60 / 14.03 / 22.53 | 29.95 | 0 |
| 4 | Follow-up | 5.99 / 14.96 / 23.93 | 31.39 | 0 |
| 8 | Initial | 5.75 / 28.84 / 51.93 | 43.38 | 0 |
| 8 | Follow-up | 34.93 / 44.49 / 54.06 | 60.23 | 0 |

All **26 responses succeeded**, producing 6656 output tokens; no server failures
or cancellations. Server telemetry observed up to **4 running / 4 waiting**.
End-of-run active KV/state counts were zero; retained radix cache accounted for
1978 KV pages and eight GDN slots. This observation is not a comprehensive
ownership-conservation proof.

GPU used memory: **23,055 → peak 23,099 MiB**. `nvidia-smi` reported **1067 MiB
free** at peak/end (its reserved-memory accounting means free is not simply total
minus used). Server RSS first / peak / last sampled: **1793.71 / 1956.43 / 1863.56
MiB**. Fresh startup and completed-response recovery passed; server stopped cleanly
and GPU processes were gone afterward. No long-duration plateau claim is made.

### Important performance finding

Both pressure configurations showed **zero reported follow-up cache reuse at four
and eight clients**, despite near-full reuse at one client. With the larger 11k
prompts, eight-client follow-ups took **50.51–78.82 seconds to first output** and
85.28 seconds median E2E. On the chosen shorter-context configuration, they still
took **34.93–54.06 seconds to first output**.

Memory fit is therefore not the main remaining usability gate. Investigate
concurrent prefix/state-cache retention, prefill scheduling and interference with
decode, and queueing before claiming practical multi-agent support. These measurements
do not by themselves establish which subsystem causes cache loss or serialize all
prefill work. Do not infer that increasing the KV pool alone will fix the behavior.
No engine changes or performance fixes were made in this capacity investigation.

## Evidence

Local, not durable release artifacts:

- `/tmp/w34-27b-mtp-capacity-64k/`: request SSE/usage journal, one-second GPU/RSS
  samples, one-second server metrics, before/after summary.
- `/tmp/w34-27b-mtp-capacity-40k/`: same evidence for the chosen candidate.
- Probe scripts: `/tmp/w34-27b-capacity-probe.py` and
  `/tmp/w34-27b-capacity-safe-probe.py`; console logs with matching names.
- `summary.json` SHA-256, 64k:
  `4c868c61069a61c2e2a49394943bc5cff215c182d0a24a381caf60be9a2926e9`.
- `summary.json` SHA-256, 40k:
  `b28e440c13ab674987ab0da1639b0e951793a7aeb05c33d150b25af7712842b4`.

Retain scripts and raw journals in durable storage before using this as a release
receipt. This investigation does not qualify 262k context, eight active requests,
real SWE workloads, multi-GPU scaling, or production SLOs.
