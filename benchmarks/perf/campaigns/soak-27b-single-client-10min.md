# Single-client 27B MTP soak — 10 minutes

## Scope and result

**Passed the configured short-soak checks.** This is synthetic request churn and
recorded-observation replay, not an intact Open-SWE trajectory, autonomous coding
benchmark, full-context workload, long-duration qualification or production sign-off.

- Engine source: `0105610b5ac89ea132b17a208900e91b9f0b6293`, unchanged during the run.
- Executable SHA-256: `54fa14085187344c2ba8b1379d6eb1775365e4bea57fd86b1e6acdbf3a9d1ce7`.
- Target: `nchapman/Qwen3.8-27B-W4A16-HGPTQ@ae51987e3339a2a70070702548571e10fe8bcbc2`.
- One RTX 3090 Ti, GPU UUID `GPU-5327e569-682f-42dc-5cb1-7a0e35373ba3`.
- TQ4, MTP-3, 12,288 KV pages, 196,608-token allocated capacity, one active sequence.
- Fresh server; six warmup responses before measurement. One client in both named
  sustained/burst phases (`--burst-multiplier 1`), not a concurrency burst test.
- Configuration: [`soak-coding-27b-single-client.toml`](soak-coding-27b-single-client.toml).
  Unknown provenance fields remain explicitly unverified.

## Outcomes

Measured duration **600.106 seconds**; **368 measured requests** comprising
**332 completed responses** and **36 expected local disconnects**. No unexpected
errors, invalid usage, stopped sessions or configured latency-ceiling breaches.
The final noncancelled inference probe passed after a five-second quiet drain.
Successful responses generated **52,796 tokens**.

Per-case client latency below uses successful measured responses only, excludes
warmup/disconnects/probe, and uses nearest-rank p95. The configured ceilings were
30 s TTFT / 90 s E2E per response; these are loose operational gates, not approved
interactive SLOs.

| Case | Completed | Prompt tokens | p95 TTFT | p95 E2E |
| --- | ---: | ---: | ---: | ---: |
| Mixed coding | 74 | 185–1588 | 1.180 s | 5.932 s |
| Four-turn coding replay | 185 | 323–1568 | 0.389 s | 2.609 s |
| Shared-prefix traffic | 73 | 830–852 | 0.658 s | 1.945 s |

Agent replay aggregate cached/prompt-token ratios by turn were **0%, 39.6%,
81.2%, 81.0%**. The separate shared-prefix case reported **zero cached tokens**.
No root cause is established: cache retention across interleaved cases needs a
separate investigation. This run does not resolve the near-full-context reuse
failure documented in the capacity report.

## Memory

Post-warmup to post-drain server RSS: **4225.89 → 4487.72 MiB**, growth
**261.83 MiB**, below the configured 512 MiB limit. Observed peak was 4487.72 MiB.
GPU used memory remained **23,257 MiB**, growth **0 MiB**, below the 256 MiB limit.
No competing GPU workload ran. The server stopped cleanly after the receipt.

| Measured window | Median RSS (MiB) |
| --- | ---: |
| 0–60 s | 4332.58 |
| 60–180 s | 4446.64 |
| 180–300 s | 4446.67 |
| 300–420 s | 4474.19 |
| 420–540 s | 4463.71 |
| 540–600 s | 4471.21 |

Most growth occurred early, but the final-minute median remains 24.58 MiB above
the 60–180 s median. Endpoint gates passed; this does **not** establish a permanent
plateau or rule out slow growth. Longer-duration measurement is still needed.

## Reproduction and evidence

Launch the server using the command recorded in the TOML. From `benchmarks/perf`,
with the actual server PID and UUID substituted:

```sh
python3 -m w34perf soak campaigns/soak-coding-27b-single-client.toml \
  --engine warp34-soak --out /tmp/w34-soak-27b-192k-10min \
  --duration-s 600 --min-completed 100 --max-duration-s 900 \
  --sustained-s 60 --burst-s 15 --burst-multiplier 1 \
  --cancel-every 10 --cancel-after-output-events 2 \
  --progress-timeout-s 120 --drain-s 5 --max-error-rate 0 \
  --telemetry-interval 1 --server-pid "$SERVER_PID" --gpu-uuid "$GPU_UUID" \
  --max-rss-growth-mib 512 --max-gpu-growth-mib 256
```

The output directory must not exist. Local evidence:
`/tmp/w34-soak-27b-192k-10min/`; runner log at the matching `.log` path.
All four `SHA256SUMS.json` inventory entries were independently verified.
Summary SHA-256: `206d966c70c75aea73db71d2216006e506992312c649aca9c48281a8a50b42c0`.
These temporary artifacts are not durable release receipts.

Campaign validation, `mise run perf-check` (126 tests), `mise run doc-check` and
`git diff --check` passed. No engine or runner source changed for this campaign;
the full GPU suite was not rerun for these configuration/documentation additions.

## Next qualification

Import a complete single-task SWE trajectory from a pinned dataset revision,
retaining ordered tool observations and source provenance. Existing assembled
Open-SWE fixtures combine tasks and are not a substitute. Recorded observations
paired with generated prose remain replay, not autonomous tool execution.
Then extend duration and prompt lengths while tracking RSS trends, per-turn cache
reuse and latency. Do not extrapolate this short-prompt run to 192 Ki prompts.
