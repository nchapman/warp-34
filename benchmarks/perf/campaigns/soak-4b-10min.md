# Single-GPU 10-minute workload check — 4 sustained / 8 burst

## Scope and outcome

**Passed the configured harness checks; no obvious runaway memory growth observed.**
This is a short memory/churn and client-latency check, not an eight-hour soak,
long-SWE-session qualification, answer-quality result, or production sign-off.

- Runtime: `1769e82`, unmodified Rust/CUDA engine; new soak runner on
  `feat/extended-workload-soak` (source digest below).
- Hardware: one RTX 3090 Ti, driver `615.71.09`, `CUDA_VISIBLE_DEVICES=0`;
  `CUDA_DISABLE_PTX_JIT=1`. Other card unused. GPUs idle before the campaign.
- Model: `nchapman/Qwen3.5-4B-W4A16-GPTQ`, cached revision
  `d5874365f7093b28b69379722d2f45d660fcbca6`; TP1, TQ4, no speculation,
  greedy sampling, thinking disabled.
- Fresh server: 4096 KV pages, max 8 active sequences, max context 8192.
- Workload: [`soak-coding.toml`](soak-coding.toml), all three cases;
  synthetic coding tasks, four-turn recorded-tool replay, shared-prefix traffic.
  Actual generated answers enter subsequent replay history. No live tools execute.
- Workload target: both 600 seconds and 1000 successful responses, hard measured
  limit 900 seconds; 60 seconds at concurrency 4 then 15 seconds at concurrency 8,
  repeating. Local disconnect every tenth measured request after two output events.
- Configured budgets: zero unexpected errors; TTFT <=30 seconds and E2E <=90 seconds
  per response; post-warmup/post-drain RSS growth <=512 MiB and GPU growth <=256 MiB.
  These are permissive smoke ceilings, **not approved interactive production SLOs**.

## Results

| Measurement | Result |
|---|---:|
| Measured workload including admitted-session completion | 603.89 s |
| Measured requests | 2140 |
| Successfully completed responses | 1926 |
| Intentional local disconnects | 214 |
| Unexpected errors / invalid usage / stopped incomplete sessions | 0 / 0 / 0 |
| Completed sessions / all measured sessions | 1000 / 1214 |
| Prompt / generated tokens on successful responses | 1,782,429 / 368,655 |
| Peak client in-flight sessions | 8 |
| Configured latency breaches | 0 |
| Post-drain noncancelled inference probe | Passed |

Warmup (six responses across three sessions) and final probe are excluded from
measured response counts. An intentional disconnect ends its replay session.
The server stopped cleanly after the campaign, and no GPU compute processes
remained. Client in-flight sessions are not a measurement of GPU-active requests.

## Memory trajectory

One-second telemetry; RSS is the server process, GPU memory is the selected UUID.
Baseline is after warmup, final reading is after a ten-second quiet interval and
before the final inference probe.

| Resource | Baseline | Final | Growth |
|---|---:|---:|---:|
| Server RSS | 1648.82 MiB | 1660.35 MiB | +11.53 MiB |
| GPU memory | 6351 MiB | 6371 MiB | +20 MiB |

| Measured window | RSS median / max (MiB) | GPU median / max (MiB) |
|---|---:|---:|
| 0–60 s | 1656.30 / 1656.82 | 6367 / 6367 |
| 60–180 s | 1657.55 / 1659.41 | 6371 / 6371 |
| 180–300 s | 1659.77 / 1659.92 | 6371 / 6371 |
| 300–420 s | 1660.05 / 1660.18 | 6371 / 6371 |
| 420–540 s | 1660.34 / 1660.35 | 6371 / 6371 |
| 540–600 s | 1660.36 / 1660.36 | 6371 / 6371 |

Most growth occurred early; GPU memory settled and RSS growth slowed substantially.
The final-minute RSS median was approximately 0.31 MiB above the 300–420-second
median. This is encouraging, **not proof of leak freedom or a basis for linear
extrapolation**. No internal allocator/pinned-memory conservation was measured.

## Client latency and cache behavior

Computed from successful measured responses only; intentionally disconnected
responses remain counted separately above. Percentiles use linear interpolation.
Times are client-observed, including server scheduling and transport.

Phase labels identify **session admission phase**, not instantaneous concurrency:
a replay admitted during a burst can continue into a sustained phase. These are
observational groups, not matched isolated concurrency trials.

| Admission group | Responses | TTFT p50 / p95 / p99 | E2E p50 / p95 / p99 |
|---|---:|---:|---:|
| All | 1926 | 72 / 268 / 354 ms | 1.48 / 2.85 / 4.09 s |
| Sustained | 1398 | 70 / 263 / 291 ms | 1.41 / 2.66 / 3.52 s |
| Burst | 528 | 100 / 336 / 426 ms | 1.67 / 3.49 / 5.01 s |

| Workload | Responses | TTFT p95 | E2E p95 |
|---|---:|---:|---:|
| Mixed coding prompts/output lengths | 386 | 334 ms | 4.09 s |
| Four-turn coding-agent replay | 1163 | 165 ms | 2.56 s |
| Shared-prefix coding prompts | 377 | 197 ms | 1.45 s |

Output-span TPOT estimate p50/p95/p99: **6.52 / 10.19 / 11.66 ms/token**.
Pooled output-bearing SSE event gap p50/p95/p99/max:
**5.40 / 12.35 / 49.85 / 102.53 ms**. Events can contain multiple tokens;
these are **not exact per-token ITL measurements**. TPOT divides first-to-last
output span by server-reported completion tokens minus one.

Agent cache accounting (all cache fields present):

| Agent turn (1-based) | Successful responses | Token-weighted cached/prompt fraction | TTFT p95 |
|---|---:|---:|---:|
| 1 | 380 | 0.0% | 146 ms |
| 2 | 286 | 45.8% | 168 ms |
| 3 | 260 | 84.9% | 181 ms |
| 4 | 237 | 83.5% | 167 ms |

First-turn prefixes are unique; later turns reuse session history. Fractions are
**cached tokens divided by prompt tokens**, not request-level cache-hit rates.
Later-turn sample counts shrink because injected disconnects stop sessions. These
short histories do not establish long-context SWE behavior. Output caps are part
of the workload; completion does not imply the generated answer solved the task.

There is a visible latency cost during burst-admitted sessions, but no multi-second
first-output tail in this workload. Without an isolated concurrency-1 baseline and
server queue/prefill attribution, these data do not identify a scheduling bottleneck
or establish acceptable production agent latency. Next: intact Open-SWE trace replay
at fixed context policy and concurrency 1/4/8, rather than enlarging synthetic
assembled traces and claiming faithful trajectories.

## Reproduction and evidence

Server (repository root):

```sh
CUDA_VISIBLE_DEVICES=0 CUDA_DISABLE_PTX_JIT=1 ./target/release/w34 serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ --offline --kv-format tq4 \
  --kv-pages 4096 --max-num-seqs 8 --max-model-len 8192 --port 18834
```

Client (`benchmarks/perf`; actual PID was 1450097):

```sh
python3 -m w34perf soak campaigns/soak-coding.toml \
  --engine warp34-soak --out /tmp/w34-soak-single-gpu-10min \
  --duration-s 600 --min-completed 1000 --max-duration-s 900 \
  --sustained-s 60 --burst-s 15 --cancel-every 10 \
  --progress-timeout-s 120 --drain-s 10 --telemetry-interval 1 \
  --server-pid "$SERVER_PID" --gpu-uuid "$GPU_UUID" \
  --max-rss-growth-mib 512 --max-gpu-growth-mib 256
```

- Artifacts: `/tmp/w34-soak-single-gpu-10min/` (local, not durable release storage).
- `soak_summary.json` SHA-256:
  `55ea4f5d4f1606ab1b044a6612dd6dacaeee5b93c27a9188f26e4cc34f6eb7e6`.
- Canonical config SHA-256:
  `dc338c979c4c3129658c7806235f6b91ac234ae730739114c9d4e8a0e3c67637`.
- Runner source-map digest:
  `3604a0a2a90bfb290c706ee4d2d7f6a7b0ddc0b49d1c6ce98f1a146856cc1a3d`.
- All files in `SHA256SUMS.json` verified against the retained bytes.

The receipt and raw journals retain configuration, source provenance, request
usage/event timing and resource samples. Upload the raw bundle to durable storage
before treating this summary as release evidence. The existing finite-campaign
`report` command does not render soak receipts; the latency/window tables above
were derived directly from the retained journals.
