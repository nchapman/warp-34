# 27B MTP soak — four sustained clients / eight burst clients

## Outcome

**Error/recovery/memory gates passed; concurrent-serving performance is not qualified.**
The server demonstrably admitted four active requests, but usually ran only three,
with poor latency and little cache reuse. Do not interpret the loose harness SLO
pass as interactive readiness or throughput scaling.

- Engine: unchanged `0105610b5ac89ea132b17a208900e91b9f0b6293`.
- Target: `nchapman/Qwen3.8-27B-W4A16-HGPTQ@ae51987e3339a2a70070702548571e10fe8bcbc2`.
- One RTX 3090 Ti; TQ4, MTP-3, 4096 shared KV pages (65,536 tokens),
  `--max-num-seqs 4 --max-model-len 16384`.
- Fresh server; actual GDN state pool **12 slots**. Four sustained clients,
  eight burst clients; 60-second sustained / 15-second burst phases.
- Same synthetic mixed coding, four-turn replay and shared-prefix fixtures as the
  [single-client soak](soak-27b-single-client-10min.md), not intact SWE trajectories.
- [Configuration](soak-coding-27b.toml). Successful prompts: **185–1588 tokens**.

Measured workload plus admitted-session completion lasted **627.569 seconds**
for a 600-second minimum. **340 completed responses + 37 expected disconnects**,
no unexpected errors, invalid usage, stopped sessions or configured SLO breaches.
Peak client sessions **8**; final noncancelled inference recovery probe passed.
Successful requests generated **54,750 output tokens**, approximately **87.2 tok/s**
aggregate. The preceding single-client run observed approximately 88.0 tok/s.
Case/turn mixes differed, so this is not a controlled scaling benchmark, but there
was no observed aggregate throughput improvement.

## Concurrency and latency

Server metrics sampled once per second recorded a maximum of **4 running** and
**5 waiting** requests (independent maxima). Of 642 samples including warmup and
probe, **608 showed three running requests**, and only 16 showed four. The configured
four-slot ceiling therefore did not translate into sustained four-way execution.

Nearest-rank p95 client latencies, successful measured responses only:

| Admission phase | Responses | TTFT | E2E |
| --- | ---: | ---: | ---: |
| Sustained | 203 | 8.199 s | 17.504 s |
| Burst | 137 | 9.949 s | 19.761 s |

These are admission groups, not isolated fixed-concurrency trials. Requests can
span phase transitions. Configured per-response ceilings were 30 s TTFT / 90 s E2E.

| Case | Responses | p95 TTFT | p95 E2E |
| --- | ---: | ---: | ---: |
| Mixed coding | 62 | 8.269 s | 21.584 s |
| Four-turn replay | 216 | 9.657 s | 15.796 s |
| Shared prefix | 62 | 8.485 s | 12.447 s |

Agent turns 1–3 reported zero cached tokens; turn 4 reused **3840 / 71,913 tokens
(5.3%)** in aggregate. Shared-prefix requests reused **256 / 52,035 (0.5%)**.
This is substantially worse reuse than the single-client agent replay.

Last sampled whole-server speculation counters (including warmup/probe) were
**4959 rounds, 1043 drafted tokens, 791 accepted tokens**. Low draft volume,
12 state slots and predominantly three active requests motivate checking admission
and speculative-state fallback together. Counters alone do not prove a root cause.
The last metric sample occurred during the recovery probe; it is not an idle-pool
conservation receipt.

## Follow-up code investigation

Read-only inspection identified two concrete mechanisms; the inspected source
paths are unchanged from the tested `0105610` engine:

- Admission allocates three private GDN states per request. A donated, locked
  radix checkpoint adds ownership beyond those three. With twelve total states,
  three active requests plus even one distinct protected checkpoint can prevent
  allocation of the fourth request's three states. This explains a possible route
  to the observed three-active plateau, not a universally enforced three-slot cap.
- Batched MTP allocates additional verification states per draft across the batch.
  Allocation failure clears that request's drafts and verifies the anchor only.
  Four bare active requests already consume all twelve states, leaving no MTP3
  verification capacity. Admission does not reserve these extras.
- Separately, the MTP batch routing predicate counts waiting requests along with
  active ones; exceeding its four-request limit can yield to ordinary decode.
  Returning to batched MTP requires the ordinary active set to empty. A burst can
  therefore disable speculation even when only four requests execute at once.

Evidence: `scheduler.rs` admission; `model/generation.rs` three-slot allocator;
`radix.rs` checkpoint donation/locking; `proposer/scheduled.rs` batched verify
allocation/fallback; `engine_worker/runner.rs` MTP yield/re-entry predicates.
The current aggregate counters cannot attribute how much time each path consumed.
Next: distinguish admission blocking, state-starved verification and burst-triggered
plain-decode routing, with focused regressions. Do not substitute another generic
KV-capacity sweep for this scheduler/speculation investigation.

## Memory and recovery

- Post-warmup to post-drain RSS: **1881.80 → 1911.00 MiB**, **+29.20 MiB**;
  configured gate 512 MiB. Observed peak **1916.52 MiB**.
- GPU memory: **23,091 → 23,113 MiB**, **+22 MiB**; configured gate 256 MiB.
- RSS medians at 60–180 / 180–300 / 300–420 / 420–540 / 540–600 seconds:
  **1888.21 / 1889.92 / 1891.75 / 1894.51 / 1897.16 MiB**.
  Small upward drift remains; this is not a leak-free long-duration qualification.
- Server stopped cleanly after the probe; no competing GPU work ran.
- The earlier attempted 192 Ki / four-slot server aborted before readiness without
  a diagnostic in captured output. This run uses the smaller workload-appropriate
  pool; no OOM/root-cause claim is made for that separate abort.

## Reproduction and receipts

Use the server launch in `soak-coding-27b.toml`. From `benchmarks/perf`:

```sh
python3 -m w34perf soak campaigns/soak-coding-27b.toml \
  --engine warp34-soak --out /tmp/w34-soak-27b-4x8-10min \
  --duration-s 600 --min-completed 100 --max-duration-s 900 \
  --sustained-s 60 --burst-s 15 --burst-multiplier 2 \
  --cancel-every 10 --cancel-after-output-events 2 \
  --progress-timeout-s 120 --drain-s 5 --max-error-rate 0 \
  --telemetry-interval 1 --server-pid "$SERVER_PID" --gpu-uuid "$GPU_UUID" \
  --max-rss-growth-mib 512 --max-gpu-growth-mib 256
```

All four receipt inventory hashes independently verified. Summary SHA-256:
`49d7b1dc0f4d24c22be5c654eebe31c39e4910c9bf519ff107635da961e5bab6`.
Extra once-per-second server metrics: `/tmp/w34-soak-27b-4x8-server-metrics.jsonl`,
SHA-256 `b5c7618919a1ebb9ef9795d803ecb41bc5fd5246f979e7985ddf3a2fac82fc6e`.
Wrapper: `/tmp/w34-soak-27b-4x8.py`; runner log at
`/tmp/w34-soak-27b-4x8-10min.log`. Temporary raw artifacts are not durable release
receipts. Campaign validation, `mise run perf-check` (126 tests),
`mise run doc-check` and `git diff --check` passed. No engine or runner source
changed for this campaign; the full GPU regression suite was not rerun.
