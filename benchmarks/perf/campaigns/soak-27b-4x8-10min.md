# 27B MTP soak — four sustained clients / eight burst clients

## Post-fix outcome

**Concurrent batching, MTP speculation, and multi-turn prefix reuse verified under
`4` sustained / `8` burst load.** All error, recovery, pool-conservation, and
resource-growth gates passed, and the three concurrent serving bottlenecks exposed
by the `0105610` baseline run were resolved:

- Target: `nchapman/Qwen3.8-27B-W4A16-HGPTQ@ae51987e3339a2a70070702548571e10fe8bcbc2`.
- One RTX 3090 Ti; TQ4, MTP-3, 4096 shared KV pages (65,536 tokens),
  `--max-num-seqs 4 --max-model-len 16384`.
- Fresh server; actual GDN state pool **24 slots** (up from **12** on `0105610` by
  accounting for desired concurrent MTP serving state + lazy vision reserves in
  `plan_host_embed_with_serving_budget` so `2.37 GiB` `embed_tokens` maps in host
  memory). Four sustained clients, eight burst clients; 60-second sustained /
  15-second burst phases.
- Same synthetic mixed coding, four-turn replay and shared-prefix fixtures as the
  [single-client soak](soak-27b-single-client-10min.md), not intact SWE trajectories.
- [Configuration](soak-coding-27b.toml). Successful measured prompts: **182–1585 tokens**.

| Metric | Baseline (`0105610`) | Post-fix |
| --- | ---: | ---: |
| Measured duration | 627.569 s | 614.802 s |
| Completed measured responses | 340 | 550 (+61.8%) |
| Expected local disconnects | 37 | 61 |
| Unexpected errors / invalid usage / SLO breaches | 0 / 0 / 0 | 0 / 0 / 0 |
| Generated output tokens (measured) | 54,750 (~87.2 tok/s) | 86,798 (~141.2 tok/s, +61.9%) |
| GDN state pool slots | 12 | 24 |
| 1 Hz samples at `running == 4` | 16 / 642 (2.5%) | 586 / 630 (93.0%) |
| 1 Hz samples at `running == 3` | 608 / 642 (94.7%) | 25 / 630 (4.0%) |
| Speculation rounds | 4,959 | 46,817 |
| Drafted / accepted tokens | 1,043 / 791 | 47,390 / 40,508 (85.5%) |
| Sustained p95 TTFT / p95 E2E | 8.199 s / 17.504 s | 1.434 s / 12.528 s |
| Burst p95 TTFT / p95 E2E | 9.949 s / 19.761 s | 6.270 s / 16.551 s |
| Agent-replay cached prompt tokens | 3,840 / ~215k (turn 4: 5.3%) | 144,384 / 305,011 (47.3%; turn 4: 65.0%) |
| Shared-prefix cached prompt tokens | 256 / 52,035 (0.5%) | 5,376 / 81,142 (6.6%) |

## Post-fix concurrency, latency, and prefix reuse

Server metrics sampled once per second recorded a maximum of **4 running** and
**5 waiting** requests. Across 630 samples (`{0: 8, 1: 10, 2: 1, 3: 25, 4: 586}`),
the server sustained four active sequences for **93.0%** of sampled seconds.

Nearest-rank p50 / p95 client latencies, successful measured responses only:

| Admission phase | Responses | p50 TTFT | p95 TTFT | p50 E2E | p95 E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sustained (`4` clients) | 394 | 0.372 s | 1.434 s | 3.360 s | 12.528 s |
| Burst (`8` clients behind `4` slots) | 156 | 3.837 s | 6.270 s | 6.774 s | 16.551 s |

These are admission groups, not isolated fixed-concurrency trials; burst-phase TTFT
includes expected queueing of four waiting clients behind `--max-num-seqs 4`.

| Case | Responses | p50 TTFT | p95 TTFT | p50 E2E | p95 E2E | Cached prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed coding (`cache = "unique"`) | 102 | 0.679 s | 5.842 s | 7.545 s | 19.007 s | 0 / 86,986 (0.0%) |
| Four-turn replay | 351 | 0.364 s | 5.785 s | 5.184 s | 10.657 s | 144,384 / 305,011 (47.3%) |
| Shared prefix (`cache = "natural"`) | 97 | 0.662 s | 5.473 s | 3.917 s | 9.760 s | 5,376 / 81,142 (6.6%) |

Multi-turn replay cache reuse by turn (`coding-agent-replay`, where turn 1 carries a
unique session header):

- Turn 1 (`n = 104`): **0 / 33,385 (0.0%)**
- Turn 2 (`n = 94`): **21,120 / 75,763 (27.9%)**
- Turn 3 (`n = 80`): **49,088 / 81,761 (60.0%)**
- Turn 4 (`n = 73`): **74,176 / 114,102 (65.0%)**

Whole-server counters at post-probe idle (`618` submitted = `557` completed + `61`
cancelled, `0` failed): **46,817 speculation rounds, 47,390 drafted tokens,
40,508 accepted tokens (85.5% acceptance)**, and **172,160 / 534,588 cached prompt
tokens (32.2%)**.

## Root causes fixed

1. **Concurrent MTP GDN state & host-embedding budgeting (`mtp.rs`, `model/engine.rs`,
   `model/rank.rs`):** `mtp::serving_state_slots(max_num_seqs, mtp_k)` now budgets
   `5 * max_num_seqs + verify_extras` (`24` slots at `max_num_seqs = 4, k = 3`:
   `12` active private + `4` locked radix checkpoints + `4` batched verify extras +
   `4` cached burst checkpoints), and `plan_host_embed_with_serving_budget` offloads
   `embed_tokens` (`2.37 GiB`) to `MappedHostBuffer` when GPU-resident embeddings
   would squeeze desired serving state/vision reserves below target.
2. **Queue-bounded MTP routing under `C > max_num_seqs` bursts (`engine_worker/runner.rs`,
   `proposer/scheduled.rs`):** `yield_single_speculation`, `extend_or_yield_mtp_batch`,
   and `admit_requests` now evaluate `desired_active = (active + waiting).min(max_num_seqs)`
   and inspect only the open admission window (`take(open_slots)`) instead of the full
   waiting queue, keeping 4-slot batched MTP active throughout 8-client bursts while
   preserving `B >= 5` plain-decode fallback when `max_num_seqs > 4`. Batched verify
   also reserves extra GDN state slots before running draft forwards.
3. **Warm `PrefixMemo` & slot KV reuse in batched MTP (`proposer.rs`,
   `proposer/scheduled.rs`, `engine_worker/runner.rs`):** Preserved host `PrefixMemo`s
   across single/batch mode switches (`MAX_MEMO_ENTRIES = 32`), tracked per-slot
   `current: Option<PrefixMemo>` across `select_mtp_batch_slot`, flushed final verified
   tokens and committed `PrefixMemo`s on slot completion (`finish_mtp_batch_slot`),
   and replaced batched MTP's `matched == 0` filter with `proposer.can_restore_prefix`
   plus longest-prefix free-slot affinity (`pick_free_mtp_batch_slot`).

## Post-fix memory, pool conservation, and caveats

- Post-warmup to post-drain RSS: **4358.73 → 4866.40 MiB**, **+507.67 MiB**;
  configured gate 512 MiB (baseline RSS is ~2.48 GiB higher than `0105610` because
  `embed_tokens` is intentionally host-mapped in `MappedHostBuffer`, and the `256 MiB`
  `PrefixMemo` cache plus glibc heap arenas warm up across the run). Observed peak
  **4898.13 MiB**.
- RSS medians at 0–60 / 60–180 / 180–300 / 300–420 / 420–540 / 540–600 seconds:
  **4631.24 / 4742.42 / 4756.04 / 4747.89 / 4801.81 / 4880.65 MiB**. Because `+507.67 MiB`
  sits close to the 512 MiB gate and late-window drift remains (~+79 MiB over the final
  two minutes), multi-hour RSS stability still requires longer-duration qualification.
- GPU memory: **22,447 → 22,447 MiB**, **+0.0 MiB** (~666 MiB lower steady VRAM than
  `0105610` despite doubling GDN state slots from 12 to 24); configured gate 256 MiB.
- Post-probe idle pool conservation (`/api/metrics`):
  - KV pages: `capacity = 4096, active = 0, cached = 436, free = 3660` (`436 + 3660 = 4096`).
  - GDN slots: `capacity = 24, active = 0, cached = 16, free = 8` (`16 + 8 = 24`).
- Fixtures remain synthetic coding/replay prompts (`182–1585` tokens), not intact
  long-context SWE trajectories or multi-hour G3 production sign-off.

## Reproduction and receipts

Use the server launch in `soak-coding-27b.toml`. From `benchmarks/perf`:

```sh
python3 -m w34perf soak campaigns/soak-coding-27b.toml \
  --engine warp34-soak --out /tmp/w34-soak-27b-4x8-post-fix-10min \
  --duration-s 600 --min-completed 100 --max-duration-s 900 \
  --sustained-s 60 --burst-s 15 --burst-multiplier 2 \
  --cancel-every 10 --cancel-after-output-events 2 \
  --progress-timeout-s 120 --drain-s 5 --max-error-rate 0 \
  --telemetry-interval 1 --server-pid "$SERVER_PID" --gpu-uuid "$GPU_UUID" \
  --max-rss-growth-mib 512 --max-gpu-growth-mib 256
```

- Post-fix `soak_summary.json` (`/tmp/w34-soak-27b-4x8-post-fix-10min/soak_summary.json`)
  SHA-256: `616c6aee0e6379dca07357f89c3cf9f5bc195dabce6b2ef7620b27b95ad8e1da`.
- Post-fix `manifest.json` SHA-256:
  `5c0a921f2e54327c5cab2fd2973e97e033b7f2b0f4e63e0eac40c8d99c1d3588`.
- Post-fix 1 Hz `/api/metrics` (`/tmp/w34-soak-27b-4x8-post-fix-server-metrics.jsonl`)
  SHA-256: `74b4cfb5ffbf338dfc230f757e527ade9835f3856998e46a5954252b4b01ce29`.
- Post-fix pre/post idle pool receipt (`/tmp/w34-soak-27b-4x8-post-fix-idle-metrics.json`)
  SHA-256: `4bd63ea34e07b03146951de8f72bb182ed0d18629890699b31f34a176f26da5f`.
- Pre-fix (`0105610`) baseline `soak_summary.json` (`/tmp/w34-soak-27b-4x8-10min/soak_summary.json`)
  SHA-256: `49d7b1dc0f4d24c22be5c654eebe31c39e4910c9bf519ff107635da961e5bab6`;
  pre-fix 1 Hz server metrics SHA-256:
  `b5c7618919a1ebb9ef9795d803ecb41bc5fd5246f979e7985ddf3a2fac82fc6e`.
- Temporary `/tmp` raw artifacts are session-local verification evidence, not durable
  release receipts. Strict `CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1 CUDA_DISABLE_PTX_JIT=1 mise run check`
  passed (**508 Rust tests, 0 failures, 13 ignored**), along with `mise run perf-check`
  (126 tests, 10 deployments / 13 cases valid), `mise run doc-check`, and `git diff --check`.
