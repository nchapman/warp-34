# Long-context coding-agent campaign

Hardware: 1–2× RTX 3090 Ti, 24 GiB each, `sm_86`; TP=2 crosses PCIe Gen4 x8
with no P2P. Same pinned GPTQ model within each size cohort; FP16 activations,
FP32 SSM, no speculation. FP16 KV and TQ4 KV are different quality/capacity
configurations, not bit-identical alternatives.

See [fixture provenance and limitations](../fixtures/OPEN_SWE.md). The nominal
64K/128K tiers reach approximately **59K/118K** prompt tokens; standalone cold
reads reach **57,612/115,107**. Two repetitions for comparisons and one for
single-card 262K qualification, short 64-token outputs, exploratory
performance measurements only. Values are medians of per-trial statistics.
These traces do not measure successful software repair.

## 9B, TP=1

Revision: `42e302203ae5d3428ce2682d7b443578ef04d5dd`.
All four deployments completed 78/78 measured requests each. Total output tok/s
includes prefill and finite-session fill/drain, not just decode.

| Deployment | 59K agent C1 tok/s | 59K agent C2 tok/s | 118K agent C1 tok/s | Cold 57.6K TTFT s | Cold 115.1K TTFT s |
|---|---:|---:|---:|---:|---:|
| warp-34 FP16 | 23.53 | 21.53 | 13.41 | 16.09 | 40.07 |
| warp-34 TQ4 | 20.30 | 17.58 | 9.69 | 20.26 | 56.43 |
| vLLM 0.31.0 FP16 | 25.98 | 28.74 | 13.75 | 14.86 | 36.39 |
| SGLang 0.5.19 FP16 | 26.14 | 28.81 | 13.62 | 15.26 | 37.59 |

**No overall long-context win:** competitors lead cold prefill and C2 throughput.
Warp-34 FP16 has lower agent C1 median first-output latency (511.80/706.28 ms at
59K/118K vs vLLM 590.37/834.00 and SGLang 628.26/870.63 ms), but that does not
outweigh its cold bursts in total session throughput.

Multi-request 64-query-row dispatch improved measured C2 throughput from
13.43 → 21.53 tok/s (FP16) and 8.41 → 17.58 tok/s (TQ4). The earlier raw warp-34
run directories were overwritten during the experiment, so these before values
are session observations, not a retained independently re-auditable A/B bundle.

Retained bundles under `benchmarks/perf/results/`:
`9b_longctx_warp34_fp16`, `9b_longctx_warp34_tq4`,
`9b_longctx_vllm_fp16`, `9b_longctx_sglang_fp16`.
The 9B competitor runs predate enabling cache-detail flags: their cache accounting
is **unknown**, not zero. Updated TOMLs enable those flags for future runs.
Warp-34 reports approximately 82.3–82.6% token-weighted prefix hits.

## 27B, TP=2

Revision: `0df81eadcfd59601cb82ccece2f16ea7d39244bf`.
Each completed deployment below served 42/42 measured requests.

| Deployment | 59K agent C1 tok/s | 118K agent C1 tok/s | Cold 57.6K TTFT s | Cold 115.1K TTFT s | Warm-turn TTFT 59K / 118K ms |
|---|---:|---:|---:|---:|---:|
| warp-34 FP16 | 12.68 | 6.38 | 36.85 | 86.41 | 901 / 1042 |
| warp-34 TQ4 | 11.16 | 5.55 | 40.19 | 100.21 | 1084 / 1310 |
| vLLM 0.31.0 FP16 | 11.97 | 6.40 | 35.68 | 83.15 | 1231 / 1667 |
| SGLang 0.5.19 FP16 | 12.00 | 6.44 | 36.58 | 85.07 | 1339 / 1307 |

vLLM uses `--enable-prompt-tokens-details`; these warm-turn statistics use actual
reported cache counts. Warp-34 FP16 improves the 59K session throughput and warm
TTFT, is essentially tied at 118K total throughput, and remains slower on cold
reads. SGLang completed all 42/42 requests. Its `--enable-cache-report` emits
null cache details on some requests (including the initial cold request), so total
cache-hit ratio stays unknown. Warm-turn TTFT is conditional on requests with
reported cache counts, not an imputation of missing counts.

Bundles: `27b_tp2_longctx_warp34_fp16`, `27b_tp2_longctx_warp34_tq4`,
`27b_tp2_longctx_vllm_fp16`, `27b_tp2_longctx_sglang_fp16`.

## Single-card 27B, 262K window

Default `tq4`, TP=1 needle at 50% depth: **passed**, exact answer `73918462`,
262,108 prompt tokens in a 262,144-token window, TTFT **592.98 s**, decode
**23.48 tok/s**. `results/needle_262k/27b_tp1_tq4_262144_d50.json`.
This was measured with 336 MiB scratch and requested 1024-token prefill chunks,
before the multi-request dispatch extension. A prior 128 MiB/256-chunk attempt
hit its 480 s timeout and produced no result; it is not a numerical speedup baseline.

The 32,768-window check also passed: 32,732 prompt tokens, TTFT 29.05 s,
42.53 decode tok/s. TQ4-all also passed the 262,144-window, 50%-depth needle:
262,108 prompt tokens, exact `73918462`, **542.18 s** TTFT, **23.88 tok/s** decode
(`results/needle_262k/27b_tp1_tq4all_262144_d50.json`). It ran on GPU 1 concurrently
with default-TQ4 HTTP on GPU 0.

Default-TQ4 **live HTTP passed 29/29 requests**, with the server launched using
`--kv-pages 16384` and **no `--kv-format` override**. It held 23,474 MiB on GPU 0
and served all three agent sessions plus the unique cold read. The largest session
reached **252,112 prompt tokens**, reported **82.7% token-weighted prefix hits**,
and had **3,987 ms** warm-turn p50 TTFT; its last turn reused **250,688 tokens**,
appended **1,424 uncached tokens**, and returned first output in **5,796 ms**.
This qualifies HTTP serving and prefix reuse, not just boot-time allocation.

| Default TQ4 TP=1 case | Req OK/total | Peak prompt tokens | Session output tok/s | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 s |
|---|---:|---:|---:|---:|---:|
| ~59K agent | 9/9 | 59,342 | 7.66 | 1,546 | 18.22 |
| ~118K agent | 10/10 | 118,138 | 3.41 | 2,039 | 42.27 |
| ~252K agent | 9/9 | 252,112 | 0.96 | 3,987 | 134.13 |
| 115K unique cold read | 1/1 | 115,107 | 0.37 | n/a | 171.40 |

TQ4-all **also passed all 29/29 HTTP requests** on GPU 1 at port 18035, with the
same 16,384-page limit. Its largest session also reached **252,112 tokens** and
**82.7% prefix hits**, with **3,603 ms** warm-turn p50 TTFT. The ~59K/~118K/~252K
sessions delivered **7.78 / 3.68 / 1.09 output tok/s**; the unique 115K cold read
had **152.86 s** TTFT. These are capacity/serving measurements, not an isolated
cross-policy quality or performance ranking: the two policies use different KV
precision and ran concurrently on separate GPUs for part of the campaign.

Bundles: `results/27b_tp1_262k_http_{tq4,tq4all}`. Single repetition each, not a
stable tail-latency claim. The **final-code default-TQ4 needle repeat passed**:
262,108 prompt tokens, exact `73918462`, **629.60 s** TTFT, **22.54 tok/s** decode
(`results/needle_262k/27b_tp1_tq4_262144_d50_final.json`). The earlier 592.98 s run
is retained rather than overwritten. Runs overlap other work on the other GPU;
the variation is not a controlled dispatch-regression measurement or a 262K
speedup claim.

## Retained cold-prefill A/B

The pre-64q and 64q-prefetch 27B TP=1 TQ4 runs have identical generated token IDs
`[220, 760, 3841, 13477]`. TTFT **35,128.69 → 28,898.66 ms** (+21.6% prompt
throughput); nsys paged-attention kernel time **12,351.77 → 6,161.82 ms** (~2×).
The latter is kernel time, not a 2× end-to-end claim. JSON copies are retained in
`results/prefill64q-ab/{before,after}.json`; original nsys SQLite traces remain at
`/tmp/nsys_27b_32k_prefill.sqlite` and `/tmp/nsys_27b_32k_prefill_64q_pf.sqlite`.

A clean `git archive a77ce42` release build provides the 9B baseline. On GPU 1,
65,536-token TP=1 TQ4 prefill improved **34,475.33 → 23,329.80 ms**
(**1,900.95 → 2,809.11 prompt tok/s**, +47.8% throughput, 32.3% lower TTFT).
Both generated `[13, 220, 760, 3841]`. Commands used `w34 bench --tp 1 --stage
decode --kv-format tq4 --kv-pages 4224 --prompt-len 65536 --gen-len 4` with the
pinned 9B model; JSONs are `results/prefill64q-ab/9b-64k-{before,after}.json`.
GPU 0 concurrently ran the 27B TP=1 HTTP campaign; there was no GPU co-residency,
but these are not host-isolated benchmark measurements.

## Verification and review checkpoint

- Single-/multi-request 64q-vs-16q parity test passes bit-for-bit for FP16, Q8_0,
  TQ4; GQA 4 and 6; odd chunks and prefix offsets.
- `cuobjdump`: 64q `REG:238 STACK:0 SHARED:43353 LOCAL:0`.
- Python performance harness: 51 tests passed after adding per-step cache/TTFT
  accounting and unknown-cache tests.
- Initial review found one important issue: the original scheduler regression
  did not fail without pinning. Replaced with a three-slot pool requiring two
  evictions: disabling pinning gives cached_tokens=64 instead of 0 and fails;
  restoring pinning passes.
- Arena-margin suggestion partly addressed: compact 16,896 MiB allocation now
  applies only to the extreme-context tier; ordinary 27B retains 16,960 MiB.
  Metadata-derived sizing for other checkpoints is deferred; the extreme tier
  is qualified against the pinned snapshot, not every 64-layer architecture.
- Other suggestions deferred: preserve explicit follow-up `user` strings for
  compatibility (omit/null selects tool-only); saturated-queue LRU statistics
  redesign is out of scope; internal forced-kernel warmup/test ABI accepts
  trusted engine descriptors and is not a public validated API.
- Follow-up review has no important/critical findings. Corrected stale scratch
  comments and narrowed the `tool_choice` documentation to agent replay. Remaining
  suggestions above are deferred, not blockers. The larger scratch tier is an
  intentional throughput/capacity tradeoff, with the pinned 262K runtime checks
  required before qualification; the compact clamp remains a generic fallback.
- Final **`PATH=/opt/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0,1 mise run check` passed**
  (exit 0, 288.44 s): formatting, CUDA/non-CUDA clippy, **216 Rust tests including
  doctests** (0 failed/ignored), **69 oracle harness tests**, **51 perf tests**.
  Log: `/tmp/longctx-final-check.log`, SHA-256
  `41004558ba6bf110eace0eae1e32712596f1ada49c9e43e76aa00b4460ddc23e`.
- All **538/538 HTTP requests** and both final-code 262K needle policies passed.
  Compact machine-readable measurements, expanded-config provenance (minus
  duplicate fixture bodies), raw-file hashes and verification metadata are in
  [`longctx-coding-agent.results.json`](longctx-coding-agent.results.json).
  Raw `results/` bundles remain local/ignored. Frozen `27b-fp16-first.toml` is
  unchanged (`git diff --exit-code` passes).
