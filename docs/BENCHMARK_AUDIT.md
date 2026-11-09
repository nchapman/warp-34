# Post-merge benchmark audit: `1503ac5`

## Scope and method

Compared pre-merge `db6c65e`, merged `1503ac5`, and `1503ac5` plus the
whole-tile persistent-Marlin optimization in this change. Reviewed the recent
small-GPU dispatch/capacity changes and the XGrammar/tool-parser replacement.

Hardware: Ryzen 9 7950X3D, two RTX 3090 Ti GPUs, 84 SMs each, 450 W power
limits, PCIe PHB topology (no NVLink). GPU jobs were serialized, with
`CUDA_VISIBLE_DEVICES=0,1`, `CUDA_DISABLE_PTX_JIT=1`, and offline cached models.
All binaries used the same local toolchain and shared CUDA libraries. No new
RTX 40-series or small-GPU hardware qualification, competitor runs, or upstream
lm-eval quality scores are claimed here.

Each native run used `--prompt-len 128 --gen-len 64 --kv-format tq4`, default
256 KV pages, CUDA graphs, and the CLI's normal warmup. 4B used `--stage all
--batch-size 8`; other models used `--stage decode`. MTP used `--mtp-k 3`.
This is a repeated synthetic prompt, not representative chat throughput.

Each full sweep below is **one fresh process per cell**. Additional alternating
old/new repetitions tested suspicious differences; the full-sweep table is not
a table of medians. Small differences are not established speedups. Flash-Next
is especially noisy and does not preserve the same generated workload across
all modes/builds.

Targets and locally verified checkpoint revision prefixes:

| Label | Target | Revision |
|---|---|---|
| 4B | `nchapman/Qwen3.5-4B-W4A16-GPTQ` | `d5874365f7` |
| 9B | `nchapman/Qwen3.5-9B-W4A16-HGPTQ` | `7b2e48b7e7` |
| 27B | `nchapman/Qwen3.8-27B-W4A16-HGPTQ` | `ae51987e33` |
| 35B | `nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ` | `4f5de21c88` |
| Flash | `nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ` | `6ee06d2936` |

Drafters: `nchapman/Qwen3.5-4B-DFlash2@8d8bb62873`,
`nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ@46b87ea125`, and
`nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ@24a62babba`.
`tools/prep_test_models.py --check --all` verified every pinned fixture.

## Decode results (tokens/second)

“Final” includes the persistent-tile optimization. All 26 final runs completed,
and **all 26 final generated-token arrays exactly match their corresponding
merged-HEAD arrays**. This does not imply plain/speculative parity for Flash.

| Model | TP | Mode | Old `db6c65e` | Merged `1503ac5` | Final |
|---|---:|---|---:|---:|---:|
| 4B | 1 | Plain | 213.04 | 210.12 | 211.51 |
| 4B | 1 | MTP-3 | 536.91 | 531.66 | 534.20 |
| 4B | 1 | DFlash2 | 1265.26 | 1284.61 | 1288.02 |
| 4B | 2 | Plain | 304.07 | 302.89 | 305.51 |
| 4B | 2 | MTP-3 | 763.14 | 763.88 | 766.01 |
| 4B | 2 | DFlash2 | 1502.95 | 1490.96 | 1488.17 |
| 9B | 1 | Plain | 133.34 | 128.36 | 129.91 |
| 9B | 1 | MTP-3 | 369.03 | 357.35 | 362.76 |
| 9B | 2 | Plain | 213.35 | 199.03 | 204.85 |
| 9B | 2 | MTP-3 | 563.77 | 532.55 | 544.76 |
| 27B | 1 | Plain | 51.83 | 50.48 | 51.90 |
| 27B | 1 | MTP-3 | 127.31 | 124.34 | 127.35 |
| 27B | 1 | DFlash2 | 121.65 | 119.09 | 121.92 |
| 27B | 2 | Plain | 85.63 | 85.16 | 86.31 |
| 27B | 2 | MTP-3 | 231.21 | 230.75 | 232.95 |
| 27B | 2 | DFlash2 | 265.47 | 264.84 | 266.11 |
| 35B | 1 | Plain | 207.60 | 202.05 | 205.21 |
| 35B | 1 | MTP-3 | 452.69 | 433.98 | 446.88 |
| 35B | 2 | Plain | 264.19 | 261.85 | 266.42 |
| 35B | 2 | MTP-3 | 584.85 | 570.67 | 569.46 |
| Flash | 1 | Plain | 34.28 | 33.12 | 33.50 |
| Flash | 1 | MTP-3 | 46.41 | 49.96 | 49.94 |
| Flash | 1 | DFlash2 | 21.03 | 19.70 | 19.65 |
| Flash | 2 | Plain | 63.90 | 58.83 | 69.45 |
| Flash | 2 | MTP-3 | 95.17 | 87.92 | 91.77 |
| Flash | 2 | DFlash2 | 69.38 | 67.30 | 70.37 |

## Confirmed 9B regression and partial recovery

Three alternating old/merged repetitions confirmed the 9B slowdown; output
IDs and MTP acceptance stayed identical:

| Profile | Old median | Merged median | Change |
|---|---:|---:|---:|
| TP1 plain | 133.05 | 128.17 | -3.7% |
| TP1 MTP-3 | 370.95 | 361.48 | -2.6% |
| TP2 plain | 213.89 | 198.81 | -7.1% |
| TP2 MTP-3 | 563.73 | 535.50 | -5.0% |

The recent one-slice Marlin change prevents cross-CTA partial reductions even
when there are more output tiles than SMs. Restoring the old arbitrary
SM-count grid would undermine that correctness fix. Instead, for those
M<=16 shapes, select the largest exact divisor of the tile count between
half the SM count and the full SM count. If none exists, keep one CTA per tile.
With an exact divisor, the existing persistent loop and final stripe both
process complete K tiles: the per-tile reduction tree is unchanged and no
cross-CTA reduction is introduced.

Two isolated candidate runs measured TP2 plain **205.05/205.19 tok/s** and
MTP **543.94/543.13 tok/s**, preserving all tokens. The final sweep confirms
the partial recovery. **9B remains approximately 2–4% below the old build**;
this audit does not claim to have eliminated the regression. The optimization
was not separately hardware-benchmarked on Ada or the smaller Ampere cards.

The added `marlin_w4_persistent_tiles_match_scalar_and_identical_rows` test
covers N=6144/10240/12288/24576, K=256, group sizes 64/128, and M=1/4/8/9/16.
It checks all M=1 columns against an independent scalar reference and every
batched row bit-for-bit against M=1, including the M=8→9 variant boundary.
The 9B HGPTQ/TQ4 `dump-diff --compare-tp2` passed with relative L2 0.041857,
minimum cosine 0.999131, and matching top-1 token.

## Correctness and remaining caveats

- Final 4B, 9B, 27B, and 35B plain/MTP/DFlash output arrays match exactly
  within each TP configuration. All six 4B multistage runs pass cold/warm
  radix parity, vision parity, and image-content cache isolation. B=8
  scheduler runs complete successfully.
- **Flash-Next is not qualified as lossless here.** At merged HEAD and final,
  MTP and DFlash first differ from plain at output index 15 (TP1) or 29 (TP2).
  The old build also has mismatches (TP1 MTP at 15; TP2 MTP at 29 and DFlash
  at 15). Thus the limitation predates these merges, but this does not prove
  the specific cause is harmless rounding. The real-model Flash tests
  already use prefix-only allowances and do not establish full 64-token
  plain/speculative parity. No tolerance or Flash assertion was weakened.
- Flash-Next's hybrid CPU/GPU execution has large timing variability. Two
  extra alternating TP2 plain repeats yielded old 67.29/67.29 versus merged
  71.48/62.94 tok/s. MTP yielded old 97.51/94.57 versus merged 93.06/92.02.
  These are not enough to establish a clean performance regression or win,
  particularly where output tokens differ. The final 69.45 versus initial
  58.83 plain result is **not** claimed as an 18% optimization gain.
- The old 35B TP2 plain run also differs from its MTP output at index 50;
  merged/final plain and MTP match. Three extra plain repeats show essentially
  unchanged throughput (old median 264.17, merged 264.80 tok/s).

## Handoff test correction and verification

The initial full check reproduced the documented failure at
`speculative_handoff.rs:365`. B=5 exceeds the MTP batch limit and uses ordinary
batched decode; the test compared it with B=1 MTP and already contained
architecture-specific, truncated-prefix exceptions. Dense cuBLAS projection
across different M shapes does not guarantee identical reduction algorithms.

Split B=5 into a dedicated comparison against **fresh non-speculative B=5**
with the same prompts/order. Both workers must reach five active requests,
perform zero speculative verify rounds, complete all five 64-token streams
without cancellation, and match duplicate slots. All five complete output
arrays must match between workers. B=2/B=4 exact single-stream parity and the
late-peer handoff tests remain intact. Three additional isolated B=5 repeats
passed. This fixes the fallback oracle; it does not establish the cause of
arbitrary B=1/B=5 differences or promise cross-batch numerical invariance.

Final exact-tree `mise run check`: **476 Rust tests passed, 0 failed, 12
ignored**, plus formatting, both Clippy modes, rustdoc, dashboard, oracle
and performance-tooling checks. The full suite now passes without the former
B=5 baseline exception.

Explicit ignored XGrammar CPU benchmark with the pinned 4B tokenizer:
metadata **303.17 ms**, cold schema **1.194 ms**, cache hit **9.709 µs**,
cached-mask access **0.509 µs/token**, accept/materialize **2.879 µs/token**.
This is one five-token microbenchmark, not an end-to-end comparison or a
statistically established improvement over previous numbers.

HTTP follow-up: 4B TP1, FP16 KV, 4096 pages, plain serving on port 18834,
using the existing `w34perf` smoke and B=8 cases with three repetitions each:
**198/198 measured requests succeeded**. Median aggregate throughput was
**185.19 tok/s** for smoke and **1099.93 tok/s** for B=8. These are finite-trial
HTTP measurements, not comparable to native decode-only tok/s. The generic
suite retains its unknown-provenance/publication warnings; this is a serving
smoke/throughput check, not a competitor ranking. The initial port-8000 launch
failed because another service already owned that port, and the attempted
smoke warmup there was rejected. That failed attempt is retained separately;
the successful run used the newly launched, isolated port-18834 server. The
existing port-8000 service was left untouched and our server was stopped.

Review found no kernel issues. Fixed timeout-result handling in the new B=5
helper. Declined restoring architecture-dependent B=1 prefix assertions:
those test a different numerical geometry, not the fallback contract. The
fixture uses existing burst coalescing and explicitly fails if speculative
rounds occur; repeated runs passed, though that is not a guarantee against
arbitrary host scheduling delays.

## Reproduction and evidence

```sh
export PATH="/opt/cuda/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1 CUDA_DISABLE_PTX_JIT=1
python3 tools/prep_test_models.py --check --all
mise run check
./target/release/w34 bench --offline \
  --model nchapman/Qwen3.5-9B-W4A16-HGPTQ --tp 2 \
  --kv-format tq4 --prompt-len 128 --gen-len 64 --out result.json
# Repeat with --speculative mtp --mtp-k 3; use the target/drafter table above
# for other models. 4B additionally uses --stage all --batch-size 8.
```

Local raw evidence is retained under
`benchmarks/perf/results/native-audit-1503ac5/` (gitignored): full JSON outputs,
command manifests, fixture revisions, build/check logs, alternating repeats,
and candidate measurements. Original scratch evidence is also under
`/tmp/w34-benchmark-audit-1503ac5/`. Historical benchmark outputs and the
pre-existing untracked `outputs/` directory were not modified.
