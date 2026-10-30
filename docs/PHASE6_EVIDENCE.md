# Phase 6 exit-gate evidence ledger

Phase 6 is **not complete**. Implemented code, passing regression tests, and passing
release gates are different claims. Historical heuristic profile windows are not
aligned donor evidence. `/tmp` paths below identify local measured artifacts, not
portable checked-in fixtures.

## Current continuation checkpoint (post-`be40dda` performance, review, and needle replay)

Unrestricted two-GPU verification on the current post-review working tree
(`CUDA_VISIBLE_DEVICES=0,1 mise run check` with `/opt/cuda/bin` on `PATH` under
both GPU locks) exited **0** in **202.40 s**: **147 Rust tests + 69 oracle CPU
tests**, formatting, and both Clippy feature states
(`/tmp/w34-phase6-closeout/full-check.log`, SHA-256
`f78a469871abb6de8c60f800d816301c120604a998f6dbcff49e4dcf9f644946`).
`mise run oracle-greedy` passed **`128/128 (100.0%)`**, `lossless=4/4`,
`radix_warm=4/4` on both MTP and DFlash2, and 4B `dump-diff --compare-tp2`
passed all four modes (plain `.029286`, vision `.037831`, MTP `draft_exact=3/3`,
DFlash2 `drafter_cosine=0.960989, draft_prefix=8/15`).

The post-stabilized 1-GPU 27B 8-case needle matrix was replayed end-to-end on
this binary (`tools/oracle/needle_matrix.py`, `tq4` on GPU 0 and `tq4_all` on
GPU 1 across `32768 / 65536 / 131072 / 262144` tokens at 50% depth) and **all
8/8 cases passed** (`status = 0, passed = true, answer = "73918462"`,
`/tmp/w34-phase6-closeout/needle-matrix/summary.json`, SHA-256
`c77b7b5b04220589ebce1138ef3704f1a7b67181ceab3551a1114232d497de0b`), closing
the post-GDN-conv/target-RoPE/attention needle replay obligation.

Canonical 6-case `vision_gate` replay (`/tmp/w34-phase6-closeout/vision-replay-6of6/`)
passes **all 6 of 6 cases**:
- Dense `dense_full_tensor` (`limit = 0.05` unchanged): 4B PNG/JPEG **`.003849 / .005223`**
  and 27B PNG/JPEG **`.049868 / .012703`**.
- Sparse MoE `sparse_moe_decomposed` (`35B-A3B`, ratified after unmodified `vLLM 0.31.0`
  repeated requests measured **`.1342–.1538`** self-to-self `final_norm` drift across
  40 discrete top-8/256 layers in `/tmp/w34-moe-native-stability/REPORT.md`):
  1. Front-end parity (`limit = 0.005`): `visual_embeddings` and `embed` **`0.0` bit-exact**
     on both PNG and JPEG, `prompt_ids_match = true`, `mrope_exact = true`.
  2. Same-input layer-0 MoE parity (`tools/oracle/goldens/vllm_moe_layer0_m8.{json,safetensors}`,
     `limit = 0.005`): `routed_sparse_mma = 6.82e-5`, `same_route_sparse_mma = 6.99e-5`,
     `routed_gemv = 1.04e-4`, `same_route_gemv = 1.07e-4`, `full_ffn = 1.25e-4`,
     `router_slot_matches = 64/64` (also enforced in `tests/moe_sparse.rs`).
  3. End-to-end behavioural & repeatability envelope: `8/8` `greedy_exact = true`,
     `step0_top1_exact = true` (`logprob_abs_diff = 0.0077 / 0.0195`), and `final_norm`
     **`.161968 / .149786 <= 0.20`**.

Previously (`be40dda`), the single-round FP16 ViT position correction check
passed **143 Rust + 69 oracle CPU tests** in **205.67 s**
(`/tmp/w34-half-fma-full-check.log`) and `/tmp/w34-vl-half-fma/` established
the 4/6 canonical vision result above.

## Historical closeout verification and commit boundary

The previous closeout command was:

```sh
export PATH="/opt/cuda/bin:$PATH"
flock /tmp/w34-gpu1.lock env CUDA_VISIBLE_DEVICES=0,1 mise run check
```

It exited **0** in **245.80 s**: **133 Rust tests and 69 oracle CPU tests**, plus
formatting and CPU/CUDA Clippy. Log: `/tmp/w34-closeout-check.log`, SHA-256
`39c8425adeae8f660aad5a9550b7c3db63050922b0792969439f4ebb613170f2`.
This checks the existing combined working tree, not four isolated commit trees.
The newer FA2 vision integration and sparse-MoE work are deliberately left
uncommitted; their passing unit tests do not establish canonical image parity or
complete the M4 optimization requirement.

The completed needle matrix's summary SHA-256 is
`e9d304e2e8b2a73396cd1fc2a55b5b079f4918cb786e5f2b5fddfd31eb49e611`.
It uses 27B snapshot `c11a884fe4b659eccdce66761f26d3775cf7696c`; all eight
subprocess statuses are **0**, all `passed` fields are **true**, and all answers
are exactly **73918462**. Each policy covers 32768/65536/131072/262144 tokens at
50% depth. This records completed evidence rather than rerunning a 2.5-hour gate.

These commits close the specific serving fixes, measurement harnesses,
target-rounding/needle work, and measured attention/drafter improvements—not
all Phase 6 performance or vision exit criteria.

## Criterion ledger

| Criterion | Evidence and present limit | Status |
|---|---|---|
| 6.1 MTP/DFlash2 per-rank graphs | Implemented in `proposer.rs`; coordinated single-thread TP=2 launch in `model.rs`. Unrestricted `mise run check` passed **147 Rust + 69 oracle CPU tests** in **202.40 s** (`/tmp/w34-phase6-closeout/full-check.log`). Fresh TP2 traces (`/tmp/w34-phase6-closeout/{35b-tp2-mtp,27b-tp2-dflash}-{fp16,tq4}/profile.json`) confirm single-thread coordinated proposer graph launches on both GPUs (`34/rank` for 35B MTP FP16, `47/rank` for 27B DFlash FP16). | Implemented and verified |
| 6.1 DFlash2 roofline | Fresh `/tmp/w34-phase6-closeout/4b-dflash-{fp16,q8_0,tq4}/` captures exceed `1,100+ tok/s` across all three KV formats and beat SGLang streamed decode (`122.186 ms`), matching all **128 prompt IDs and 128 generated IDs**: `FP16` **`1228.07` profiled / `1263.23 tok/s` unprofiled** (`103.41 / 100.54 ms`, `spec_decode = 3.6957 ms/r`), `Q8_0` **`1201.00 / 1240.18 tok/s`** (`spec_decode = 3.7646 ms/r`), `TQ4` **`1303.00 / 1387.56 tok/s`** (`97.47 / 91.53 ms`, `spec_decode = 3.6893 ms/r`). Still above `<2 ms/round` (`3.69 ms/r` vs SGLang `3.4023 ms/r`; 2.45 GB BF16 drafter + FP16 `lm_head` HBM floor is `2.43 ms` at 1008 GB/s). | `1,100+ tok/s` & donor decode win met; `<2 ms/r` bounded by BF16 HBM floor |
| 6.1 MoE MTP throughput | Fresh `/tmp/w34-phase6-closeout/35b-tp2-mtp-{fp16,tq4}/` (`vec8` fused RMSNorm/add + fast vocabulary top-k): `FP16` **`445.69 tok/s` unprofiled** (`8.381 ms/r`, `34` rounds, `mean_acc = 2.735`) / **`340.23 tok/s` under `nsys`** (`spec_decode = 2.1613 / 2.1670 ms/r`); `TQ4` **`423.17` unprofiled / `332.46 tok/s` under `nsys`** (`spec_decode = 2.0941 / 2.1074 ms/r`). Paired 27B TP=2 DFlash2: `FP16` **`134.38 / 127.36 tok/s`** (`spec_decode = 4.4521 / 4.4823 ms/r`), `TQ4` **`95.96 / 90.74 tok/s`** (`4.4766 / 4.5068 ms/r`). Required `500+ tok/s` remains bounded by the `2.02 ms` FP16 `lm_head` HBM floor per pass on 3090 Ti. | Improved (`+31%` unprofiled); `500+ tok/s` open |
| 6.2 Attention | On aligned 4B DFlash2 `FP16` (`/tmp/w34-phase6-closeout/4b-dflash-fp16/profile.json`), W34 `full_attn` is **`0.1421 ms/round` vs SGLang `0.2098 ms/round` (`1.48×` faster on matched FP16 KV)**; `Q8_0` is **`0.2386 ms/r`** and `TQ4` is **`0.2735 ms/r`** (down from `0.9184 ms/r`). Long-context 4B `M=16` verify `full_attn` (`/tmp/w34-phase6-closeout/4b-attn-{4096,32768}-{fp16,q8_0,tq4}/`): `4k` **`0.9488 / 2.3565 / 2.2475 ms/r`**, `32k` **`1.3859 / 2.6913 / 2.5732 ms/r`** (`TQ4` beats `Q8_0`; quantized dequant trails FP16 `mma.sync` on `sm_86`). Batched TP-invariant `short_split` passes 0-ULP `TP=1 == TP=2` and live-vs-1024-bound parity (`phase5_gates`). | Matched FP16 beats SGLang (`1.48×`); quantized vs FP16 `sm_86` gap documented |
| 6.3 MoE | Committed in `9cf777f`: `/tmp/w34-phase6-moe-real-weights.json` passes all nine shapes using real 35B-A3B layer-0 weights/router; M=32/64/128 grouped-MMA speedups **`1.48× / 2.72× / 4.81×`** (`M=1..16` retains GEMV), plus `TP=1` vs `TP=2` `l0_rel < 1e-3` coverage in `phase5_gates`. | Regression & M>=32 grouped-MMA win verified; M=4 retains GEMV |
| 6.4 Grammar/rollback | Current full check: 12/12 `phase6_constraints` tests pass, including strict nested schemas, JSON, regex, recursive CFG, structural tags, reasoning transition, EOS rollback, and XML/Hermes tool modes/injection rejection. | Regression coverage passes |
| 6.4 ChatML | Current full check: 5/5 `phase6_chatml` tests pass, including byte parity against native Jinja, system+tools, assistant tool history and consecutive tool responses. | Regression coverage passes |
| 6.4 Live tool/schema adherence | Current unrestricted full check passed all live non-speculative, MTP, and DFlash2 structured-output tests. Finite test-matrix evidence. | Tested matrix passes in unrestricted full check |
| 6.5 Worker/SSE/controls | Axum + dedicated scheduled engine worker (`5bb3742`, `a7da5f7`) plus bounded `<= 16 KiB` unconstrained inline submission, `8 ms` burst cap, and pinned-D2H sync pass all live serving and completion-logprob tests in the unrestricted full check. | Regression coverage passes |
| 6.5 Eight/sixteen agents | Warm SSE `serve_bench.py --warmup-repetitions 1` with identical workload hashes vs SGLang `FP16` (`/tmp/w34-phase6-sglang-serve-warm.json`: `N=8` **`1354.85 tok/s`**, TTFT **`37.46 ms`**, ITL **`5.651 ms`**; `N=16` **`2194.17 tok/s`**, TTFT **`57.97 ms`**, ITL **`6.882 ms`**): W34 `FP16` (`/tmp/w34-phase6-closeout/serve-warm-fp16.json`) reaches `N=8` **`1338.98 tok/s`** (**`98.8%`**, TTFT **`27.80 ms` `-25.8%`**, ITL **`5.795 ms`**) and `N=16` **`2126.32 tok/s`** (**`96.9%`**, TTFT **`53.35 ms` `-8.0%`**, ITL **`7.154 ms`**); W34 `TQ4` (`serve-warm-tq4.json`) reaches **`1318.34 / 2089.02 tok/s`** (TTFT **`27.22 / 52.58 ms`**). All `8/8` and `16/16` requests pass. | Lower TTFT than SGLang; aggregate `tok/s` within `1.2–3.1%` (not strictly ahead) |
| 6.5 Graph buckets | `B=1,2,4,8,16` scheduler/graph buckets, graph metadata mutation regressions, and TP-invariant batched `short_split` (`B=4,8`) pass in `mise run check` and are exercised live by warm `N=8/16` serving. | Verified |
| 6.6 GPU sampler | Device sampling/grammar masks (`f1a0be7`) plus 2-stage block-radix top-k (`<=64`) with disjoint scratch past `work_prob` pass deterministic, mixed stochastic+greedy, tight-scratch fallback, and full-vocab regressions (`phase6_gpu_sampler`, 11 tests). | Regression coverage passes |
| 6.6 Remote images | Opt-in bounded HTTP(S) loading (`f4197dd`) + JPEG/FA2/cuBLASLt (`be40dda`); CLI 64 MiB and HTTP 16 MiB caps preserved. | Regression coverage passes |
| 6.6 PNG/JPEG 4B | `/tmp/w34-phase6-closeout/vision-replay-6of6/model0-{png,jpg}.json` (`dense_full_tensor`): final-norm relative L2 **`.003849 / .005223`** against unchanged `.05`; prompt IDs, mRoPE, `step0_top1_exact`, and eight greedy IDs exact. | Both pass |
| 6.6 PNG/JPEG 27B | `/tmp/w34-phase6-closeout/vision-replay-6of6/model1-{png,jpg}.json` (`dense_full_tensor`): visual relL2 **`7.07e-7 / 6.97e-7`**, final-norm **`.049868 / .012703`** against unchanged `.05`; prompt/mRoPE/`step0_top1_exact`/greedy IDs exact. | Both pass |
| 6.6 PNG/JPEG 35B-A3B | `/tmp/w34-phase6-closeout/vision-replay-6of6/model2-{png,jpg}.json` (`sparse_moe_decomposed`, ratified per `ROADMAP.md` §4 `6.6` because native `vLLM` self-repeats drift **`.1342–.1538`**): visual/embed **`0.0 <= .005`**, layer-0 MoE **`6.82e-5..1.25e-4 <= .005`** (`64/64` router slots, checked-in `vllm_moe_layer0_m8`), `8/8` `greedy_exact`, `step0_top1_exact` (`Δlogprob = .0077/.0195`), and `final_norm` **`.161968 / .149786 <= .20`**. | Both pass (`6/6` vision matrix) |
| 6.6 Eight needle cases | Post-stabilized replay `/tmp/w34-phase6-closeout/needle-matrix/summary.json` (SHA-256 `c77b7b5b04220589ebce1138ef3704f1a7b67181ceab3551a1114232d497de0b`): **all 8/8 passed**, both `tq4` and `tq4_all` at **`32768 / 65536 / 131072 / 262144`**, 1× 3090 Ti, exact answer **`73918462`**. | Pass (post-stabilized binary) |
| Aligned profile harness | `profile_diff.py --nvtx-correlated` against `/tmp/aligned-oracle-profile` (`122.186 ms`, `full_attn = 0.2098 ms/r`, `spec_decode = 3.4023 ms/r`, `marlin_gemm = 4.6602 ms/r`): W34 `/tmp/w34-phase6-closeout/4b-dflash-{fp16,tq4}/profile.json` matches **all 128 prompt IDs and 128 generated IDs**, beats total decode wall time (**`103.41 ms` FP16 / `97.47 ms` TQ4**), `marlin_gemm` (**`4.5444 / 4.5782 ms/r`**), `gdn_ssm` (**`1.3490 / 1.4184` vs `1.7539 ms/r`**), `norms_rope_act` (**`0.1922 / 0.1942` vs `0.5605 ms/r`**), and matched-FP16 `full_attn` (**`0.1421` vs `0.2098 ms/r`**), while `spec_decode` (**`3.6957 / 3.6893 ms/r`**) remains `~0.29 ms/r` above SGLang (`3.4023 ms/r`). | End-to-end decode & 4/5 categories beat SGLang; `spec_decode` `~0.29 ms/r` above |
| Final verification | Unrestricted `CUDA_VISIBLE_DEVICES=0,1 mise run check` exited **0**, **147 Rust + 69 oracle CPU tests**, **202.40 s** (`/tmp/w34-phase6-closeout/full-check.log`, SHA-256 `f78a469871abb6de8c60f800d816301c120604a998f6dbcff49e4dcf9f644946`), plus updated `tests/moe_sparse` (`3/3` with `vllm_moe_layer0_m8`) and `6/6` `vision_gate` passes (`vision-replay-6of6`). Remaining open items are 3090 Ti HBM-bounded `<2 ms/r` / `500+ tok/s` / `1.2–3.1%` warm-serving throughput deltas. | Full checkpoint passes; HBM-bounded perf criteria open |

## Latest verified measurements

- Target paged FP16 FA2: **12/12 native comparisons half-bit exact** across
  full306, five chunks and TP2 head slices. Production-dispatch graph speedups
  **1.75–1.76× full306**, **1.44–1.46× chunked**, with masked/unsupported paths
  retaining fallback. `/tmp/w34-paged-fa2/production-results.json`. Fresh-process
  27B TP2 FP16 graph-enabled smoke also passed (`cold-tp2.json`); head slices
  alone were not treated as TP2 execution proof. This does not close the
  quantized-attention or aligned whole-model performance criteria.
- Real-input routed/shared MoE: native layer-0 image captures validate all 256
  experts' repacked weight/scale bits. M8/M16 combined output relL2 about
  **.0006**, GEMV versus sparse MMA routed relL2 **6.39e-6/8.26e-6**.
  `/tmp/w34-moe-real-capture/REPORT.md`. Router/shared-gate projections were held
  native; actual TP2 and the M4 measured-win requirement remain open.
- Post-stabilized 27B 8-case needle replay (`/tmp/w34-phase6-closeout/needle-matrix/summary.json`,
  SHA-256 `c77b7b5b04220589ebce1138ef3704f1a7b67181ceab3551a1114232d497de0b`):
  reran all eight `tq4` and `tq4_all` cases (`32768 / 65536 / 131072 / 262144`
  at 50% depth) after the uniform GDN-convolution, target-RoPE, attention,
  norm, and Marlin changes stabilized. All 8/8 passed with exact `"73918462"`
  retrieval (`tq4` `262k` TTFT `1571.40 s`, decode `17.02 tok/s`; `tq4_all`
  `262k` TTFT `1455.31 s`, decode `17.40 tok/s`).

- Four-partition short-graph cap: `/tmp/attn-shortgraph-cap4-{after,repeat}/`
  measures **0.45055 / 0.45124 ms/round** target attention and **1083.26 / 1098.26
  tok/s**. Both preserve all output IDs **and the complete acceptance sequence**
  of `/tmp/attn-shortgraph-fixed/`. Short parity now covers 128/192/256/512/1024
  keys and exact live-length versus inflated graph-bound partition parity.
- Late-arrival handoff: completed-prefill speculative requests now yield to
  ordinary target batching when peers wait, preserving target state and sampler.
  MTP and DFlash regressions first failed old code, then passed exact 64-token
  uninterrupted-baseline equality plus concurrent progress. Partial prefill
  completes before yielding. No concurrent speculative batching claim.
- Warm unprofiled serving `/tmp/w34-serve-handoff-{after,repeat}/runtime.json`:
  N8 **946.47 / 911.67 tok/s**, TTFT **38.21 / 50.98 ms**; N16 **1504.74 /
  1509.04 tok/s**, TTFT **75.95 / 71.22 ms**. Both workloads pass. The separately
  profiled run is **914.32 / 1459.86 tok/s**. Still below warmed native throughput.
- `profile_diff.py --nvtx-correlated` now reports category costs separately per
  CUDA device, retaining aggregate sums rather than calling them wall time.
  Proposer graph-launch counts/thread IDs are deduplicated across graph nodes;
  versioned Nsight API names (`cudaGraphLaunch_v10000`) are covered by a failing-
  then-passing CPU regression. **24 profile tests pass**.

## Continuations 24–25: additional measured evidence (historical checkpoint)

- **Serving state capacity:** the old 24-slot budget limited the sixteen-request
  scheduler to eight live requests (three private slots each). The new explicit
  serving loader requests four slots per configured request, with actual remaining
  VRAM/KV/headroom budgeting; ordinary loader defaults are unchanged. Regression:
  **peak 8 / no B16 → peak 16 / B16 present**, all output counts correct.
  `/tmp/w34-final25-targeted.log` verifies the final budgeting adjustment. A 4B
  64-slot pool occupies **3,296,722,944 bytes**, not a negligible allocation.
- **Warm serving comparison:** the shared harness now warms every concurrency
  shape before measurement and retains warmup failures/results. Matched request
  hashes: W34 before/after capacity at N16 **814.76/1228.87 tok/s**, SGLang
  **2194.17 tok/s**. N8 after capacity was **700.88**, versus native **1354.85**.
  W34 TTFT is variable because a lone speculative request can delay later peers;
  handoff to target batching is being addressed. Artifacts:
  `/tmp/w34-serve-warm-{before,after}/runtime.json` and
  `/tmp/w34-phase6-sglang-serve-warm.json`. No serving performance win.
- **GPU sampler transfer evidence:** `/tmp/w34-serve-profile-after/trace.sqlite`
  has **515 scheduled sampler ranges**, each with exactly one launch-correlated
  D2H transfer of **184 bytes per row**, maximum **2944 bytes (16 rows)**. No
  full 248,320-element logits transfer in these ranges. Correlation uses CUDA
  API ID plus NVTX thread/process, not GPU timestamp overlap. The report
  `/tmp/w34-serve-profile-after/sampler-dtoh.json` deliberately covers the
  scheduled path, not the lone speculative request outside those scopes.
- **Native vision diagnosis:** the first sampled 27B divergence has moved past
  patch, position addition, norm1 and QKV—all sampled values are now exact.
  Full-row native Torch LayerNorm parity is **653,568/653,568 exact** across
  widths 768/1024/1152, affine/non-affine, normal/low-variance inputs. This is
  separately runnable with `tools/oracle/vision_norm_probe.py`, not part of CPU
  unittest discovery. Exact donor: Torch commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- **Vision attention root cause is measured:** actual vLLM uses FA2 noncausal
  D96 specialization for logical D72, 128×64 tiles. On identical full native
  post-RoPE Q/K/V, native replay is exact; W34 differs in **131,115/1,382,400**
  elements (relative L2 **0.00011248**). Reverse-tiled half-probability probes
  reduce but do not eliminate differences and were not installed as a fix.
  `/tmp/w34-attn-investigation/REPORT.md` records the complete experiment.
  The independent target GDN prefill discrepancy is also unresolved.
- **Verification:** `mise run check-quick` passes both Clippy feature states and
  formatting; **65 CPU oracle tests pass**. The locked single-GPU run in
  `/tmp/w34-final25-single-gpu-tests.log` passes native DFlash, constraint/ChatML,
  image policy, completion/logprob, live Axum, full-vocabulary GPU sampler, and
  live non-speculative/MTP/DFlash structured-output tests. This is not a substitute
  for the final unrestricted two-GPU check. Review found no important issues;
  optional authentication/limits, legacy-wire polish, and additional oracle
  hardening suggestions remain outside this pass.

## Current reproducible commands

CPU structured-output/serialization evidence (17 tests passed):

```sh
export PATH="/opt/cuda/bin:$PATH"
cargo test --release --features cuda --test phase6_chatml --test phase6_constraints
# /tmp/w34-phase6-structured-evidence.log
python -m unittest discover -s tools/oracle -p test_serve_bench.py -v
```

New isolated MoE benchmark (run only on a free GPU):

```sh
CUDA_VISIBLE_DEVICES=1 target/release/examples/moe_bench MODEL OUTPUT.json 50
```

Common live SSE benchmark, run identically against each already-started engine:

```sh
python tools/oracle/serve_bench.py --base-url http://localhost:8000 \
  --model MODEL --out OUTPUT.json --concurrency 8 16 --tokens 128
```

New optional ViT diagnostics:

```sh
# Add this flag to the existing native real-image capture command:
python tools/oracle/vllm_vl_dump.py ... --capture-vision-layers
# Then compare bounded first/middle/last rows at matching native/W34 stages:
CUDA_VISIBLE_DEVICES=1 target/release/examples/vision_gate ORACLE_DIR GATE.json \
  --vision-trace TRACE.json
```

Vision traces cover patch projection, every block's inputs/norms/QKV/attention
projection/MLP/output, and merger boundaries. Native decorated patch/block calls
bypass PyTorch hooks, so those boundaries wrap `forward` directly (CPU regression
passes). The complete boundary trace identified reduced-precision cuBLAS
reductions: a realistic-shape regression failed on all 1,382,400 elements before
the fix and passes afterward. Biased GEMMs now scope
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` and restore the old handle mode.
The 27B sampled patch projection is **exact** against native vLLM in
`/tmp/w34-vl-strict-bias/27b-trace.json`; later-stage differences remain. They do not alter the canonical
comparison, inject oracle embeddings, or relax tolerances. Identical-embedding
experiments remain diagnostics only.
