# Phase 6 exit-gate evidence ledger

Phase 6 is **not complete**. Implemented code, passing regression tests, and passing
release gates are different claims. Historical heuristic profile windows are not
aligned donor evidence. `/tmp` paths below identify local measured artifacts, not
portable checked-in fixtures.

## Closeout verification and commit boundary

The final closeout command was:

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
| 6.1 MTP/DFlash2 per-rank graphs | Implemented in `proposer.rs`; coordinated single-thread TP=2 launch in `model.rs`. Unrestricted `mise run check` passed **132 Rust + 65 oracle CPU tests** in **211.31 s** (`/tmp/w34-final25-full-check.log`). Fresh TP2 traces show both ranks' proposer graph launches on the same host thread: 32/rank for MoE MTP, 38/rank for 27B DFlash. | Implemented and verified; performance below target |
| 6.1 DFlash2 roofline | Fresh `/tmp/w34-qk8-{before,after}` traces: eight-key BF16 QK improves `spec_decode` **5.2754 → 4.1218 ms/round** and decode **956.79 → 1,052.82 tok/s**, with identical acceptance counts over 10 rounds. All 10 native DFlash regression tests pass. Still does not meet `<2 ms/round` or 1,100 tok/s. | Open |
| 6.1 MoE MTP throughput | Fresh `/tmp/w34-phase6-tp2-moe-mtp/` captures **339.02 tok/s**, 32 rounds, proposer **2.1539/2.1625 ms/round per GPU**. Required 500+ tok/s and aligned donor win remain unmet. The paired 27B DFlash capture measures **149.66 tok/s**, 38 rounds, **4.6650/4.6918 ms/round per GPU**. | Open |
| 6.2 Attention | GQA-grouped MMA and split-KV implementations exist in `op_paged_attn.cu`; full suite includes the fragmented-layout/causal/long-context gates. Fresh `/tmp/attn-shortgraph-fixed/` measures **0.4825 ms/round** versus the earlier aligned **0.9184**, after increasing short-query occupancy and enabling 64-key partitions for the 1024-token graph bound. All 128 output IDs match; last acceptance counts changed `[13,3] → [15,1]`, so this is not identical-round-shape timing. Attention parity test passes. Native SGLang is still **0.2098 ms/round** with FP16 KV. | Correctness tested; performance gate open |
| 6.3 MoE | `/tmp/w34-phase6-moe-real-weights.json`: all nine shapes pass existing absolute/relative tolerances using real 35B-A3B layer-0 weights/router. M=32/64/128 grouped-MMA speedups **1.48×/2.72×/4.81×**; M=1..16 retains GEMV. Shared experts omitted identically; not end-to-end throughput or a new TP=2 measurement. | Measured partial coverage |
| 6.4 Grammar/rollback | Current command below: 12/12 `phase6_constraints` tests pass, including strict nested schemas, JSON, regex, recursive CFG, structural tags, reasoning transition, EOS rollback, and XML/Hermes tool modes/injection rejection. | Regression coverage passes |
| 6.4 ChatML | Current command below: 5/5 `phase6_chatml` tests pass, including byte parity against native Jinja, system+tools, assistant tool history and consecutive tool responses. | Regression coverage passes |
| 6.4 Live tool/schema adherence | Last full check passed the live structured-output tests. This is finite test-matrix evidence, not an assertion that every possible schema/input was exercised. | Tested matrix passes in unrestricted full check |
| 6.5 Worker/SSE/controls | Axum + dedicated scheduled engine worker is committed (`5bb3742`). Later terminal logprob/UTF-8 offset fixes passed 4 live serving + 6 completion-logprob tests and the unrestricted full check. | Regression coverage passes |
| 6.5 Eight/sixteen agents | `/tmp/w34-phase6-serve-load.json`: all **8/8 and 16/16** requests stream before finish and return exactly 128 selected-token records matching usage. Aggregate **669.00/952.83 tok/s**, mean TTFT **446.31/577.62 ms**, mean client ITL **7.479/8.149 ms**. SGLang target-only `/tmp/w34-phase6-sglang-serve-load.json` passes identical workload hashes: **178.96/2,191.16 tok/s**, TTFT **3,000.58/60.87 ms**, ITL **21.417/6.870 ms**. Runs were ordered 8 then 16; native first-run compilation makes the 8-client advantage unsuitable as a steady-state claim. W34 had DFlash enabled but batches use scheduled target decode. | Both measured; 16-client performance gate fails |
| 6.5 Graph buckets | B=1,2,4,8,16 scheduler/graph implementation and graph metadata mutation regression coverage exist; live 8/16-client timing is a separate obligation. | Regression coverage; performance open |
| 6.6 GPU sampler | Device sampling/grammar masks are committed (`f1a0be7`), with deterministic/stochastic sampler and scheduled/speculative coverage in the full check. | Regression coverage passes |
| 6.6 Remote images | Opt-in bounded HTTP(S) loading committed (`f4197dd`); CLI 64 MiB and HTTP 16 MiB caps preserved by subsequent JPEG decoder changes. | Regression coverage passes |
| 6.6 PNG/JPEG 4B | Latest `/tmp/w34-vl-welford/model0-{png,jpg}.json`: final-norm relative L2 **`.024956/.059718`** against unchanged `.05`; exact 8-token greedy matches. JPEG regressed despite locally native-exact position/norm corrections; do not count the earlier JPEG pass as current. | PNG pass; JPEG fail |
| 6.6 PNG/JPEG 27B | Latest `/tmp/w34-vl-welford/model1-{png,jpg}.json`: final-norm relative L2 **`.146638/.082821`**; greedy 8/8 each does not excuse tensor failure. | Fail |
| 6.6 PNG/JPEG 35B-A3B | Latest `/tmp/w34-vl-welford/model2-{png,jpg}.json`: final-norm relative L2 **`.190177/.220284`**; greedy 8/8 each does not excuse tensor failure. | Fail |
| 6.6 Eight needle cases | Corrected-binary matrix `/tmp/w34-phase6-needle-native-fp16/summary.json` completed: **all eight passed**, both TQ4 and `tq4_all` at **32k, 64k, 128k, 262k**, on one GPU. One fixed secret at midpoint per case; not statistical retrieval coverage. Historical interrupted runs are not counted. | Pass |
| Aligned profile harness | `profile_diff.py --nvtx-correlated` accounts for launch threads, processes, graph nodes, and asynchronous completion. Fresh `/tmp/aligned-oracle-profile` and `/tmp/aligned-w34-profile` captures have **identical 128 prompt IDs and identical 128 generated IDs**. W34/SGLang draft GPU cost **4.1074/3.4023 ms/round**, target attention **0.9184/0.2098**: both category gates still fail. Native uses FP16 KV versus W34 TQ4. Native denominator is 11 measured worker calls (10 scheduler rounds, extra overlapped work retained); W34 has 10 rounds. W34 decode 1,059.48 tok/s; donor streamed decode 122.186 ms. Parser `gate_eligible=false` remains; this is not proof of complete protocol equivalence or a performance win. | Actual correlated comparison; gates fail |
| Final verification | Unrestricted `CUDA_VISIBLE_DEVICES=0,1 mise run check` exited **0**, **132 Rust + 65 oracle CPU tests**, **211.31 s** (`/tmp/w34-final25-full-check.log`), including handoff and capped short-attention dispatch. Follow-up review had no important findings. Subsequent harness/kernel changes require re-verification; passing the suite does not close failing vision/performance gates. | Full checkpoint passes; release incomplete |

## Latest verified measurements

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
