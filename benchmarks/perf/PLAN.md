# Performance-suite plan and task ledger

## Objective and decisions

Produce regularly refreshable, shareable comparisons against **vLLM, SGLang,
llama.cpp, and Don-Chad/ninfer-3090**, independent of development oracles.
Primary hardware: one and two RTX 3090 **Ti** 24 GB cards; no NVLink/P2P.
**4-bit weights are the common ground**, with best practical equivalents for
GGUF/`.ninfer` and identical GPTQ wherever possible. KV precision is a separate
axis. No oracle dependencies, tensor parity, donor patches, or `nsys` requirements.

A standalone suite is a product with its own fixtures, configuration, evidence,
metrics, release rules, tests, and versioning. Running it must not change engines,
kill someone else's server, install dependencies, or download models implicitly.

## Publication methodology

Two explicitly separated comparisons:

1. **4-bit deployment equivalents (primary):** one common source-model revision,
   approximately matched quantization/quality and full artifact provenance.
   Baseline KV + speculation off, then independent KV/speculation sweeps.
2. **Same-artifact subset (controlled):** engines that accept the identical GPTQ
   weights/tokenizer/template. This isolates more implementation differences;
   it is not a universal five-engine comparison.

Don't issue one composite winner score. Report latency, throughput, goodput,
capacity, quality, and resource tradeoffs separately. Tuning runs are not
publication runs. Pin settings before collecting results. Randomize/interleave
engine order over independently restarted campaigns to reduce thermal/time bias.
Retain all failed, unsupported, and untested cells rather than dropping them.

### Workload/hardware matrix to freeze

| Axis | Campaign targets |
|---|---|
| Source models | Mutually supported 27B dense first; then 4B, 9B, 35B-A3B MoE where supported |
| Weights | GPTQ INT4 g128 shared subset; nearest independently qualified GGUF Q4 / ninfer 4-bit variants |
| GPUs | 1 card; 2 cards using documented TP/layer/pipeline strategy; separate 2× single-card replicas later |
| KV | Native 16-bit baseline (label FP16/BF16); qualified 8-bit, 4-bit, mixed policies; no FP8 support assumptions on sm_86 |
| Speculation | Off; native MTP; DFlash2 where supported, with pinned proposer and block settings |
| Input | Exact calibrated short/512/2K/8K/32K/64K/128K/262K buckets, constrained by real model/context support |
| Output | Natural EOS at 32/128/512/1024 budgets; separate fixed-length generation lane if supported |
| Load | Closed-loop C1/2/4/8/16/32; fixed/Poisson rate sweeps through saturation; mixed short/long jobs |
| Cache | Unique-leading prompts; attempted primed reuse; natural multi-turn; later explicit eviction/capacity pressure |
| Agents | Coding/debugging, repository review, research/RAG, support; 1/8/16 sessions; tool delays, branching and growing histories |
| Other common serving | Streaming/nonstreaming; constrained JSON/tool schemas; reasoning on/off; image+text; startup/restart; cancellation and endurance |

The initial template is a **small runnable starting matrix**, not a completed
Cartesian product. Do not run every possible combination blindly: use a staged
funnel (smoke → representative short matrix → long/capacity → publication trials).
Budget maximum duration and disk space before long-context/endurance campaigns.

## Phase A — Independent runner and reproducible evidence (implemented)

Acceptance: runnable using only Python + HTTP endpoint, no oracle imports; raw
wire requests/events retained; no chunk/token confusion; deterministic unit and
HTTP tests; configured one/two-card profiles; clear failure/partial-run behavior.

- [x] Create `benchmarks/perf/` as a copyable stdlib-only suite.
- [x] Validate TOML deployments/cases and fixture hashes; explicit engine selection.
- [x] Add all five engine profiles for one/two cards; mark ninfer pipeline unqualified.
- [x] Record 4-bit artifact/source/tokenizer/template/quality metadata requirements.
- [x] Implement OpenAI SSE transport, usage accounting, finish/error validation,
      hard total deadlines (including trickle streams), bounded response size,
      credentials via environment-variable name, and no retries.
- [x] Closed-loop sessions and bounded open-loop fixed/Poisson arrivals;
      retain client overload and schedule lag rather than introducing hidden queues.
- [x] Retain flushed request/session JSONL, manifests, per-trial JSON, artifact hashes,
      and derived Markdown/CSV reports with publication blockers and small-sample warnings.
- [x] Separate compile warmup, explicit priming, and measurement namespaces/phases.
- [x] Include short/decode-heavy/prefill-heavy/long/mixed/concurrent/prefix/load cases.
- [x] Include four-turn coding-agent tool-result replay with actual generated histories,
      no arbitrary tool execution, explicit `tool_choice=none`, and simulated tool delay.
- [x] Add CPU/local-HTTP tests and `mise run perf-check` to project checks.
- [x] Complete initial verification/review and record exact outcomes below.

## Phase B — Artifact and engine qualification (next, publication blocking)

- [ ] Select the shared **27B source revision** after mapping available `.ninfer`
      and GGUF artifacts. Do not assume similarly named Qwen 3.5/3.6/3.8 are identical.
- [ ] Acquire/convert closest 4-bit GPTQ/GGUF/ninfer artifacts; retain license,
      conversion/calibration commands and revisions, file sizes and shard SHA-256s.
- [ ] Add an artifact inventory/preflight command that verifies local weight,
      tokenizer and template hashes and matches advertised `/v1/models` IDs.
- [ ] Freeze separate release pins/container digests and verified launch recipes
      for each engine on this Linux machine; save `--help` and startup logs.
- [ ] Run each engine's real 1-card smoke, then 2-card smoke with documented strategy.
      Verify no CPU weight offload/fallback unless explicitly in that lane.
- [ ] Probe ninfer Linux pipeline; retain unsupported/failing result if unqualified.
      No claim of TP=2 or multi-GPU DFlash2 without direct evidence.
- [ ] Add a machine-readable campaign coverage ledger with required/passed/failed/
      unsupported/not-tested cells, exact launch failures and reasons.
- [ ] Add opt-in managed server launch/readiness/process-group cleanup and GPU
      reservation. Keep externally managed endpoints as the universal extension path.

## Phase C — KV capacity, quality, and advanced workloads

- [ ] Materialize baseline + 8-bit + 4-bit/mixed KV profiles for both card counts;
      record real effective bytes/token, cache/GDN precision, memory allocation,
      pool/admission settings. Explicitly retain unsupported backend combinations.
- [ ] Implement tokenizer-pinned calibrated input buckets, including rendered chat
      templates/BOS/EOS. Current context repetitions are **not** token-length buckets.
- [ ] Add independent public functional quality/retrieval fixtures (several depths,
      lengths, secrets, code/JSON tasks). Gate each quant/KV/spec deployment without
      importing development oracle fixtures; include baseline-quality deltas.
- [ ] Add sampled per-GPU VRAM/utilization/clocks/temperature/power and host RSS,
      retaining raw timestamps and labeling sampling intervals. Energy is optional
      and must integrate actual power samples, never nameplate TDP.
- [ ] Add bounded context/concurrency capacity search with timeouts and retained
      OOM/startup failures, then equal-memory-budget comparisons. Report successes
      at quality-qualified lengths, not configured max-model-len alone.
- [ ] Add read-only RAG/research and support-agent replay fixtures, realistic tool
      latency distributions, branching, parallel tool result bundles, and trace licenses.
- [ ] Add optional **live** tool-choice/JSON-schema agent loop with deterministic
      sandbox tool simulator and explicit task-success checks. Replay is not a
      functional tool-calling or autonomous coding benchmark.
- [ ] Add reasoning on/off, structured output, vision, nonstreaming, disconnect,
      timeout recovery, and soak/endurance lanes; label supported feature subsets.
- [ ] Add separately measured startup/readiness/model load and warm first-request costs.
- [ ] Add optional pooled connections and token-bearing-event instrumentation
      (with overhead measurement), never deriving token ITL from SSE chunk count.

## Phase D — Regular publication campaigns

- [ ] Freeze daily quick / weekly full / release publication presets. Suggested
      quick: smoke + C1/C8 + agent replay; full: context/KV/rate/agent sweep;
      release: all qualified common cells plus quality/capacity/feature supplements.
- [ ] Add randomized/interleaved managed campaign order, independent restarts,
      explicit cold-cache controls, steady-state windows, cooldown policy and
      minimum-duration/load-generator-capacity checks.
- [ ] Collect at least 3–5 repetitions and enough samples for meaningful tail
      claims; show variability and actual EOS/output lengths. Add bootstrap
      intervals over independent trials where justified, not over correlated tokens.
- [ ] Add cross-run comparison guarded by source/fixture/decoding/hardware/cohort
      equivalence, with explicit waivers for deployment-level quant/template differences.
- [ ] Generate static release tables/charts and artifact bundle; include all failure
      and untested rows, quality, model bytes, peak VRAM, SLO goodput and topology.
- [ ] Set a scheduled manual/CI GPU runner after GPU reservation and data retention
      are in place. Don't commit bulky raw generations or paid downloads to Git.
- [ ] Perform a human release audit and sign/archive the immutable bundle.

## Initial verification evidence

This section records implementation validation, **not competitive results**.
Results use the local cached 4B GPTQ model; GGUF/ninfer weight equivalence and
competitor recipes remain unqualified. Exact commands/config are retained in
run manifests.

- `mise run perf-check`: **22 tests pass**, including real local SSE transport,
  truncated/error/usage-less streams, hard trickle deadlines, credentials,
  malformed JSON, replay histories, priming separation, deterministic overload,
  interrupted/failed reporting, and artifact generation. Manifest validation
  passes for **10 profiles / 13 cases**.
- Real **TP=1 / FP16 KV** smoke passes short, concurrent, primed prefix,
  Poisson-arrival and coding-agent replay requests:
  `/tmp/w34-perf-tp1-smoke-fixed/` (one repetition, two sessions/case,
  32-token ceilings, reduced context). All **16 measured requests** succeed
  with valid usage, including eight turns across two agent sessions.
- Real **TP=2 / TQ4 KV** smoke passes short, reduced mixed-traffic, and coding-agent
  replay cases: `/tmp/w34-perf-tp2-smoke/`, all **12 measured requests** succeed
  with valid usage. Both temporary servers were stopped after validation.
- The initial TP1 replay attempted new tool calls despite prose instructions
  (`/tmp/w34-perf-tp1-smoke/`). A regression test reproduced missing
  `tool_choice=none`; the explicit control fixed the replay. The original failed
  run is retained and not treated as a pass.
- Initial review found three important issues: unvalidated RNG seed, malformed
  upstream values breaking journaling, and malformed bearer credentials leaking
  through exception text. Failing regressions were observed, then all three
  were fixed. The follow-up review found **no Important/Critical issues**.
- Also fixed unique request identifiers, overload-session evidence, incomplete-run
  reporting, invalid URL ports, and warmup/priming failure exit classification.
- Final `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **212.22 s**:
  **208 Rust tests/doc-tests + 69 oracle tests + 22 independent suite tests**,
  formatting and both Clippy feature states. Log: `/tmp/w34-perf-final-check.log`.
  `git diff --check` passes.

Review suggestions intentionally deferred: additional malformed **operator-owned**
config/fixture schema validation and config JSON hardening, extremely deep SSE
JSON nesting limits, graceful SIGTERM/heartbeat, and spreadsheet-formula escaping
for trusted metadata. These are not publication or engine-speed claims. External
fixture paths are intentional (portable public trace corpora), not a sandbox
boundary. Treat configs and fixtures as trusted and don't open untrusted CSV
metadata in spreadsheet applications. SIGKILL/SIGTERM may leave `status=running`;
use raw flushed evidence and don't interpret that as a completed run. A carried-
over Docker health/liveness suggestion is unrelated to this suite and remains
explicitly documented as HTTP responsiveness in the repository README.

Next concrete task: **Phase B source-model/4-bit artifact mapping**, followed by
real competitor 1-/2-card qualification. No comparative leaderboard is claimed.
