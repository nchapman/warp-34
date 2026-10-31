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

- [x] Map 27B candidate artifacts with immutable HF revisions/sizes/hashes and
      distinguish local byte verification from remote-advertised hashes
      (`qualification/27b-candidates.json`, `engines.md`).
- [ ] Freeze the shared **27B source revision** after completing provenance for `.ninfer`,
      GPTQ and GGUF artifacts. Do not assume similarly named Qwen 3.5/3.6/3.8 are identical.
- [ ] Acquire/convert closest 4-bit GPTQ/GGUF/ninfer artifacts; retain license,
      conversion/calibration commands and revisions, file sizes and shard SHA-256s.
- [x] Add offline streaming `inventory`/`verify` for local weight/tokenizer/template
      bytes, symlinked HF snapshots, file/glob scope, relocation and drift detection.
- [x] Add online preflight matching configured IDs against `/v1/models`, checking
      local inventory/profile hashes and indexed-shard completeness; retain new
      passed/failed/skipped JSON without attesting remote loaded bytes.
- [ ] Automatically attach trusted inventory/preflight/startup evidence to run
      bundles. For now retain the separate files and ledger hashes together.
- [ ] Freeze separate release pins/container digests and verified launch recipes
      for each engine on this Linux machine; save `--help` and startup logs.
- [ ] Run each engine's real 1-card smoke, then 2-card smoke with documented strategy.
      Verify no CPU weight offload/fallback unless explicitly in that lane.
- [ ] Probe ninfer Linux pipeline; retain unsupported/failing result if unqualified.
      No claim of TP=2 or multi-GPU DFlash2 without direct evidence.
- [x] Add explicit support/enable states, mandatory skip reasons, shared `plan`/`run`
      GPU filters/exclusions and machine-readable measured/failed/skipped/not-run
      deployment × selected-case coverage. All-skipped runs retain evidence without
      network/GPU probes. No precision/parallelism fallback.
- [x] Manually retain observed pre-HTTP launch failures and probe outcomes in
      `qualification/27b-one-card.json`, including explicit BF16 vs FP16 variants.
- [ ] Automatically record managed launch failures/OOM/quality failures; the HTTP
      runner alone cannot observe externally managed pre-HTTP startup errors.
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

## Optional dual-card coverage and qualification follow-up

- Implemented `enabled`, `support = supported|unverified|unsupported`, mandatory
  `skip_reason` for disabled/unsupported entries, minimal non-runnable placeholders,
  `--gpus 1|2` and repeatable `--exclude-engine` on both `plan`/`run`. Ninfer Linux
  pipeline is disabled/unverified by default, not mislabeled unsupported.
- Coverage includes all deployments for the selected cases, selection/availability
  reasons and completed/required repetitions. All-skipped manifests use `skipped`;
  CSV `row_type=coverage` entries have blank measurements, never zero tok/s.
- Inventoried and re-verified **19,438,009,793 bytes / 11 visible files** in cached
  GPTQ snapshot `c11a884...`; stable inventory hash is recorded in the candidate
  evidence. No model downloads. Source candidate `Qwen/Qwen3.8-27B@1d4bf0f2...`
  is documented by the 3090-specific NInfer artifact. GPTQ/GGUF exact source
  provenance remains unresolved; **no frozen cross-format equivalence claim**.
- Real vLLM **0.31.0**, **27B / TP1 / FP16 KV / eager / text-only** smoke succeeds:
  two sequential requests, two at concurrency 2, four coding-agent replay turns;
  **8/8 measured requests** with valid usage. `/v1/models` ID matches. Offline
  cached snapshot, max context 2048, max sequences 2, GPU utilization 0.90,
  explicit CPU offload 0, no speculation; startup logs show Marlin + Triton GDN.
  `--enforce-eager` disables compilation/graphs: this is a startup/API smoke,
  **not an optimized speed comparison**. Eager startup loaded 16.81 GiB model
  memory per vLLM log (not a sampled peak). Reduced `batch-8` case used C2, not C8.
  Server stopped and GPU processes cleared before final project checks.
  Evidence: `/tmp/w34-perf-vllm27-eager-smoke/`,
  `/tmp/w34-perf-vllm27-{startup.log,models.json}`,
  `/tmp/w34-perf-vllm-serve-help-all.txt`,
  `/tmp/w34-perf-27b-gptq-inventory.json`.
- New regression coverage includes all-skipped no-network behavior, omitted
  endpoint fields, credential rejection even in disabled metadata, filter/CLI
  parity, failure/not-run distinctions, inventory drift/relocation/symlink names,
  and complete deletion of included files. Initial tests exposed empty-exclusion
  handling and symlink logical-name bugs; tests were observed failing before fixes.
- Review returned **three nonblocking suggestions, no Important/Critical findings**.
  Fixed complete-deletion verification misclassification with a reproduced failing
  regression. Clarified that inventory hashes are **not signatures/authenticity**
  and inventory root/include are trusted operator inputs. Declined mandatory
  signatures/out-of-band trust-anchor flags and mandatory `--root` as unnecessary
  for this offline trusted-operator tool; explicitly require independent trust or
  external signing for publication. No follow-up review needed for these small
  fixes/clarifications.
- Final post-review `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **209.38 s**:
  **208 Rust tests/doc-tests + 69 oracle tests + 36 standalone performance tests**,
  formatting and both Clippy feature states. Log:
  `/tmp/w34-perf-selection-final-check.log`. Candidate JSON fields and suite local
  doc links validated; `git diff --check` passes. Real CLI single-card planning
  selects five profiles, and `/tmp/w34-perf-disabled-dual-smoke/` retains a skipped
  ninfer two-card run with an empty journal and rendered coverage report.

## Repeated one-card 27B qualification follow-up

- Added `preflight` with trusted local inventory/profile hash verification,
  safetensors index/shard checks, bounded single-attempt authenticated `/models`
  discovery, configured-ID matching and exclusive passed/failed/skipped evidence.
  This is not server loaded-byte attestation or automatic run-bundle attachment.
- Explicitly selected newer cached **`0df81ead...` for all three runnable probes**:
  13 files / **19,610,855,960 bytes**, full checked-in inventory and independent
  verification. Original `c11a884...` evidence remains intact and distinct.
- Retained SGLang v0.5.19 startup failures: c11 quantized width 96 rejected by
  Marlin; 0df text-only flag unsupported for Qwen3.5; 0df FP16 GDN convolution
  BF16/FP16 Triton branch mismatch. Worker failure can yield Docker exit 0;
  never infer serving readiness from that code alone. No engine patches.
- Explicit **BF16 SGLang**, graph-enabled **FP16 vLLM 0.31.0**, and **FP16 warp-34**
  each pass preflight and **42/42 measured requests / nine trials**: three
  repetitions of short C1, offered C8 and four-turn C1 agent replay. Preset and
  machine-readable outcomes/hashes are in `qualification/27b-one-card.{toml,json}`.
  The recorded executed config is in each raw manifest; a later metadata-only
  correction pins the disabled c11 placeholder separately from 0df defaults.
- SGLang is active-limit 2, BF16 KV and loads vision; vLLM requests active 8,
  text-only, FP16 KV and confirmed torch.compile + piecewise/full graph captures.
  Warp-34 uses its built-in renderer, not artifact Jinja. These remain **different
  operational deployments**, not token/dtype/admission/memory-aligned rankings.
- All temporary servers stopped; no downloads, engine patches, quality pass,
  autonomous-agent success, long-context, saturation or two-card claim.
- Review returned **two nonblocking suggestions, no Important/Critical findings**.
  Clarified that missing/invalid inventory checks produce retained failed evidence
  (exit 1), while config/selection/evidence-file setup errors exit 2. Reproduced
  empty output on interrupted inventory read, then fixed Ctrl-C to retain
  `status=interrupted` before propagating exit 130. Unexpected termination/deeply
  malformed trusted operator JSON remain outside graceful-interrupt guarantees.
  Declined the proposed timer-success override: the hard discovery deadline
  intentionally includes reading **and validating** the JSON; expiry before
  validation completes fails closed, consistent with SSE transport. A response
  at the exact deadline is not promised a pass. No follow-up review needed for
  this small error-path fix and documentation clarifications.
- Final post-review `CUDA_VISIBLE_DEVICES=0,1 mise run check` passes in **214.83 s**:
  **208 Rust tests/doc-tests + 69 oracle tests + 47 standalone suite tests**,
  formatting and both Clippy feature states. Log:
  `/tmp/w34-perf-one-card-final-check.log`. Final `mise run perf-check` also passes
  after metadata corrections. Validated the five-profile/three-case preset, all
  JSON and doc links, every ledger/run-bundle checksum and executed config hash;
  rendered Markdown/CSV for all three runs. `git diff --check` passes and GPUs
  are free. Raw evidence remains in `/tmp`; archive before any public release.

Next concrete tasks: finish source/converter provenance and independent quality;
acquire selected GGUF/ninfer artifacts with verified checksums; qualify llama.cpp
and NInfer Linux one-card runtimes. Resolve or deliberately retain SGLang's FP16
startup failures, then align rendering/dtype/admission for controlled comparisons.
Probe two-card routes only where useful/supported; retain explicit skips otherwise.
No comparative leaderboard is claimed.
