# Standalone performance coverage audit

Audit baseline: `496d1a1` (post-HGPTQ merge). Inventory below describes that
baseline; the follow-up Flash-Next campaign is `campaigns/flash-next-hgptq.toml`.
Scope: the standalone HTTP
`w34perf` suite, not internal `w34 bench`, oracle parity, kernel profiling, or
model-quality evaluation. No GPU benchmarks or server qualification were run
for this audit. Configured coverage is not evidence of successful execution.

## Verdict

**The harness is a useful foundation; the current campaign matrix is not
sufficient for the current models.** Keep the harness and historical evidence.
Add a new, explicitly versioned campaign rather than rewriting old configurations
and making historical results appear to describe new checkpoints.

The main blockers are missing HGPTQ target/drafter profiles, missing Flash-Next
DFlash2 profiles, and an agreed comparison-checkpoint matrix. Workload coverage
also needs stronger representative decode cases and explicit Flash-Next state
protocols. Existing internal performance results do not fill these HTTP cells.

## Inventory: what actually exists

Paths below are relative to `benchmarks/perf/` unless stated otherwise.

| Area | Existing standalone coverage | Gap for the refresh |
|---|---|---|
| Generic harness | `suite.toml`: 4B template; C1/C8/C16, mixed requests, offered load, agent replay, unique/primed prefixes | Template is not a current flagship-model campaign; provenance includes placeholders |
| 2B | No dedicated campaign | HGPTQ plain and MTP, TP1/TP2 |
| 9B | `campaigns/9b-longctx-coding-agent.toml`: plain GPTQ long-context replay | HGPTQ and short/medium-context decode coverage |
| 27B | `27b-completion.toml`, `27b-kv-spec-completion.toml`, `27b-spec-and-tp2.toml` and long-context/capacity campaigns: GPTQ, KV sweeps, MTP, BF16 DFlash2 drafter, multiple engines | HGPTQ target; HGPTQ drafter; updated engine binaries and newly qualified serving configurations |
| 35B-A3B | `35b-a3b-moe-serving.toml`: third-party `llmfan46/...uncensored-heretic...GPTQ-Int4`, MTP-k2 | HGPTQ; plain versus MTP; source-model equivalence cannot be assumed for the modified heretic checkpoint |
| Flash-Next | `flash-next-serving.toml`, `flash-next-longctx.toml`: `albucino/...W4A16-FP8PLE`, warp-34 target/MTP TP1/TP2; specialized vLLM overlays and other practical baselines | HGPTQ/lut4 PLE target; quantized DFlash2 drafter; current residency policy; new serving qualification |
| Multimodal | Workload schema supports only text and agent replay (`w34perf/config.py`, `workloads.py`) | No first-class image performance cases; a gap if claiming coverage of all serving capabilities |

No campaign TOML currently names an HGPTQ checkpoint. The 27B DFlash2
configurations use `incoai/Qwen3.8-27B-DFlash2`, not the new quantized drafter.
Flash-Next's configured speculation is MTP, not DFlash2.

Historical standings and completion ledgers remain valid only for their recorded
binaries, artifacts and workloads. In particular, Flash-Next profiles record
older hot-expert/MTP residency values and versions; those descriptions must not
be copied onto current runs without checking startup allocation. Specialized
`qfn-*` overlays are not stock vLLM support.

## Comparison checkpoint policy

Maintain two distinct comparison tracks:

1. **Matched checkpoint / engine comparison:** use identical plain GPTQ bytes
   on warp-34, vLLM and SGLang where supported. Keep our existing plain GPTQ
   checkpoints for now. Match source model, tokenizer/template, workload,
   hardware and precision controls; label remaining differences. Compare
   plain decoding first, then explicitly labeled speculation/KV sweeps.
2. **Warp-34 quant comparison:** HGPTQ versus the existing plain quant on
   warp-34, holding hardware, workloads and decoding mode constant. This is not
   a second cross-engine campaign. For Flash-Next the existing reference is
   AutoRound W4A16 with FP8 PLE, not plain GPTQ; label that distinction and do not
   infer quality equivalence from bit width.

Third-party replacement selection is **deferred**; it is not needed for this
refresh and no publisher is being declared popular or equivalent. Require the same underlying model/version (not a
fine-tune, uncensored derivative, or similarly named predecessor), pinned
revision, quantization coverage/group sizes, tensor exclusions, tokenizer and
chat template. For Flash-Next explicitly record PLE representation and size;
FP8 PLE versus lut4 PLE materially changes host memory and paging. Record MTP
weight format and any filtered draft vocabulary too. Quality qualification is
an external prerequisite for equivalence claims, not part of this performance
audit.

Use separate target and drafter inventories/revisions. The native BF16 drafter
and HGPTQ drafter are separate experimental configurations, even when the target
is identical. Missing upstream support is an explicit unqualified/unsupported
cell, not a reason to patch an engine silently or substitute another model.

Current inventory starting points are in repository `docs/HGPTQ.md` and
`docs/STATE.md`. Resolve full revisions before preparing the pod: the 27B HGPTQ
document lists two target revisions, and Flash-Next's quantized drafter must get
its own pinned revision and inventory. Do not guess a current revision from a
repository name or use a moving `main` in retained launch evidence.

## Proposed minimum matrix (not yet implemented)

Avoid a full Cartesian product. Use the following tiers, and retain explicit
skip/failure/capacity outcomes rather than dropping unsuccessful profiles.

### Tier 1: current warp-34 capability coverage

Scope is limited to four mainstream, unmodified model families: **9B, 27B,
35B-A3B, and Flash-Next**. Exclude uncensored/heretic derivatives. Do not add 2B
or legacy 4B to this refreshed campaign; their inventory entries above describe
historical coverage only.

All rows use HGPTQ, the same recorded KV policy initially (TQ4), and TP1 and TP2.
Every enabled serving profile must be qualified on the pod before being labeled
supported. MTP-k3 is the initial common comparison, not a claim that k3 is optimal.

| Target | Decode modes | Initial profile count |
|---|---|---:|
| 9B | Plain, MTP-k3 | 4 |
| 27B | Plain, MTP-k3, DFlash2 with native BF16 drafter, DFlash2 with HGPTQ drafter | 8 |
| 35B-A3B | Plain, MTP-k3 | 4 |
| Flash-Next | Plain, MTP-k3, DFlash2 with HGPTQ drafter | 6 |

These **22 candidate profiles across four model families** establish coverage,
not all performance tuning. Historical 2B/4B evidence stays intact but is outside
the refreshed campaign. Probe other MTP depths / DFlash2 block sizes only after
this core is working.

### Tier 2: cross-engine comparisons

Start with the existing 27B plain-GPTQ common checkpoint on all three engines,
TP1/TP2, plain decode and then qualified speculative modes. Extend the common
checkpoint lane to the other model families where available. Qualify Flash-Next
stock-engine support separately from custom overlays. Do not expand this refresh into third-party practical comparisons. Retain the
old FP8-PLE Flash-Next lane as a labeled warp-34 reference, not an equivalent
HGPTQ artifact; historical runs on different hardware are not controlled quant
comparisons.

For KV, first establish a common supported precision where capacity allows,
then compare best-supported policies at a declared memory/context budget. An
OOM at the requested budget is a result, not permission to reduce it silently.

### Tier 3: workload and capacity depth

Run a small representative C1 battery across Tier 1; expand only selected
profiles to the larger battery:

- Short chat, code generation, reasoning, and realistic non-repetitive document
  prompts; include 256- and 1024-token output ceilings. Retain actual output
  lengths and EOS behavior; a ceiling is not a fixed generation length.
- Short, approximately 4K, and approximately 8K input buckets calibrated against
  the pinned tokenizer/template, with actual server usage retained. Existing
  `context_repetitions` measures repeated text, not tokens.
- Explicit primed-prefix pairs and existing recorded coding-agent replay. Keep
  Open-SWE-style 64K/120K/252K fixtures as separate capacity/long-context lanes;
  generated history can diverge across engines, so replay is not identical
  per-turn token work or an autonomous-agent score.
- C1 then C4/C8 where serving permits, with enough sessions to sustain the offered
  concurrency. Measure the real speculative serving policy, including fallback
  or disabling at concurrency. Existing Flash-Next profiles use max-seqs=1;
  offering C8 to those would test queueing, not eight active decoding sequences.
- One bounded offered-load sweep with TTFT/end-to-end SLOs on selected deployments,
  rather than only closed-loop aggregate throughput.
- Add image workloads later if the release claims multimodal performance coverage.
  They are not required to answer the initial text HGPTQ/DFlash2 questions.

## Measurement and harness gaps

Existing `runner.py`, `transport.py` and `metrics.py` already support streaming
HTTP timing, usage accounting, warmups/repetitions, closed/open-loop load,
primed prefixes, replay, failures, immutable bundles and optional GPU/host
telemetry. Use these rather than create another runner.

1. **State protocol (required for state claims).** `unique` prevents exact prefix
   reuse; it does not flush expert LRU, PLE/OS page cache or CUDA initialization.
   Distinguish steady-state, explicitly primed-prefix, and first-request-after-
   fresh-server runs. The latter needs an external restart protocol and zero
   client warmups; it still is not cold OS storage. Record request order and
   host memory/cache conditions. Do not flush a shared host's caches casually.
2. **Drafter provenance (required).** Configs can retain extra metadata, but the
   current validated fields and `compare.py` control keys do not give target and
   drafter independent first-class identities. Add/check these before treating
   differently quantized drafters as the same experiment.
3. **Comparison semantics (required before automated cross-quant reporting).**
   `compare.py` conservatively marks changed artifact/model revision, suite
   source, cache configuration, etc. incompatible. This protects matched runs,
   but is not a practical cross-quant comparison mode. Preserve those warnings;
   explicitly label deployment comparisons rather than weakening equality
   globally or fabricating identical metadata.
4. **Speculative and residency diagnostics (useful, not a speedup prerequisite).**
   Acceptance/tokens-per-round and expert-cache counters are not collected as
   standardized metrics. Retain sanitized server evidence where available and
   mark absent counters unknown. HTTP performance comparisons remain valid
   without them; attribution to a specific internal mechanism does not.
5. **Timing interpretation (existing limitation, not a new bug).** Output chunk
   gaps are not token ITL, particularly with speculative bursts. Usage/span TPOT
   is an estimate; aggregate completion tokens/s includes prefill and HTTP
   overhead, not pure decode-kernel throughput. Keep these labels.
6. **Repetition independence.** Release presets repeat within one server lifetime.
   For publishable comparisons, use fresh-server repetitions and interleave
   engine order. Keep individual trials and variability, not just a best run.

## Pod handoff and execution order

No pod was provisioned and no downloads, server launches, or GPU runs performed
by this audit. Once the user supplies the pod:

1. Record exact GPUs/VRAM, UUIDs, driver/CUDA, PCIe/NVLink topology, CPU ISA/core
   topology, RAM, available disk and locked-memory limits. Flash-Next offload
   performance is strongly host-dependent; do not attribute differences between
   local AVX-512 and a pod's AVX2 CPU solely to GPU generation.
2. Budget disk/RAM for selected snapshots before downloading: Flash-Next HGPTQ
   alone is documented at about 94 GiB, and comparison snapshots add substantial
   storage. Do not assume that fitting weights on disk establishes RAM capacity.
3. Pin binaries/containers, target/drafter artifacts and templates; create new
   pod-specific launch profiles with actual paths and measured residency.
4. Run inventory/preflight and a smoke request for each selected profile,
   preserving failures and startup logs. Validate the actual supported context,
   concurrent sequence budget, reported usage and speculation mode.
5. Execute C1 Tier 1, then matched-checkpoint baselines, then concurrency/cache/
   capacity expansion. Run one deployment at a time on the selected GPUs; keep
   local GPUs free for development.
6. Retain raw bundles, telemetry and sanitized evidence; attach existing external
   correctness/quality qualification as appropriate without replacing this
   performance audit with a new quality-evaluation project.

## Audit verification

`mise run perf-check` passed: 89 CPU-only tests and template validation
(10 deployments, 13 cases). Campaign tests validate/plan the checked-in campaigns
offline. This establishes harness/configuration checks, not serving support or
performance. Documentation review reported no actionable findings. The full
CUDA project check was not run for this documentation-only audit.

## Next actions

- **P0:** Agree and encode the new profile/checkpoint matrix; pin both drafters;
  refresh launch metadata without overwriting historical campaigns.
- **P0:** Define representative decode cases and state protocols; qualify serving
  profiles on the pod. No new performance claims before this step.
- **P1:** Add drafter comparison controls and explicitly labeled within-warp-34
  quant reporting. Third-party checkpoint replacement is deferred.
- **P1:** Expand concurrency, long-context and offered-load coverage; capture
  optional speculative/residency diagnostics.
- **P2:** Multimodal cases and wider tuning sweeps.
