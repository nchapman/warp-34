# 27B and Flash Next benchmark completion

## Objective

Finish the independent HTTP comparison suite for Qwen3.8-27B first, then
Qwen3.8-Flash-Next. Reuse `w34perf`; do not couple publication runs to development
oracles, tensor parity tests, or profiler tooling. This ledger supplements
[PLAN.md](PLAN.md), whose broader feature backlog is not all on this critical path.

Primary machine: 2× RTX 3090 Ti 24 GiB, 124 GiB host RAM. Record actual topology,
software, clocks and available memory for each campaign rather than assuming
historical settings still apply. Base implementation at planning: `66974d8`.

## Execution ledger

- [x] Reconcile historical manifests/summaries into deployment × workload coverage.
      `qualification/27b-historical-coverage.json` records 31 local historical
      bundles, per-cell counts/repetitions/blockers and hashes of input files.
      Reproduce the full read-only inventory with `python3 -m w34perf audit results`.
      Counts are recorded summary counts, not independent verification of raw requests.
- [x] Retain provenance, preflight, startup and quality evidence with run bundles.
      Missing historical/startup receipts remain explicit; attachments are not
      attestations that a remote server loaded the declared bytes.
- [x] Qualify missing 27B baseline references: llama.cpp and ninfer-3090, then
      complete the selected same-artifact vLLM/SGLang comparison points.
- [x] Complete 27B supported KV/speculation comparisons and separate TP2 from
      two independent TP1 replicas (the latter remains explicitly historical).
- [x] Add calibrated input lengths, independent functional/retrieval quality,
      resource sampling and guarded cross-run reports needed by these campaigns.
- [x] Freeze the completed 27B qualification comparison with explicit failed cells:
      four baseline runs, 20 advanced profiles and two capacity profiles;
      `qualification/27b-*-measurements.{json,md}`. Reports do not certify
      publication equivalence: cross-format source lineage, independent restarts,
      thermal/host isolation and failed quality gates remain explicit blockers.
- [x] Qualify Flash Next reference projects, immutable artifacts and launch recipes.
      Both DominikBucko overlays and Strata have retained runtime outcomes.
- [x] Add and run Flash Next short-serving, coding/agent and long-context campaigns.
      13 startup outcomes, nine run bundles, 375/381 measured HTTP requests
      successful. Startup, warmup, quality and replay failures remain distinct.
- [x] Publish quick/full/release presets and reproducible result indexes.
      See `campaigns/standings.md`, `campaigns/completion-results.md`, compact
      qualification reports, detailed latency reports and comparison hash receipts.
- [x] Run `mise run check`, review changes, update `docs/STATE.md`, and commit
      verified task-related work. Full check passed both before the infrastructure
      milestone and after final campaigns (357 Rust tests passed, zero failed,
      six ignored; 89 perf tests). Final evidence audit reproduced reports and
      verified hashes/counts. See `qualification/completion-checks.json` and
      `qualification/completion-review.md`.

## Existing evidence: do not confuse missing indexing with missing runs

The following local bundles exist; their historical success is not a new-build
qualification or proof that publication blockers are resolved:

- `results/27b-spec-and-tp2/`: TP1 FP16, TP2 FP16, TP2 MTP-2, TP2 DFlash2,
  and two TP1 replicas each have eight trials, two repetitions and 56 requests
  with zero recorded request failures. Cases: decode C1, prefill ~3K C1,
  speculation C2 and throughput C8. Competitor points remain to be reconciled.
- `results/27b-tq4-longctx/`: TP1 TQ4 and TQ4-all each have six trials,
  two repetitions and 68 requests with zero recorded request failures.
  Cases: long-context ~8K C1 and throughput C8/C16.
- `campaigns/27b-fp16-first.md`: historical three-engine TP1 comparison.
- `campaigns/longctx-coding-agent.md`: historical long-context comparisons,
  including TP2 27B and TP1 near-maximum-context retention.

The speculation summaries explicitly retain unverified artifact/tokenizer/template/
source/quantization metadata, no independent quality evidence, and fewer than
three repetitions. Keep those blockers until supported by new evidence.

## Comparison contract

1. Keep identical-GPTQ comparisons separate from practical GGUF/ninfer 4-bit
   deployment equivalents. Freeze source/conversion provenance and quality before
   claiming equivalence across formats or similarly named model revisions.
2. Baseline: native 16-bit KV, speculation off. Sweep KV and speculation as
   independent axes. Never silently change precision, CPU offload, or parallelism.
3. For every selected cell retain measured, failed, unsupported, or not-run state
   and evidence/reason. Unverified is not unsupported. Missing measurements are
   not zeros. Startup/OOM failures must not disappear from the report.
4. Historical runs remain historical; fresh current-build runs get new paths.
   Tuning is distinct from publication. At least three independent publication
   trials, with more samples required for meaningful tail claims.
5. Report client TTFT, output-span TPOT estimates, aggregate output throughput,
   agent-session latency, quality-qualified capacity and resource usage separately.
   SSE chunk counts are not token counts; no composite winner score.
6. Stage smoke → representative short matrix → long/capacity → publication.
   Do not blindly run a Cartesian product. Bound durations and storage first.
7. No implicit dependency installation, model downloading or server termination
   inside the portable HTTP runner. Externally managed endpoints remain supported.

## Flash Next qualification

Qualify `DominikBucko/qwen38-flash-next-3090`,
`DominikBucko/qwen38-flash-next-2x3090`, and `Niko1221/Strata`, identifying shared
runtime/overlay lineage rather than counting wrappers as independent engines.
Check actual upstream vLLM/SGLang support and capacity on this hardware. Pin
revisions and retain real startup/HTTP evidence before marking support.

Record hot-expert budget/policy, pinned host store, PLE mmap/page-cache state,
CPU threads, storage, PCIe topology, KV precision and MTP artifacts. Offload is
integral to this model and must be labeled, not treated as an accidental fallback.
Published reference numbers on other hardware are contextual, not local results.

Campaign groups:

- Short serving: TP1/TP2, target-only/MTP, C1/C2/C8; include reference-compatible
  128:256 and 128:512 shapes after tokenizer/template calibration.
- Coding/agents: 4K/8K inputs, longer outputs, multi-turn replay, prefix reuse;
  distinguish fixed-output from natural-EOS lanes.
- Long context/residency: staged 32K through 256K qualification, independent
  retrieval checks, memory/capacity and cold-start versus warmed residency.

The Rust `test_qwen4_exp_reference_bench_suite` is useful development evidence,
not a substitute for these standalone HTTP campaigns.

## Executed scope versus broader design targets

The selected full campaigns are complete, not every possible shape listed above.
Flash uses C1, the reference recipes' single-sequence serving limits and natural
EOS; C2/C8 and fixed 128-token reference-shaped prompts were **not measured**.
There is no Flash concurrency-capacity claim. 27B has measured C8; the two-TP1
replica point remains explicitly historical. Quick/release presets validate and
are refreshable, but only full campaigns are claimed as measured here, not three
independent restarted release trials. NInfer Flash's IQ2 artifact is excluded
from the approximately 4-bit cohort; stock Flash vLLM/SGLang support is not
inferred from specialized overlay results. These limitations remain visible in
reports rather than being counted as successful qualification cells.

## Scope discipline

Implement only runner infrastructure needed to make these comparisons honest.
Full managed orchestration, autonomous live-tool loops, all vision/structured
output modes and endurance testing remain separate backlog unless a measured
failure makes one necessary. Preserve unsupported/failed evidence without calling
an incomplete, untested matrix complete.
