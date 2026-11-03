# Completion-track review triage

Initial review: `.pi/review/call_149656223c5e418aaca48f6f7415fe5f_fc_0e0c1a151f17172e016ae984c95908819ab79849c8d6f335f1.md`.

- Fixed important Flash Next command error: `--gpus` takes card counts 1/2,
  not device indices. Physical selection belongs to server launch.
- Accepted preset validation, nested manifest shape handling, deployment identity
  reporting and artifact/template equivalence controls, early missing-evidence
  checks, and rejection of PID sampling without telemetry enabled.
- Clarified historical ledger is a projection of audit output, not byte-identical
  output from an unfiltered audit of an evolving results directory.
- Deferred arbitrary JSON size/depth security limits for operator-owned local
  bundles; this suite does not advertise a hostile-artifact ingestion boundary.
  Corrupt common manifest shapes remain worth handling, separately from a broad
  adversarial parser redesign.
- Mid-copy filesystem failures may leave partial evidence directories; these are
  not certified bundles. No automatic deletion of operator evidence is introduced.
- Hub timeout and near-zero-uptime progress issues were carried over by review,
  outside this benchmark diff. They are not modified as part of this task.

## Follow-up review

Report: `.pi/review/call_4ff4c994121542f5ab5b445b115c6cf2_fc_0e0c1a151f17172e016ae98f956028819a9844b97e8f137fd1.md`.

- **Important calibration finding rejected with measured evidence.** Independently
  re-ran `tools/perf_calibrate_quality.py` against both pinned model snapshots to
  `/tmp/w34-flash-calibration-review.json` and `/tmp/w34-27b-calibration-review.json`.
  Each generated JSON equals its corresponding retained ledger exactly, including
  distinct tokenizer/template hashes. Both independently produce counts
  `32340,32340,127540,127541,251641,251640`. Different tokenizer/template files can
  tokenize these text-only requests identically; asserting that their counts must
  differ would be an invalid test. No counts were copied between models.
- Fixed stale Flash validation count: actual command reports **8 deployments,
  5 cases**. Updated README to make attachment-based quality gating authoritative.
- Fixed omission of unretained startup receipts: the generated baseline report
  now contains **4 runs and 4 startup outcomes**. NInfer explicitly lacks a startup
  receipt; its matching attached quality evidence is assessed separately. Added
  a regression assertion, observed its expected failure, then passed the tests.
- Preserved run-time artifact declarations in generated baseline evidence rather
  than retrospectively changing profile digests or removing original blockers.
  The subsequent inventories (`27b-gguf.inventory.json`,
  `27b-ninfer.inventory.json`, `flash-next.inventory.json`) establish current
  bytes, not what an earlier process loaded. These are supplementary evidence.
- Historical ledger is a deliberately compact projection. Its reproduction note
  now enumerates omitted top-level fields; absent cells are not
  proof of an unsupported/unrun deployment.
- Declined atomic evidence staging and early telemetry-value validation as
  nonblocking polish. A failed bundle must remain uncertified; preserving a
  failed manifest/partial files is acceptable and fresh output names are required.
  Avoiding filesystem side effects for every operational/configuration error is
  not a release criterion.
- Unchanged Hub transport timeout finding remains outside this benchmark task.

## Final Flash evidence audit

A separate read-only evidence audit after the runtime campaigns checked new Flash
configs/reports/docs, not unchanged implementation or previously declined
suggestions. It verified all nine bundle / 13 startup / 375-of-381 counts,
recomputed case rates, checked report-input hashes and resource peaks, reproduced
all seven detailed Markdown reports and the 253-pair guarded comparison
byte-for-byte, and confirmed scope exclusions are explicit.

One important documentation error was fixed: TP1 MTP3 failed the **Python
code-output check** (`0` instead of `14`), not the separate arithmetic check
(which passed with `93`). SSE and warmup failures remain separately recorded.
No runtime changes or new measurements were needed. The audit found no other
material inconsistencies in the final evidence update.

This records review triage; full-check receipts and raw campaign outcomes are
retained separately.
