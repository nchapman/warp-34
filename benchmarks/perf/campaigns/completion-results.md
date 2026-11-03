# Completion campaign evidence — in progress

These are qualification results, **not a publication-ready winner table**.
Historical results remain separate. Raw bundles live under `benchmarks/perf/results/`.

## Current measured cells

| Deployment | Quality fixture | HTTP evidence | Interpretation |
|---|---|---|---|
| warp-34 66974d8 27B TP1 FP16 | 6/6 baseline checks | `completion-27b-warp34-tp1-fp16`: 96/96 requests, 3 repetitions | Qualification; overlapped reference build/download and later a second-card probe; not an isolated publication trial |
| NInfer 115dee6b 27B TP1 INT8 KV | 6/6 baseline checks | `completion-27b-ninfer-tp1-full`: 96/96 requests, 3 repetitions | Practical mixed-Q4/Q5 deployment, not identical GPTQ; one server lifetime |
| llama.cpp 25ae3a9b3 27B TP1 FP16 KV | 5/6 baseline checks | `completion-27b-llama-tp1-quick`: 12/12 requests | Quality failed: returned `30` for Python `sum(x*x for x in range(4))`, expected `14`; retain metrics but do not call quality-equivalent |

The initial `completion-27b-warp34-tp1-quality.json` used the 1200-line retrieval
fixture against an 8192-token-capacity profile. Four checks passed; both larger
retrieval requests returned HTTP 400. The full fixture is retained as
`fixtures/quality-long.json`; subsequent `quality-short` evidence uses a separate
256-line baseline fixture. Passing baseline checks does not clear the original
capacity failures.

The serial baseline campaign completed vLLM, llama.cpp and warp-34 under
`completion-27b-baselines/`: **96/96 HTTP requests per deployment**, three
within-process repetitions. vLLM and warp-34 pass baseline quality; llama.cpp
retains its quality failure. Together with NInfer, the derived
[`27b-baselines-measurements.md`](../qualification/27b-baselines-measurements.md)
and accompanying JSON cover **four runs and four startup outcomes**. NInfer's
startup receipt was not retained, explicitly distinguished from its passing
attached quality evidence. Builds/downloads still overlapped some host activity;
these are not fully isolated publication trials.

The **20-profile advanced 27B matrix is complete**: 19 deployments reached HTTP
readiness and completed all measurements; vLLM MTP2 failed startup as detailed
below. Baseline quality passes in 17 of the 19 measured deployments. llama.cpp
Q8 KV and TP2 layer-split profiles fail the arithmetic check; Q4 KV passes this
small fixture, which is not proof that Q4 is generally more accurate.

Both fresh native **TQ4 and TQ4_all capacity profiles pass all six early/late
retrieval checks**, with actual server usage of 32,340 / 127,540–127,541 /
251,640–251,641 prompt tokens. Their long HTTP workloads also complete.
[`27b-advanced-measurements.md`](../qualification/27b-advanced-measurements.md)
and accompanying JSON retain **21 measured deployments, 22 startup outcomes,
1,944/1,944 HTTP requests**, including both capacity profiles. Together with the
four baseline runs this is 2,328 successful measured HTTP requests, not independent
restart trials or universally quality-equivalent results.

The guarded baseline comparison is retained under `results/`, identified by
[`27b-comparison-receipt.json`](../qualification/27b-comparison-receipt.json).
All 120 case/deployment pairs retain incompatibilities or blockers; no ranking
or artifact-equivalence certification is emitted.

### Evidenced startup failures

- **`vllm-tp2-mtp2`**: exited 1 before HTTP readiness. Both ranks report
  `ValueError: There is no module or parameter named
  'layers.0.mlp.down_proj.weight_packed' in Qwen3_5MultiTokenPredictor`; the
  advertised parameter is `layers.0.mlp.down_proj.weight`. Evidence:
  `completion-27b-kv-spec/vllm-tp2-mtp2/startup.log` (lines 129/201) and
  `qualification.json`. This is a failure of this pinned packed-MTP checkpoint
  and recipe, **not** a claim that all vLLM MTP models are unsupported. No latency,
  quality pass or zero-throughput score is assigned to the failed startup.

Each driver-managed deployment is restarted, its advertised model ID checked,
then independently quality-tested before HTTP measurement. The operator driver
retains startup failure evidence and stops only its own processes/containers.
This driver is external to the portable runner.

## Reproduction and comparison

- Frozen profiles: `27b-completion.toml`, `27b-kv-spec-completion.toml` and
  `flash-next-serving.toml`; engine launch commands remain explicit/operator-owned.
- Run `python3 -m w34perf quality ...` before `run ... --preset full`, attaching
  quality/startup/inventory evidence with `--evidence` and enabling telemetry.
- `python3 -m w34perf compare <run-directory> ...` retains different cohorts,
  fixture/decoding/hardware/artifact controls, quality failures and provenance
  blockers. It does not certify or rank incompatible runs.
- Quick/full/release presets use 1/3/5 within-process repetitions respectively.
  Publication still requires independently restarted/interleaved trials and
  artifact/source/template qualification, not just more within-process samples.

## Reference preparation

- Downloaded pinned 27B GGUF and ninfer artifacts listed in
  `qualification/reference-pins.json`; actual runtime probes now exist.
- Built NInfer from pinned source with CUDA sm_86 and the pinned Flash Next
  vLLM-overlay Docker recipe. Flash image ID:
  `sha256:d52ca1279fdc80543a7ca3538cd10a0bea3f90884866f7497c66aa7792acd4f1`.
- The Flash reference wrapper mounts only its model directory. HF snapshot
  symlinks would point outside that mount, so a dereferenced hard-link view was
  prepared on the same filesystem at `~/.cache/w34-benchmark-models/flash-next`.
  It uses the same cached bytes without another 121 GiB copy; reference mounts
  remain read-only. Native warp profiles keep the original snapshot path.

## Capacity qualification and current outstanding work

Pinned tokenizer/template calibration is retained in
`qualification/{27b,flash}-capacity-calibration.json`: the six early/late probes
render to 32,340 / 127,540–127,541 / 251,640–251,641 reference tokens. Both ledgers
were independently reproduced after review; matching counts are not evidence of
shared files or fabricated calibration. These counts remain **reference renderer
counts**, not actual server attestations.

`27b-capacity-completion.toml` completed with `fixtures/quality-capacity.json`;
`flash-next-longctx.toml` remains queued with that fixture. Quality and HTTP
outcomes must be retained even when capacity requests fail. Strata's native
`--spec 0` profile failed before readiness: this IQ pack requires a verify window
of at least 2. The original evidence remains in `completion-flash-strata/`.
A separately named prompt-lookup window-4 profile (no MTP drafter) is queued;
this is a declared recipe change, not an overwritten target-only result.
See `strata-preparation.md` for commands.
The one-GPU specialized Flash overlay has its own built image and queued profile,
not an alias for the two-GPU image.

Not yet complete: Flash HTTP execution and resource
qualification, actual long-context quality/capacity outcomes, final cross-campaign
indexes and full checks. The follow-up code review is triaged; `perf-check` passes
89 tests. No publication certification or task commit has been made.
