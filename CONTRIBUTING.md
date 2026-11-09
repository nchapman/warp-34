# Contributing to warp-34

Start with [FEATURES](docs/FEATURES.md), the [readiness register](docs/READINESS.md), [architecture/contracts](docs/ROADMAP.md), and [hardware constraints](docs/HARDWARE.md). A feature's existence is not a blanket support guarantee. Report and preserve the configuration in which behavior is expected.

## Bug reports

Include:

- Readiness ID(s), expected behavior, actual behavior, and the smallest reproducible request/command.
- Source commit or image digest; model and drafter repository **and revision**; quantization/KV format; plain/MTP/DFlash2 and sampling settings.
- GPU model/count, TP, topology (`nvidia-smi topo -m`), CPU/RAM where offload is involved, driver and CUDA versions.
- Context length, concurrency, cold/warm cache state, reproducibility frequency, and whether eager/graph or plain/speculative changes the outcome.
- Sanitized logs, exact error, test output, and relevant latency/memory measurements. Distinguish expected overload rejection from unexpected failure.

Never publish API keys, authorization headers, private prompts/images, model credentials, or unredacted user data. For a suspected exploitable security/isolation defect, use the repository's private vulnerability reporting channel if enabled; otherwise arrange a private channel with the maintainer before sharing exploit details publicly. This guide does not assert that GitHub private reporting has been configured.

## Pull requests

1. Keep the change focused and identify affected readiness IDs/contracts and configurations.
2. For a bug, add a regression and demonstrate that it fails for the expected reason before the fix. Do not loosen numerical tolerances or remove assertions merely to make a check pass.
3. Preserve typed KV addressing, FP16 target/BF16 drafter/FP32 recurrent-state policy, state/grammar rollback, valid-row masking, and typed collectives. Register new CUDA specializations in execution-plan warmup before graph capture. See [contracts](docs/contracts/).
4. State compatibility changes explicitly, including API errors, streaming terminal events, schema/parser semantics, checkpoint formats, or newly unsupported configurations.
5. Run the checks below. Report missing hardware, skipped/ignored tests, and fixture limitations—not just the command exit status.
6. Attach evidence and update affected readiness records/documentation. Relevant code/compiler/model changes require scoped requalification; do not relabel an old benchmark as current.

### Verification

Check GPU occupancy first; select free cards and never interrupt another user's process. CUDA tests require `--features cuda`.

```sh
export PATH="/opt/cuda/bin:$PATH"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
python3 tools/prep_test_models.py --check
CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1 mise run check
```

Choose device IDs appropriate to the available hardware. `mise run check` includes formatting, CUDA/CPU Clippy, rustdoc, CUDA release tests, oracle-tooling/performance-tooling tests and dashboard checks. Some tests require two GPUs, large cached checkpoints, or explicit `--ignored`; see the [release checklist](docs/RELEASE_CHECKLIST.md). Native fatbinary qualification also sets `CUDA_DISABLE_PTX_JIT=1`.

Without a GPU, run `mise run test-cpu`, formatting/CPU Clippy and relevant Python/dashboard checks; disclose that GPU validation remains outstanding. Hosted CI currently builds CUDA but does not run GPU tests, and its Rust CPU execution covers library/doctests rather than all CPU integration tests. Maintainers must obtain a scoped GPU receipt for CUDA/runtime changes before treating them as qualified. Documentation-only changes need applicable documentation/link checks, not a new model benchmark.

Performance-sensitive changes also need before/after measurements on matched model revision, prompt/workload, hardware, KV/quant format and batch shape, alongside correctness results. Use `benchmarks/perf/` and `tools/oracle/`; report repetitions/variability and regressions. Small quality smoke runs and kernel timings are not end-to-end quality/serving guarantees.

### PR evidence checklist

- [ ] Scope/readiness IDs and behavioral contract identified.
- [ ] Regression demonstrated before fix, or reason this is not a bug fix.
- [ ] Checks and relevant integration tests, with exact commands and results.
- [ ] Hardware/fixtures, ignored/skipped cases, and untested configurations disclosed.
- [ ] Matched benchmarks for performance changes; correctness/quality preserved.
- [ ] Changes reviewed; compatibility/docs/readiness evidence updated.

Release acceptance additionally requires [G0–G7](docs/READINESS.md#release-gates). These are review requirements, not a claim that branch protection or GPU CI enforcement is already configured.
