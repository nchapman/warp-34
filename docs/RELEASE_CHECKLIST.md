# Production launch checklist

Companion to [READINESS.md](READINESS.md). This is a **go/no-go worksheet**, not a record that production approval has been granted. Copy the receipt template into a release-specific artifact; leave unsupported configurations out of the advertised support matrix.

## This week's sequence

1. **Scope and baseline:** name the release owner, freeze the candidate configurations/SLOs, run strict checks, and enumerate ignored/conditional tests. Resolve correctness failures before adding features.
2. **Correctness and boundaries:** qualify the chosen model/API/hardware matrix and the interaction table in READINESS.md; run auth, abuse, cancellation and failure exercises.
3. **Sustained operation:** run the approved soak/load campaign on the actual launch hardware. In parallel, exercise the exact container, recovery and rollback procedure.
4. **Evidence and launch:** review failures/limits, close gates with receipts, publish only qualified configurations, then start a monitored canary. Reserve time for fixes and reruns; calendar pressure is not a waiver.

Suggested first qualification lane: pinned 4B GPTQ on the available RTX 3090 Ti TP1/TP2 setup, followed by the actual production model(s). This is a test-order recommendation, **not a decision to limit the product to 4B**. Add MTP, DFlash2, other formats, Ada, TP4/NVLink, remote images, and decision profiles as separately qualified scopes. Plain greedy coverage does not qualify stochastic serving defaults.

## Gate tracker

All owners are **unassigned** pending maintainer acceptance. A checkbox closes only with linked evidence and an owner/date.

- [ ] **G0 — Scope/SLOs:** model/drafter commits; image digest; GPUs/topology; driver/toolchain; KV/quant formats; APIs/client versions; sampling/spec modes; maximum context/concurrency; trust boundary; p95/p99 TTFT/ITL, error, memory and recovery budgets.
- [ ] **G1 — Baseline:** strict full check; all selected ignored/model/format tests; exact branch/skip inventory; native execution on every advertised architecture/topology.
- [ ] **G2 — Correctness:** tensor/token/quality contracts; cold/warm and eager/graph; speculative rejection and grammar; supported wire/client replay; stochastic marginal qualification when enabled.
- [ ] **G3 — Load/soak:** approved duration/request count, mixed lengths and features, cancellation churn, overload burst, ownership conservation, bounded memory and SLOs. Default proposal: at least 8 hours **and** 10,000 completed requests; approve before starting.
- [ ] **G4 — Security/failure:** auth and scope negatives; public operational endpoints; bounded image/grammar/body workloads; cache isolation; worker/bridge failure propagation and supervisor recovery.
- [ ] **G5 — Quality/performance:** representative task/long-context evidence, predeclared tolerances, known-regression disposition, matched benchmark claims.
- [ ] **G6 — Deployment:** exact digest under production user/cache/driver; real request readiness; stall alert; shutdown, restart, key rotation, upgrade and rollback rehearsal.
- [ ] **G7 — Sign-off/contribution:** named owner per in-scope capability, release manifest, durable artifacts, PR validation policy, no unresolved in-scope blocker, canary/rollback criteria approved.

**Current CI limitation:** `.github/workflows/docker-image.yml` runs CPU checks and builds the CUDA image, but does not execute a GPU qualification job. Its CPU test step uses `cargo test --lib` and doctests, not all CPU integration tests. A green workflow is not G1/G2. Until GPU receipts are enforced automatically, the release owner must review them before creating a `v*` tag (which publishes release image tags). Do not claim this document changed branch protection or CI enforcement.

## Baseline commands

Do not overlap GPU campaigns on the same cards. Check occupancy before every launch, select free devices, and keep all exit codes. Do not download an entire optional checkpoint matrix unintentionally.

```sh
export PATH="/opt/cuda/bin:$PATH"
git status -sb
git rev-parse HEAD
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
nvidia-smi topo -m

# Check-only: does not download missing checkpoints.
python3 tools/prep_test_models.py --check
# Optional broader checkpoint inventory (then acquire only what's needed).
python3 tools/prep_test_models.py --all --check

# Actual CUDA release suite; helper-based missing prerequisites must fail.
CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1 CUDA_DISABLE_PTX_JIT=1 mise run check

# Broader CPU integration coverage than the current hosted CI job.
mise run test-cpu
```

`mise run test-prep` acquires the core pinned snapshots if missing. `mise run test-prep-all` acquires the broader matrix and may require substantial storage/downloads. Consult `tools/prep_test_models.py` and [MODELS.md](MODELS.md) first.

Strict mode does **not** execute `#[ignore]` tests or guarantee that topology/VRAM-specific branches ran. List available tests, read their prerequisites, and run selected qualification cases explicitly. Examples (require appropriate real checkpoints and two sufficiently sized GPUs):

```sh
CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1 CUDA_DISABLE_PTX_JIT=1 \
  cargo test --release --features cuda --test qwen4_flash_next \
  test_qwen4_flash_next_128tok_plain_mtp3_dflash2_exact_parity \
  -- --ignored --exact --test-threads=1 --nocapture

CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1 CUDA_DISABLE_PTX_JIT=1 \
  cargo test --release --features cuda --test quant_formats \
  cached_hf_quantized_and_unquantized_checkpoints \
  -- --ignored --exact --test-threads=1 --nocapture
```

For TP4/NVLink, run the relevant `collectives_and_gdn_wy` cases and selected full-model parity on that hardware, recording **which transport and branches actually executed**. Two visible GPUs do not qualify four ranks; two PCIe cards do not qualify P2P/NVLink.

Use the existing `benchmarks/perf/` campaign tooling and `tools/oracle/` drivers as starting points for G2/G3/G5. The new [`w34perf soak` mode](../benchmarks/perf/SOAK.md) enforces duration **and** successful-completion targets, sustained/burst admission, local disconnect injection, completion-progress and optional endpoint memory-growth gates, and a post-drain inference probe. Its checked-in coding campaign is synthetic/replay, not a production trace corpus; select approved representative fixtures and thresholds first. A passing short harness smoke is not G3 qualification, and image/grammar/Responses-cancel mixtures, internal ownership checks and supervisor recovery still require separate exercises. `mise run oracle-check` tests the Python tooling; it is not a live vLLM/SGLang parity run. `mise run vision-gates` and `mise run needle-gates` are opt-in GPU/model campaigns; inspect the commands and pins before executing. Existing short benchmark campaigns are not automatically the sustained G3 soak. Additional workload/failure drivers may be needed.

## Fresh verification from this assessment

- Source: `507661b0e1271e4f3757e824aa1fe1b7c3baf811`, 2026-11-09; this assessment changes documentation only.
- Hardware: two RTX 3090 Ti GPUs; no compute processes reported before starting; approximately 24 GiB free per card.
- `python3 tools/prep_test_models.py --check`: **PASS**, 15 tracked fixtures plus the four core snapshot entries. The 27B drafter prerequisite is config-only; this does not verify all optional real-model artifacts.
- `CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1 CUDA_DISABLE_PTX_JIT=1 mise run check`: **PASS**, exit 0 in 332.79 seconds. **476 Rust tests passed, 0 failed, 13 ignored** (49 test-result summaries, including doctests); 69 oracle-tooling, 91 performance-tooling and 18 dashboard tests passed. Formatting, both Clippy modes and rustdoc also passed. Driver `615.71.09`; two `24,564 MiB` cards, `PHB` topology.
- Local raw log: `/tmp/w34-readiness-check-507661b.log`, SHA-256 `aee547d5a60fa7772bc40f88f2a5bffc61711ce2584f61c7a0f3115e2eea00c7`. This checked-in summary preserves the outcome, not the raw log; upload the log to durable release storage before using it as a release receipt. No printed `[SKIP]` lines were captured, but passing-test output is normally captured by Cargo: this is **not** proof that every conditional branch executed.
- Ignored cases: `bench_prefill_query_tiles`, `benchmark_flash_next_pool`, `canonical_checkpoint_host_store_tp1_and_ep2_agree`, `pinned_hgptq_dflash2_lossless_and_tp2`, `pinned_hgptq_moe_35b_text_vision_mtp_and_tp2`, `pinned_hgptq_text_graph_mtp_and_tp2`, `pinned_hgptq_vision_matches_vllm_and_tp2`, `benchmark_block_hadamard`, `cached_hf_quantized_and_unquantized_checkpoints`, `test_qwen4_exp_real_checkpoint_tp1_and_tp2`, `test_qwen4_flash_next_128tok_plain_mtp3_dflash2_exact_parity`, `test_qwen4_exp_reference_bench_suite`, and `benchmark_cached_qwen_constraint`. These include qualification tests, not just optional timing benchmarks.
- Fresh ignored real-model/format qualification, external oracle, performance/quality campaign, sustained soak, container, Ada, NVLink and TP4 runs: **not performed in this assessment**.

## Lifecycle hardening verification

Source: the hardening changes committed with this receipt on `fix/production-lifecycle-hardening`, parent `bc2a3e5`. This supplements rather than replaces the documentation-only baseline above.

- Same local two RTX 3090 Ti / driver 615.71.09 / PHB setup; GPUs idle before and after the campaign.
- `CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1 CUDA_DISABLE_PTX_JIT=1 mise run check`: **PASS**, exit 0, 346.83 seconds; **491 Rust tests passed, 0 failed, 13 ignored**, 69 oracle-tooling, 91 performance-tooling and 18 dashboard tests. Formatting, both Clippy modes and rustdoc passed. The ignored inventory above is unchanged. An earlier attempt failed the new test modules' placement lint; corrected before this rerun.
- Full-check log: `/tmp/w34-lifecycle-hardening-check-final.log`, SHA-256 `7b875a4b965d4ec86db91a6a7575ca2f91916f77486ed373913002f0a62f6f85`.
- Targeted live Responses cancellation/readiness and native sampled-terminal-grammar tests also passed independently. The source review reported no actionable findings.
- `cargo build --release --features cuda` and three sequential benchmark smoke runs passed, using `CUDA_VISIBLE_DEVICES=0,1 CUDA_DISABLE_PTX_JIT=1` and the common command below (default TP1). Each reported radix/vision lossless parity, distinct-image cache isolation, eight active scheduler sequences and 256 total scheduler output tokens.

```sh
./target/release/w34 bench --model nchapman/Qwen3.5-4B-W4A16-HGPTQ --offline \
  --stage all --kv-format tq4 --prompt-len 128 --gen-len 128 --batch-size 8
# Repeat with --speculative mtp --mtp-k 3
# Repeat with --speculative dflash2 --draft-model nchapman/Qwen3.5-4B-DFlash2
```

| Mode | Decode tok/s (single run) | Local log | SHA-256 |
|---|---:|---|---|
| Plain | 211.68 | `/tmp/w34-lifecycle-bench-plain.log` | `238622b2cc07ff177380b94123abcab22c211b5ce0203c47ee26b70f4cfd7e3a` |
| MTP-3 | 537.66 | `/tmp/w34-lifecycle-bench-mtp.log` | `b0c7649695f9568e58ba8c13dd994e27e580411b2a3c00da6a23a1444f53be5f` |
| DFlash2 | 1427.58 | `/tmp/w34-lifecycle-bench-dflash2.log` | `51ee7c9b702e739ac431f228cfd235ce8b474480b24ceea2565d4bb3075dc95a` |

These are smoke results, not matched before/after, competitor, or sustained-load qualification. Logs remain local and need durable upload for release sign-off. Sampled stateful-grammar handoff, external hard-stop/restart rehearsal, and all previously open hardware/model/soak gates remain open. See [READINESS.md](READINESS.md#focused-lifecycle-hardening-follow-up) for scope and [SERVING.md](SERVING.md#readiness-shutdown-and-bounded-recovery) for the operational procedure.

## Release receipt template

Fill one per configuration; shared evidence may be referenced, not copied and relabeled. Do not place credentials or private user prompts in reports.

```yaml
release: REQUIRED
assessed_at: REQUIRED
owner: REQUIRED
source_commit: REQUIRED
image_digest: REQUIRED
binary_sha256: REQUIRED
scope:
  readiness_ids: []
  model_and_revision: REQUIRED
  drafter_and_revision: NONE_OR_PIN
  hardware_topology_cpu_ram: REQUIRED
  driver_cuda_toolchain: REQUIRED
  tp_kv_quant_spec_sampling: REQUIRED
  context_concurrency_request_limits: REQUIRED
  apis_client_versions: REQUIRED
  trust_boundary_ingress_egress: REQUIRED
  excluded_capabilities: []
slo:
  latency_error_memory_recovery_budgets: REQUIRED_BEFORE_RUN
dimensions:
  correctness_quality: Unknown
  compatibility_coverage: Unknown
  reliability_recovery: Unknown
  security_isolation: Unknown
  performance_capacity: Unknown
  operability_regression: Unknown
gates:
  G0: OPEN
  G1: OPEN
  G2: OPEN
  G3: OPEN
  G4: OPEN
  G5: OPEN
  G6: OPEN
  G7: OPEN
evidence: [] # command, exit code, executed/skipped cases, artifact URI + SHA256
known_limits_and_risks: []
rollback_digest_and_trigger: REQUIRED
canary_monitoring_and_owner: REQUIRED
decision: NO_GO
signoff: REQUIRED
```

### Canary / rollback minimum

Before sending production traffic, freeze the initial traffic/concurrency limit, observation window and on-call owner. Alert on worker unavailability/stalled completions, unexpected error rate, sustained latency breach, and memory growth. Any state corruption, isolation failure, deadlock, or correctness-contract violation is an immediate stop/rollback trigger; do not wait for an averaged error budget. Other thresholds come from G0. A last-known-good image **and its compatible model/configuration** must be available. Restart loses in-memory Responses history; clients must have a defined retry/recovery policy. Do not assume retrying a partially streamed response is transparent.
