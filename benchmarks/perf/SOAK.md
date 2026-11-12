# Extended workload / soak runner

`w34perf soak` extends the finite benchmark runner with repeated, bounded HTTP
workload execution. It **does not launch or restart servers, execute tools, inject
GPU faults, or approve a production release**. A passing receipt means the selected
client-side checks passed for the recorded workload and policy.

## Workload and deployment

Copy [`campaigns/soak-coding.toml`](campaigns/soak-coding.toml), fill in the exact
model/drafter revisions, server source/image and launch metadata, and choose the
endpoint and request settings. It mixes:

- Unique technical/coding tasks with varying prompt and output lengths.
- Four-turn coding-agent replay: recorded tool observations plus actual generated
  assistant responses carried into the next turn.
- Repeated shared-prefix requests to exercise warm-cache behavior.

The fixtures are sanitized synthetic examples, **not a production trace dataset
or an autonomous coding benchmark**. Replace or augment them with representative,
permissioned fixtures before claiming workload representativeness. Tool results
are replayed, never executed. Generated prose is not automatically scored for
semantic correctness. The default example uses greedy sampling; explicitly set
`request_overrides` and record them to qualify sampled or speculative serving.

All cases in the selected configuration participate. Use a dedicated config,
not the full benchmark suite with its unrelated open-loop/priming cases. Soak
uses closed-loop sessions, round-robin case selection, and a global concurrency
limit based on the largest case concurrency. Case `requests` and run `repetitions`
are finite-run settings, not soak stop conditions. `warmup_requests` counts
complete sessions per case. Natural cache sharing is supported; explicit priming
and open-loop arrival policies are not soak modes.

Launch the server separately using the configuration's recorded command. For the
example configuration, from the repository root on an otherwise idle GPU:

```sh
export PATH="/opt/cuda/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 ./target/release/w34 serve \
  --model nchapman/Qwen3.5-4B-W4A16-HGPTQ --offline \
  --kv-format tq4 --kv-pages 4096 --max-num-seqs 8 --max-model-len 8192 \
  --port 18834
```

Keep this in the foreground under your normal service/task manager. Record its
actual host PID for RSS sampling. For authenticated endpoints, configure
`api_key_env` with an environment-variable **name**, never a literal secret.
Logs contain request/response text; treat artifacts as sensitive if your fixtures
are sensitive. Raw evidence grows with workload volume; provision disk space.

## Short harness smoke

From `benchmarks/perf`, against that already-running endpoint:

```sh
python3 -m w34perf validate campaigns/soak-coding.toml
python3 -m w34perf soak campaigns/soak-coding.toml \
  --engine warp34-soak --out results/soak-smoke-001 \
  --duration-s 60 --min-completed 100 --max-duration-s 180 \
  --sustained-s 20 --burst-s 5 --cancel-every 10 \
  --progress-timeout-s 90 --drain-s 5 \
  --telemetry-interval 1 --server-pid "$SERVER_PID"
```

The output directory must not exist. This command **does not enforce a memory
growth budget**; telemetry alone is evidence, not a resource pass. Successful
completed responses, not sessions, count toward `--min-completed`: a four-turn
agent session can contribute four. Intentional disconnects, warmup and the drain
probe do not count. Both elapsed workload duration **and** completion count must
be satisfied. Reaching only one is not success. In-flight sessions are drained;
counts may exceed the requested minimum.

## Proposed extended campaign

Approve the workload, thresholds and hardware first. The repository's proposed
8-hour / 10,000-completion minimum is not an automatic release decision. Example:

```sh
# Set SERVER_PID and GPU_UUID explicitly for the actual local deployment.
# Discover UUIDs with: nvidia-smi --query-gpu=index,uuid --format=csv
python3 -m w34perf soak campaigns/soak-coding.toml \
  --engine warp34-soak --out results/soak-candidate-001 \
  --duration-s 28800 --min-completed 10000 --max-duration-s 36000 \
  --sustained-s 300 --burst-s 30 --burst-multiplier 2 \
  --cancel-every 10 --cancel-after-output-events 2 \
  --progress-timeout-s 120 --drain-s 10 --max-error-rate 0 \
  --telemetry-interval 1 --server-pid "$SERVER_PID" --gpu-uuid "$GPU_UUID" \
  --max-rss-growth-mib 512 --max-gpu-growth-mib 256
```

**All budgets above are examples, not approved SLOs.** Repeat `--server-pid` and
`--gpu-uuid` for each relevant local identity. For a remote endpoint, local
`nvidia-smi` and `/proc` describe the load-generator host, not the server; collect
server telemetry separately and do not attach misleading local growth gates.

### Execution and checks

- **Bounded submission:** no growing executor queue; only the current concurrency
  limit is admitted. The single-GPU example alternates 4-session sustained and 8-session
  burst phases. Existing sessions finish when transitioning back to sustained;
  the limit is an admission target, not forced eviction of active sessions.
  Workload latency determines attained throughput/concurrency. For a separately
  qualified dual-GPU deployment, a practical starting shape is 8 sustained / 16
  burst: set case concurrency to 8, `run.max_inflight` to 16, and record the actual
  TP2 server launch/capacity. TP2 does not guarantee twice the throughput.
- **Deadline and progress:** the measured workload stops admitting when both
  targets are met, or fails at its maximum duration or lack-of-completed-response
  deadline. Request timeouts are bounded by the remaining workload deadline.
  Warmup, setup/provenance, and final drain/probe are additional time; this is not
  an OS-level total-process timeout. Use an external supervisor if a hard process
  lifetime is required. `/health` alone is not a forward-progress check.
- **Disconnect injection:** every Nth measured request receives an early local
  stream close after the specified number of output-bearing SSE events. These
  are events, not tokens. Expected local closes are counted separately from
  unexpected errors and do not satisfy completion/progress targets. Completion
  before injection is recorded separately. This is **not** a Responses cancel
  endpoint exercise or acknowledgment of server-side cancellation/resource reuse.
- **Errors and latency:** unexpected errors are checked against the configured
  maximum fraction. Missing/invalid token usage is not a successful measurement.
  Configured `ttft_slo_ms`, `e2e_slo_ms` and `tpot_slo_ms` are per-response ceilings;
  a breach fails the run. They are not p95/p99 percentile thresholds. Choose
  ceilings that make sense for the selected context/output/concurrency mix.
- **Resource checks:** optional per-PID RSS and explicit per-UUID GPU memory
  growth compare post-warmup baseline with post-drain readings. Missing readings
  for a requested budget fail closed, rather than becoming zero usage. These are
  endpoint growth checks, not peak-memory, slope, pinned-memory, or allocator
  ownership-conservation proofs. PID reuse/restarts are not detected; keep the
  sampled server processes fixed and qualify recovery separately. Continuous
  telemetry permits further analysis.
- **Drain:** after bounded completion of admitted client work, wait a quiet
  interval and send a noncancelled real inference probe. Its success is a recovery
  signal, not proof that every server request or pool resource has been reclaimed.

## Receipts and limitations

The new directory retains `manifest.json`, append-only `requests.jsonl`,
`telemetry.jsonl`, `soak_summary.json`, and `SHA256SUMS.json`. The manifest records
configuration, fixture/source provenance and policy; the summary records outcomes,
targets, counts, failures and resource checks. Interrupted campaigns retain partial
evidence where filesystem writes remain possible. Unexpected exceptions must not
be interpreted as passed runs.

Exit codes: **0** checks passed, **1** completed with failed checks, **2** invalid
configuration/setup error, **130** interrupted. Inspect the receipt, not just a
zero exit code. An unconfigured gate is not an implicit pass. The existing
`report`, `compare` and `audit` commands consume the finite campaign schema;
**do not use them to certify soak bundles**. Read `soak_summary.json` directly and
verify the SHA-256 file inventory before transferring evidence to release storage.

Still outside this increment: image/grammar/Responses lifecycle mixtures,
autonomous tools, semantic answer scoring, rank/device fault injection, server
termination/restart/rollback, internal pool-conservation assertions, and fully
automated production sign-off. A short smoke run does not qualify the eight-hour
campaign or close [G3–G6](../../docs/RELEASE_CHECKLIST.md#gate-tracker).
