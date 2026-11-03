# Standalone serving performance suite

Engine-neutral, **4-bit-weight deployment comparisons** for 1–2× RTX 3090 Ti.
Python 3.11+ standard library only. Copy this directory to another machine and
run it without Cargo, CUDA libraries, engine packages, or `tools/oracle`.
This is a black-box HTTP client, **not** an oracle, profiler, kernel microbench,
or automatic benchmark-results claim.

- [Plan and persistent task list](PLAN.md)
- [Engine recipes, qualification, and KV matrix](engines.md)
- [Experiment template](suite.toml) — 10 deployment profiles, 13 workload cases
- [First 27B FP16 comparison](campaigns/27b-fp16-first.md) — warp-34, vLLM, SGLang

## Quick start

From the repository root, start **one** server (no download or process management
is performed by the suite):

```sh
export PATH="/opt/cuda/bin:$PATH"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
CUDA_VISIBLE_DEVICES=0 ./target/release/w34 serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ \
  --kv-format fp16 --kv-pages 4096 --port 8000
```

In another terminal:

```sh
cd benchmarks/perf
python3 -m w34perf validate suite.toml
python3 -m w34perf plan suite.toml                 # no network or GPU access
python3 -m w34perf run suite.toml --engine warp34-1-fp16 \
  --case smoke --out results/warp34-fp16-smoke-001
python3 -m w34perf report results/warp34-fp16-smoke-001 > /tmp/perf-report.md
python3 -m w34perf report results/warp34-fp16-smoke-001 --format csv > /tmp/perf-report.csv
```

An existing output directory is never overwritten. Select several `--case`s or
omit `--case` to run the entire battery. Engine selection is mandatory. Each
profile describes the server **you have actually launched**; the runner never
runs `launch_command`. Most template profiles deliberately share a port: run
those one at a time, stopping/relaunching the server between profiles. Selecting
two profiles sharing an endpoint is rejected to prevent accidental relabeling.
Multiple `--engine`s require distinct already-running endpoints and isolated
hardware. Don't run competing engines simultaneously on the same cards.

`run` exits 0 for completed, usage-accounted trials **or an explicitly all-skipped
selection** (see coverage); 1 for request/session or
token-accounting failures; 2 for configuration/setup errors; 130 on interrupt.
A zero exit does **not** mean the run is publication-ready. Unknown provenance,
quality evidence, failures, and release-audit requirements are visible in reports.
No engine has been automatically installed, modified, or tuned.

## Completion campaigns, quality and retained evidence

The [completion ledger](COMPLETION_PLAN.md) tracks current 27B and Flash Next
work separately from historical results. The new campaign configurations are
`campaigns/27b-completion.toml` and `campaigns/flash-next-serving.toml`.

```sh
python3 -m w34perf plan campaigns/27b-completion.toml --preset quick
python3 -m w34perf quality campaigns/27b-completion.toml \
  --engine warp34-tp1-fp16 --out results/quality-001.json
python3 -m w34perf run campaigns/27b-completion.toml \
  --engine warp34-tp1-fp16 --preset full --telemetry-interval 1 \
  --evidence results/quality-001.json --evidence /path/to/sanitized-startup.log \
  --out results/run-001
python3 -m w34perf audit results > /tmp/historical-ledger.json
python3 -m w34perf compare results/run-001 results/run-002 > /tmp/comparison.json
```

`quick`, `full`, and `release` explicitly select campaign cases and repetitions;
`--preset` cannot be combined with `--case`. Release repetitions are within one
server lifetime, **not independent restarts**. Repeat release campaigns with
fresh server processes and interleaved engine order for publication. None of
these commands certify publication readiness or rank incompatible deployments.

`quality` exits nonzero on failed checks and preserves responses and hashes.
The default fixture tests small functional tasks and retrieval up to 256 archive
lines; `--fixture fixtures/quality-long.json` separately tests 1200-line retrieval.
These are not guaranteed token counts or proof of general task quality. Record
actual usage and capacity failures; passing short tests never clears long-context
failures. Quality results apply only to the exact fixture and deployment tested.

`fixtures/quality-capacity.json` adds early/late retrieval around 32K/128K/252K.
Its pinned 27B reference-template calibration is retained in
`qualification/27b-capacity-calibration.json` (32,340 / 127,540–127,541 /
251,640–251,641 tokens). Reproduce offline from the repository root with the
optional preparation tool (requires Transformers, **not** required by w34perf):

```sh
python3 tools/perf_calibrate_quality.py /path/to/pinned/snapshot \
  benchmarks/perf/fixtures/quality-capacity.json --out /tmp/capacity-calibration.json
```

The tool requires local tokenizer bytes, disables remote checkpoint code and
records tokenizer/template/fixture hashes. These are reference-renderer counts,
not attestations of a server's chat renderer. Use actual server usage to label
tested capacity; do not run this large fixture on an 8K profile and call rejection
a retrieval-quality regression.

`--evidence` is repeatable and copies files into the immutable run bundle. Supply
only sanitized logs: attachments are **not automatically redacted**. Hashes prove
byte identity, not source authenticity or remote model residency. Attaching a
report does not certify publication readiness. A recognized quality report only
clears the missing-quality blocker for its exact profile; any attached matching
failure takes precedence over passes. Other provenance/release blockers remain.

Telemetry is opt-in. `--telemetry-interval 1` records GPU UUID, memory, utilization,
clocks, temperature, power and host memory; repeat `--server-pid PID` to include
explicit host-process RSS. Samples include timestamps and source errors; missing
readings remain unknown, not zero. Sampling subprocess overhead is part of the
measurement environment. Compare sampled and unsampled runs before interpreting
small speed differences. No energy claim is derived from nameplate power.

## Optional two-card runs and coverage

Use the same filters for `plan` (offline) and `run`:

```sh
python3 -m w34perf plan suite.toml --gpus 1
python3 -m w34perf run suite.toml --engine warp34-1-fp16 --gpus 1 \
  --case smoke --out results/single-only-001
# Exclusions can be repeated; other already-running selected endpoints still run.
python3 -m w34perf plan suite.toml --gpus 2 --exclude-engine ninfer-2-experimental
```

Each deployment has `enabled = true` by default and `support = "unverified"`.
Use `support = "supported"` only after direct qualification; unverified profiles
remain runnable for probes, with a publication blocker. Mark a known missing
backend `support = "unsupported"`, or temporarily opt out with `enabled = false`.
Either requires a nonempty `skip_reason`. Minimal placeholders need no fake URL,
model, launch command or quantization fields:

```toml
[[engines]]
id = "example-2"
engine = "example"
gpu_count = 2
support = "unsupported"
skip_reason = "This release only supports one CUDA device"
```

The ninfer two-card example is **disabled/unverified**, not declared unsupported:
upstream documents a Linux pipeline but has not requalified it. Opt in only for
an explicit probe. No automatic GPU-count, KV, or parallelism fallback occurs.

`coverage.json` retains every configured deployment × **selected case**, including
`unsupported`, `disabled`, `not_selected`, `excluded`, `filtered`, `not_run`,
`in_progress`, `failed`/`interrupted`, `measurement_failed`, and `measured` cells.
Availability reasons take precedence over operator filters. Completed repetitions
count trials that produced a summary, even if usage/requests failed. If nothing
runnable remains, the run writes `status=skipped`, an empty journal/summary and
checksummed coverage **without network calls, credentials, or GPU probes**.
Skipped rows have no latency/throughput values, not zero. Markdown includes a
coverage table; CSV uses `row_type=trial|coverage` with blank measurement columns
for coverage. Filter `row_type=trial` before computing performance statistics.

## Offline artifact inventory

Hash a pinned snapshot (including tokenizer/template and optional sidecars), or
a single GGUF/`.ninfer` file, independently of engine internals:

```sh
python3 -m w34perf inventory /models/pinned-snapshot --out results/weights-001.json
python3 -m w34perf verify results/weights-001.json
# Relocated copies can be checked without changing the original inventory:
python3 -m w34perf verify results/weights-001.json --root /other/pinned-snapshot
# Optional repeated relative-name globs deliberately narrow the inventory:
python3 -m w34perf inventory /models/pinned-snapshot \
  --include '*.safetensors' --include '*.json' --include '*.jinja' \
  --out results/selected-files-001.json
```

Inventories are exclusive-create, outside the model directory, and use streaming
SHA-256 over file **contents** (including HF blob symlinks). They record logical
names, sizes, total bytes and a path-independent `inventory_sha256` for the
`artifact_sha256` field. Verification rehashes the same visible-file/glob scope,
checks missing/added/changed files, and exits 1 on mismatch (2 for setup errors).
Dotfiles/directories and symlink directories are excluded. Include globs are
saved: an intentionally narrower inventory does **not** certify excluded files.
Inventories, like configs, are **trusted operator inputs**: the recorded `root`
and `include` control which local paths are read. Inspect received inventories
and choose `--root` yourself; do not verify untrusted files blindly. Hashes detect
byte differences against a trusted inventory, **not authenticity**: there is no
signature, and a replacement inventory can certify replacement bytes. Retain
an out-of-band trusted hash or externally sign the evidence for publication.

`verify` checks local bytes, not shard-index completeness, source-model equivalence,
quality, advertised model IDs, or what an HTTP server actually loaded. No model
downloads occur. Use the separate preflight for indexed shards and model discovery:

See [27B candidate evidence](qualification/27b-candidates.json) and
[qualification notes](engines.md#27b-candidate-mapping-2026-10-30) for verified
local hashes versus remote-advertised hashes and unresolved provenance.

## Online preflight and 27B one-card probes

After starting a server, verify a **trusted** inventory and its indexed shard
references, then compare the configured model ID to `/v1/models`:

```sh
python3 -m w34perf preflight qualification/27b-one-card.toml \
  --engine vllm27-1-fp16-0df \
  --inventory qualification/27b-gptq-0df.inventory.json \
  --out results/vllm27-preflight-001.json
# Add --root /other/pinned-snapshot if local artifact bytes were relocated.
python3 -m w34perf run qualification/27b-one-card.toml \
  --engine vllm27-1-fp16-0df --gpus 1 --out results/vllm27-probe-001
```

Preflight is single-attempt, deadline/response-size bounded and exclusive-create.
It retains passed/failed/skipped evidence, checks the profile's inventory hash
when known, and refuses missing, uninventoried or parent-traversing indexed shards.
Local drift/index failure happens **before HTTP**; disabled/unsupported profiles
skip without reading inventories or contacting servers. Exit 0 means passed or
explicitly skipped; 1 failed inventory/index/discovery checks (including missing
or invalid inventory inputs); 2 config/selection/evidence-file setup errors.
Ctrl-C writes `status=interrupted` evidence and exits 130. It does
**not** attest which bytes the server loaded, prove rendering/quality equivalence,
or automatically attach evidence to a subsequent run. Keep inventory, preflight,
startup log and run bundle together. Inventory/root paths remain trusted inputs.

The checked-in [one-card preset](qualification/27b-one-card.toml) pins the newer
GPTQ `0df81ea...` for C1, **offered C8**, four-turn agent replay and an added
primed longer-prefill case, three repetitions and 64-token ceilings (32 for
primed prefill). Earlier three-engine runs used only the original three cases;
their raw manifests remain authoritative. The additional SGLang FP16 profile
explicitly sets its supported `SGLANG_MAMBA_CONV_DTYPE=float16` environment
option: `--dtype float16` alone does not change its default BF16 convolution
history. FP32 recurrent SSM state remains separate. This is the **only current GPTQ
candidate**: obsolete `c11a884...` was removed from the local cache and local
`refs/main` now points to 0df. Old hashes/failures remain historical evidence,
not instructions to fetch or reuse c11. Never start these profiles
simultaneously on the same GPU/port. See the [qualification ledger](qualification/27b-one-card.json)
for startup failures and successful probes. SGLang's original BF16 probe is
**not relabeled as FP16**: the separate explicit-convolution FP16 run passes
**48/48 measured requests over 12 trials**, including six added longer-prefill
requests. Both variants have active-request limit 2, so offered C8 includes server
queueing. Warp-34's built-in renderer is not the artifact's Jinja executor.
These are operational probes, not token-aligned or quality-qualified rankings.

## Configuration and extension

Copy `suite.toml` alongside its `fixtures/` directory (or adjust relative fixture
paths). Add an `[[engines]]` table to support another OpenAI-compatible endpoint;
no Python change is needed. `base_url` includes `/v1`. For a different API, add a
small transport adapter and the same HTTP contract tests; do not fork metrics.
Profiles can point to containers or another host. Local GPU metadata then
identifies the **client**, not the remote server: attach server-side provenance
and explicitly label remote/network latency in your release audit.

Required **runnable** deployment fields: `id`, `engine`, `version`, `model`, `base_url`,
`gpu_count` (1 or 2), `parallelism`, `weight_format`, `kv_format`, `speculation`,
`cache_config`, and `launch_command`. Shared fields go in `[engine_defaults]`.
Before release, also fill `gpu_ids` (physical UUIDs), `source_model_revision`,
`model_revision`, `artifact_sha256` (an inventory hash for sharded weights),
`tokenizer_revision`, `chat_template_sha256`, and `quantization_details`.
`unknown` is allowed for exploration but becomes a publication blocker.
The legacy `quality_evidence` text field is a manual audit note only: attach
functional-quality JSON with `--evidence` to affect the automated quality gate.
Only attachments matching the exact profile digest count; any matching failure
wins over matching passes.
**Metadata is operator-supplied, not proof that the server used that setting.**

`request_overrides` holds engine-specific sampling/template options; e.g.
`chat_template_kwargs.enable_thinking = false`. Workload, model, streaming,
choice count, and usage-request fields cannot be overridden. For authentication,
set `api_key_env = "BENCH_API_KEY"`; only the environment-variable name is saved,
not the resolved key. Do not put secrets in URLs, commands, prompts, or config.
There are no implicit retries: a 429, OOM, timeout, bad stream, or rejection stays
in the trial. The client does not consult proxy environment variables.

Cases support:

- `requests`: text requests or **agent sessions**; `concurrency`: outstanding
  closed-loop requests/sessions. Agent turns are sequential within each session.
- `arrival = "closed"` (default), `"fixed"`, or `"poisson"`. Open-loop
  `rate_rps` is offered **sessions/s** (requests/s for text). A fixed-seed
  schedule is generated independently of completions. `run.max_inflight` caps
  simultaneous sessions, not server admission; exhausted client slots are
  recorded as `client_overload`, never hidden in an executor queue.
- `max_tokens` is a ceiling, **not forced output length**. Natural EOS is kept.
  `context_repetitions` scales fixture context; it is not a token-count promise.
  Actual server usage is retained. Calibrate pinned token/context buckets before
  publishing comparisons at exact 8K/32K/128K/262K lengths.
- `mix` cycles deterministic `{context_repetitions, max_tokens}` shapes for
  mixed prefill/decode traffic. All payloads are saved; shape-specific slicing
  can be done from the raw journal.
- `cache = "unique"`: a different leading session tag before shared system
  text for each request, and a disjoint tag for warmup. This avoids full shared
  prefixes; it does not prove a completely cold cache or disable cache lookup.
- `cache = "primed"`: explicitly replay the entire identical input set before
  each trial. Priming artifacts are retained but excluded from measured totals.
  Capacity/eviction may prevent hits: primed means **attempted reuse**, not a
  verified 100% hit rate. Actual cache settings and counters require audit.
- `cache = "natural"`: shared system context and growing agent histories.
  Each repetition uses a new namespace. Engine JIT warmups use a disjoint one.
- Optional `ttft_slo_ms`, `e2e_slo_ms`, `tpot_slo_ms`. The legacy-style TTFT
  name is the threshold for **first observable output**, not a verified first
  generated token. Goodput is successful requests satisfying all supplied SLOs
  divided by the entire measured wall interval.

## Battery and agent replay

Included: smoke, single-client interactive, long generation, prefill-heavy,
long-context, concurrency 8/16, primed prefix reuse, Poisson arrivals, mixed
traffic, and coding-agent replay at 1/8/16 concurrent sessions. Duplicate cases
to sweep concurrency, request rate, context, and output ceilings. Settings are
explicit data, not special cases in engine implementations.

The original checked-in coding-agent fixture covers repository discovery,
issue triage, file reads/search, regression-test feedback, patch proposal, and
final review. Each four-turn session retains the **actual preceding generated
assistant responses**, appends recorded tool-call/result pairs, and waits a
configurable simulated tool/think interval between turns. Different sessions
can interleave. Tools never read files, run tests, execute commands, or call
external services. This isolates inference costs and repeatable prefix/growing-
context pressure. Unexpected live tool calls fail the replay session.

This is a **realistic agent-shaped replay**, not a SWE-bench score, autonomous
agent, or validated tool-selection/patch-success benchmark. Identical fixture
inputs do not guarantee identical later inputs: generated histories can differ.
The report must not claim token-aligned multi-turn workloads. Autonomous tool
loops and additional research/support fixtures are explicit follow-up tasks.

## Measurement contract

Every HTTP request uses a new connection. E2E and first-output latency include
connect/TLS overhead, request transmission, queueing, and transport. This is a
consistent client-observed serving baseline, not kernel decode speed. Connection
pooling and remote-client isolation are future lanes.

- **First output**: first nonempty content, reasoning, or tool-call delta;
  role-only, usage-only, and keepalive events are excluded.
- **First text**: first nonempty `content` delta, separately recorded.
- **E2E**: submission to `[DONE]`, including final usage. Finish reason and
  observable output are required; truncated/error SSE streams fail closed.
- **Completion tokens**: server-reported `usage.completion_tokens`, validated
  as a positive integer. Counts include whatever the engine reports (potentially
  reasoning and hidden terminal tokens). Full usage, including cache/reasoning
  detail where present, is saved. No output retokenization or chunk-count fallback.
  If any successful response lacks valid usage, trial token throughput is `null`.
- **tok/s**: successful accounted completion tokens over full trial wall time,
  including fill, drain, errors, agent think time, and client-side overhead.
  This is finite-run throughput, **not steady-state saturation capacity**.
- **TPOT estimate**: `(last output event - first output event)/(usage tokens-1)`.
  It is only a usage/span estimate, **not ITL**; buffered/speculative output,
  hidden tokens, and reasoning can bias it. Single-burst streams yield `null`.
- **Chunk gaps**: actual output-event gaps, labeled as chunks. No token-arrival
  histogram or GPU decode-latency claim is made from these.
- **Failures** and offered sessions remain visible. Failed partial outputs do
  not contribute successful throughput. Schedule lag reveals client limitations.
  Per-request hard deadlines also cover trickle streams. Response bytes are bounded.
- **Statistics**: p50/p90/p95/p99, mean, max, sample counts; median and min/max
  throughput across independent configured trials. Trials are never pooled into
  a misleading p99. Fewer than 100 successes warns against tail claims; meaningful
  p99 work needs much larger samples. No confidence interval is fabricated.

Agent session latency includes tool delays; per-turn latency does not. Session
throughput and request throughput are both saved. Session completion measures
transport/replay completion only, not answer quality. Early EOS, stop reasons,
and actual output-length distributions must accompany claims.

## Evidence and reproducibility

Each directory contains:

- `manifest.json`: resolved config, parsed fixture data and hashes, selected
  cases/deployments, suite source inventory/hash/version, Git revision/status,
  timestamps, Python/platform, local GPU UUID/model/driver/power limit/link and
  topology snapshots, GPU processes, completion/failure state.
- `requests.jsonl`: flushed append-only request/session records, exact wire
  payloads and hashes, every parsed SSE event with client timestamp, content,
  reasoning/tool deltas, usage, finish reason, timings, and errors. Warmup,
  priming, and measurement phases are separate. Interrupted runs retain evidence.
- `coverage.json`: deployment × selected-case availability, selection reasons,
  measurement status and completed/required repetitions, including skips.
- `summary.json`: per-trial distributions and publication blockers (written
  after each measured trial).
- `SHA256SUMS.json`: finalized artifact-file hashes. Derived Markdown/CSV reports
  go to stdout and don't mutate signed/raw evidence.

Runs stay under ignored `results/` by default. Raw prompts, tool results, and
model responses may be sensitive; **use only public/sanitized fixtures** for
shareable runs. Do not blindly publish arbitrary real agent traces.

Before sharing, attach exact server executable/container digest, full launch
log/config, immutable weight-file inventory and byte size, source-model mapping,
quality evidence, hardware/OS/driver/runtime details, cache/speculation settings,
GPU clocks/temperature/power conditions, and all failure/unsupported rows. Local
snapshots are not sampled peak VRAM or energy measurements. There is deliberately
no automatic speedup/winner ranking across unqualified deployments.

## Testing

```sh
python3 -m unittest discover -s tests -v
python3 -m w34perf validate suite.toml
# From repo root: mise run perf-check (also included in mise run check).
```

Tests use pure calculations and real local HTTP/SSE servers. No engine, GPU,
or oracle fixtures are needed. See `PLAN.md` for remaining release gates.
