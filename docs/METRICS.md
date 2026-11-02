# Metrics and built-in dashboard

`w34 serve` includes a read-only, warm-dark React dashboard at **`GET /`**.
No external collector, CDN, Node runtime, or frontend files are needed at runtime:
Vite's production assets are embedded in the Rust executable. Prometheus remains
optional; use it for durable history, alerting, and fleet-wide aggregation.

## Interfaces

| Route | Contract |
|---|---|
| `/` | Embedded dashboard; HTML revalidates on navigation |
| `/assets/*` | Content-hashed, immutable production assets |
| `/metrics` | Prometheus text exposition 0.0.4, with HELP/TYPE metadata |
| `/api/metrics` | Versioned JSON snapshot (`schema_version: 1`), `Cache-Control: no-store` |
| `/health` | Existing worker-health endpoint, independent of dashboard polling |

Prometheus and JSON are projections of **one shared telemetry snapshot**. Collection
reads CPU telemetry only: no CUDA synchronization, allocator traversal from the HTTP
thread, or engine request round-trip. Resource occupancy is published by the engine
at tick boundaries. No prompt/response content, request IDs, user labels, or credentials
are included. The configured model name is operational metadata; if you use a local
path as the served model name, that name is visible too.

**Exposure:** the server does not enforce API-key authentication. Bind to loopback,
use a trusted private network, or protect the entire service with an authenticated
reverse proxy. Authorization headers used to scope stored Responses are not access
control. The dashboard adds no generation, benchmarking, or administrative actions.
Its production CSP restricts scripts and connections to the same origin, and assets
are served from an embedded allowlist, never the filesystem.

## Metric meanings

All counters reset with a process restart. Time measurements are monotonic seconds.
Model configuration labels appear only on `w34_model_info`; labels on other families
have fixed, bounded values. Use Prometheus target labels to distinguish servers.

### Work and outcomes

- `w34_requests_total`: requests admitted to the worker ingress queue, across all
  inference APIs. A decision API call can create more than one engine request.
- `w34_active_requests` / `w34_waiting_requests`: scheduled unfinished requests vs
  ingress plus scheduler waiting. These are actual lifecycle gauges, not estimates
  from peak load.
- `w34_completed_requests_total`, `w34_cancelled_requests_total`,
  `w34_failed_requests_total`: mutually exclusive terminal outcomes. Cancellation
  includes disconnected/slow consumers; failures include worker shutdown/panic.
- `w34_http_responses_total{class="success|client_error|server_error"}`: inference
  HTTP response **headers**, including validation/admission errors before worker
  submission. A successful SSE header is not proof that generation later succeeded;
  consult worker outcomes too. Scrapes, dashboard assets, status, and token-count
  routes do not increase this counter.

### Tokens and reuse

- `w34_prompt_tokens_total`: prompt tokens submitted to the worker.
- `w34_computed_prompt_tokens_total`: prompt tokens in successfully executed prefill
  chunks. This reflects work, not merely the declared input length.
- `w34_cached_prompt_tokens_total`: prompt tokens actually restored on admission.
- `w34_tokens_generated_total`: committed model output, including stop tokens,
  whether or not text reaches the client. Draft proposals are not counted here.
- `w34_radix_cache_hit_rate`: lifetime restored/submitted prompt-token ratio.
  Prefer rates of the underlying counters for a rolling reuse ratio. It is not a
  request hit rate. Pending/cancelled prompts mean computed + cached need not equal
  submitted at an arbitrary snapshot.

### Latency histograms

Each family exposes `_bucket{le=...}`, `_sum`, and `_count`, including the `+Inf`
bucket. Histograms measure **engine latency**, not HTTP client-perceived latency:
request parsing, tokenization, image preprocessing, network transport, and client
backpressure after output publication are not part of TTFT.

| Family | Observation |
|---|---|
| `w34_queue_seconds` | Enqueue → first scheduler admission |
| `w34_ttft_seconds` | Enqueue → first committed model token; prefill-only requests excluded |
| `w34_e2e_seconds` | Enqueue → terminal outcome, including cancellation/failure |
| `w34_time_per_output_token_seconds` | Per-request `(last token time - first token time) / (generated tokens - 1)`, when at least two tokens exist |

Do not interpret per-output-token time as an observed interval between every SSE
message. Speculation emits multiple tokens together; measuring rapid channel drains
would report misleading near-zero inter-token latency. Request-average pacing is
explicitly named instead. It is not pure GPU kernel time either.

### Resource capacity

`w34_pool_units{pool="kv|gdn",state="capacity|active|cached|free"}`:

- KV units are logical **pages** (16 tokens/page); GDN units are **state slots**.
- Counted once across tensor-parallel ranks, not summed as if each rank served
  independent requests. These are target pools, not total device VRAM usage.
- `active` includes protected and reserved allocations.
- `cached` means evictable radix-cache resources; these can be reclaimed.
- `free` means immediately unallocated resources.
- Unknown values are omitted in Prometheus and represented by `null` in JSON.
  In particular, failed allocator accounting must not be disguised as zero pressure.

High cached occupancy is useful reuse, not automatically saturation. GDN state
pressure can block new requests even while KV capacity remains available.

### Speculation and engine information

- `w34_speculation_rounds_total`, `w34_speculation_accepted_tokens_total`: successful
  per-request verification rounds and accepted drafts (excluding the bonus token).
- `w34_speculation_draft_tokens_total`: actual draft rows included in successful
  verification, after budget, stop, constraint, and capacity trimming. This is not
  the configured maximum or discarded proposal work. JSON permits `null` for an
  unavailable denominator; the shipped MTP/DFlash paths publish the actual count.
- `1 + accepted / rounds` is mean acceptance length (accepted drafts plus the normal
  bonus/anchor contribution), not a claim about wire-delivered tokens after stops.
- `w34_model_info{model,tp_size,kv_format,speculation} 1`, `w34_uptime_seconds`,
  and `w34_status{state="ready|unhealthy|stopping"}` provide bounded operational context.

## Dashboard behavior

The dashboard polls JSON every two seconds with one in-flight request and a five-second
timeout. Hidden tabs pause polling. It keeps at most 151 observations / five minutes
in memory; navigation, reload, or a new `instance_id` clears history. It does not persist
or backfill data. Charts break across observation gaps rather than inventing continuity.

- Throughput is aggregate output tokens/sec over the trailing ten observed seconds.
- TTFT and queue percentiles are approximate, derived by interpolating histogram
  bucket deltas within the observed window. Empty intervals show `—`, not zero latency.
  Overflow quantiles are shown as lower bounds (`≥`).
- Prefix reuse in the dashboard is `cached / (cached + computed)` over observed
  deltas, avoiding a misleading denominator of requests still waiting for prefill.
- The TTFT trend uses a shorter trailing observation window; summary percentiles use
  the full retained window. Pool/request gauges are current snapshots; outcome/token
  totals and speculative summaries are since startup.
- Disconnections keep the last snapshot visibly marked stale, suppress the live
  throughput number, and retry automatically. Worker health is distinct from whether
  the metrics HTTP request succeeded.
- The Connect page provides same-origin API and scrape setup information. No speed
  test is started automatically, and no UI controls mutate engine state.

## Prometheus examples

```yaml
scrape_configs:
  - job_name: warp34
    scrape_interval: 10s
    static_configs:
      - targets: ['127.0.0.1:8000']
```

```promql
# Aggregate generated output per second
rate(w34_tokens_generated_total[1m])

# Engine TTFT p95
histogram_quantile(0.95, sum by (le, instance) (rate(w34_ttft_seconds_bucket[5m])))

# Token-weighted prefix reuse; no value when there are no prompt observations
rate(w34_cached_prompt_tokens_total[5m])
  / (rate(w34_cached_prompt_tokens_total[5m]) + rate(w34_computed_prompt_tokens_total[5m]))

# Active target KV occupancy (evictable cache is deliberately not pressure)
w34_pool_units{pool="kv",state="active"}
  / ignoring(state) w34_pool_units{pool="kv",state="capacity"}
```

### Migration from the initial endpoint

The original endpoint's request/token/cancellation names are retained where useful,
but now count the common engine lifecycle rather than selected HTTP consumers.
`w34_last_ttft_ms` and `w34_last_itl_ms` are removed in favor of seconds histograms.
The legacy `w34_peak_active_requests` and internal `w34_decode_bucket_mask` remain
as compatibility diagnostics; prefer current concurrency gauges and external
historical maxima for operational dashboards. Do not combine
old and new token-accounting definitions in a single historical comparison without
marking the deployment boundary.

## Development and build

Node and pnpm are pinned in `.mise.toml`; frontend package versions and transitive
resolution are pinned by `dashboard/package.json` and `dashboard/pnpm-lock.yaml`.

```sh
mise install
mise run dashboard-dev    # Vite on localhost:5173, API proxy to localhost:8000
mise run dashboard-check  # TypeScript, Prettier, Vitest / React Testing Library
mise run build            # locked install + Vite build + CUDA Rust binary
mise run check            # frontend checks plus existing CPU/CUDA/oracle/perf checks
```

For direct Cargo commands, first run `mise run dashboard-build`. The embedded
`dashboard/dist` is generated and ignored by Git, not a second hand-maintained source.
Docker builds the same locked frontend in a separate Node stage and copies only
`dist` into the Rust build; the runtime image contains no frontend toolchain.

## Donor requirements

The first version follows vLLM 0.31.0's counter/histogram approach
(`v1/metrics/{stats,loggers,buckets}.py`) and SGLang v0.5.19's separation of active,
evictable, and free hybrid pools (`srt/observability/metrics_collector.py` and
`scheduler_components/pool_stats_observer.py`). We adopt their operational semantics,
not their entire metric inventory or Python implementation. High-cardinality request
tracing, device utilization polling, persistent chart storage, and benchmark controls
are intentionally outside this initial dashboard.
