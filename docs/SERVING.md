# Serving deployment controls

`warp34 serve` targets one Qwen model on one or two GPUs, not a general-purpose model
gateway. Defaults remain loopback (`127.0.0.1:8000`), **no authentication**, 16 active
sequences, automatic KV capacity, and the existing scheduler tuning. The Web UI
is enabled by default; `--no-webui` disables its pages and assets without changing
API or metrics access.

```sh
# Export WARP34_API_KEY securely through your deployment's secret manager first.
CUDA_VISIBLE_DEVICES=0 ./target/release/warp34 serve \
  nchapman/Qwen3.5-4B-W4A16-HGPTQ --offline \
  --alias coding-model \
  --api-key-env WARP34_API_KEY \
  --max-num-seqs 4 -c 8k --max-model-len 8k

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $WARP34_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"coding-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Optional API authentication

- `--api-key VALUE` enables one shared deployment key. Prefer
  **`--api-key-env NAME`**, which reads the named environment variable once at
  startup instead of exposing the secret in the server's process arguments.
  These flags are mutually exclusive. There is no implicit environment fallback.
  Missing, non-Unicode, empty, whitespace-containing, or non-visible-ASCII keys
  fail before model acquisition. Errors never print the supplied secret.
- Omit both flags to retain unauthenticated local use. `HF_TOKEN` is a separate
  Hub download credential, never a serving key.
- Clients can send `Authorization: Bearer <key>` (case-insensitive scheme) or
  `x-api-key: <key>`, including Anthropic clients. If Authorization is present it
  takes precedence; invalid Authorization does **not** fall back to x-api-key.
  Duplicate credential headers are rejected. Query-string credentials are ignored.
- When a key is configured, all inference, model-list, tokenization, token-count,
  and stored-Responses routes require authentication, including unversioned aliases (`/decisions`,
  `/systemone`, `/responses/input_tokens`, `/chat/completions/input_tokens`).
  Authentication runs before body processing, remote image fetching, or inference.
  Failures return HTTP 401 with an OpenAI error envelope, or an Anthropic envelope
  on `/v1/messages` and `/v1/messages/count_tokens`.
- **Enabled Web UI pages/assets, `/health`, `/metrics`, and `/api/metrics`
  remain unauthenticated**, so monitoring and the dashboard continue to work.
  Overview and Connect are read-only; Playground API requests require the same
  configured key as any other client.
  They expose operational metadata, including the served model name and load.
  Protect the **entire listener with a firewall or authenticated reverse proxy**
  if that information is private. The API key is not TLS, rate limiting, or
  multi-tenant isolation. Use TLS termination for traffic outside a trusted host.
- Keys are stored as SHA-256 digests and compared in constant time. Credentials
  are not rewritten: stored Responses retain their existing per-credential scope.
  All clients using the same credential scope share that scope; this is not a
  per-user authorization system. Restart to rotate the key; stored Responses are
  in-memory and do not survive restart.

For Docker, pass `-e WARP34_API_KEY` before the image name and
`--api-key-env WARP34_API_KEY` after `serve` to forward an already-exported secret.

## Listener and Web UI exposure

Network reachability controls access to observability and the Web UI; the optional
API key controls API operations. Binding is not per-user authentication.

| Surface | Access policy |
|---|---|
| Web UI: `/`, `/connect`, `/playground`, nested Playground paths, favicon, and `/assets/*` | Unauthenticated when enabled; HTTP 404 with `--no-webui` |
| Metrics: `/metrics` and `/api/metrics` | Always unauthenticated, including with API authentication or `--no-webui` |
| Health: `/health` | Always unauthenticated, minimal availability status only |
| Inference, models, tokenization/counts, decisions, stored Responses, and aliases | Require the configured API key; unauthenticated if none is configured |
| Stored Responses retrieval, deletion, cancellation, and chaining | Also enforce existing credential-scope ownership; a response ID alone grants no access |

Keep the default `--host 127.0.0.1` for local access (`WARP34_HOST` sets the fallback bind address when `--host` is omitted; the official Docker image sets `WARP34_HOST=0.0.0.0` so `-p 127.0.0.1:8000:8000` works out of the box). To expose the service, select
an explicit interface with `--host`; a Tailscale address limits the listener to
that interface but still requires appropriate tailnet access rules. `--host
0.0.0.0` listens on all IPv4 interfaces, subject to firewall rules. Use a firewall,
Tailscale ACLs/grants, or an authenticated HTTPS reverse proxy to restrict the
whole service. Prevent direct backend access when a proxy supplies access control.

Startup reports API authentication and Web UI status without printing secrets.
A non-loopback listener emits an exposure warning **even when an API key is set**:
health, metrics, and enabled Web UI pages are still unauthenticated.

```sh
# API-only presentation, still with public metrics/health on the selected interface:
warp34 serve MODEL --host 127.0.0.1 --no-webui --api-key-env WARP34_API_KEY
```

`--no-webui` is a serve-only boolean flag; it takes no value. It removes UI routes,
not the embedded bundle from the binary, and does not hide telemetry or change
API authentication. It is not a substitute for access control. There is no built-in
browser account/login system; use a proxy if the entire dashboard needs a login.

## Model identity

`--served-model-name NAME` sets the public model identity in `/v1/models`,
generative responses, and metrics. It does not change the download repository,
revision, tokenizer, or loaded weights. Without an alias the original `--model`
ID or local path is retained, not the resolved Hub cache path. A public alias also
avoids exposing local filesystem paths in telemetry.

This is still a single-model server: request `model` is not a routing selector,
and adding an alias does not introduce strict model-name validation. Decision
adapter responses retain their existing wire format. Configure clients to use
the advertised alias rather than relying on permissive request-name handling.

## Concurrency and context

| Option | Default | Meaning |
|---|---|---|
| `--max-num-seqs N` (aliases: `--max-running-requests`, `--parallel`, `-np`) | 16 | Active sequences, **1–16**; used for both GDN state-pool planning and worker scheduling |
| `-c N`, `--ctx-size N` (aliases: `--context-length`, `--kv-tokens`) | Automatic (up to `32k` tokens on `serve`) | Shared target KV pool in **tokens** (supports `k`/`M` suffixes like `8k`, `32k`, `64k`, `128k`; rounds up to 16-token pages) |
| `--max-model-len N` | `min(allocated_kv_tokens, max_position_embeddings)` | Per-request total prompt + requested output admission ceiling (supports `k`/`M` suffixes; auto-expands target KV pool when `-c`/`--kv-pages` is omitted) |
| `--kv-pages N` (alias: `--num-gpu-blocks-override`) | Automatic (up to `2,048` pages on `serve`, including `--speculative`) | Low-level alternative to `--ctx-size`: shared target KV pool in 16-token pages (mutually exclusive with `--ctx-size`) |

`--max-num-seqs` is an upper bound, not a promise that N arbitrary contexts fit at
once. GPU memory may reduce state capacity, and KV availability can serialize or
queue requests. The ingress/admission bounds remain 32; this flag is not an HTTP
connection or queue-size control. Larger than 16 requires engine graph/IO changes,
not merely a CLI override. Lower concurrency can reduce planned GDN state memory.

`--ctx-size` (`-c`) sizes the shared GPU KV pool directly in tokens (so you never
have to divide by 16-token pages), accepting human-friendly suffixes (`4k`, `16k`,
`32k`, `64k`, `128k`, `256k`).

`--max-model-len` sets the per-request **admission ceiling** (in tokens, also
supporting `k`/`M` suffixes). When omitted, `warp34` defaults `max_model_len` to
`min(allocated_kv_tokens, max_position_embeddings)` when the checkpoint declares
`max_position_embeddings` (logging a warning if `--max-model-len` is explicitly
set above `max_position_embeddings`), and advertises `max_model_len` on every
`GET /v1/models` entry. When neither `--ctx-size` nor `--kv-pages` is
explicitly passed, `--max-model-len` automatically expands the target KV page
request when needed (for example, `warp34 serve Qwen/Qwen3.5-4B --max-model-len 64k`
or `warp34 serve Qwen/Qwen3.5-4B --speculative mtp --max-model-len 16k`
automatically requests `4,096` or `1,024` KV pages). When `--ctx-size` or
`--kv-pages` *is* explicitly passed alongside `--max-model-len`, `warp34` validates
`max_model_len <= kv_tokens` immediately at CLI parse time before any Hub download
or GPU initialization.

The limit checks the fully tokenized prompt (including expanded image tokens)
before worker enqueue across all generative APIs (streaming and non-streaming)
and decision-model prefills (zero output budget):
- Prompts with `prompt_tokens >= max_model_len` (or `prompt_tokens > max_model_len`
  for zero-output decision prefills) fail immediately with HTTP 400 (`code:
  "context_length_exceeded"` on OpenAI Chat/Completions and Responses; `"prompt is
  too long: X tokens > Y maximum"` on Anthropic `/v1/messages`) with no silent
  prompt truncation.
- Token-count/tokenization utilities (`/v1/messages/count_tokens`,
  `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/tokenize`)
  are exempt so coding agents can count oversized histories before compacting.
- When `prompt_tokens < max_model_len`, an explicit `max_completion_tokens` /
  `max_tokens` / `max_output_tokens` above the remaining window (`max_model_len -
  prompt_tokens`) is clamped to `max_model_len - prompt_tokens` rather than
  rejecting static client output budgets (such as Claude Code `32000` or OpenCode
  `8192`) mid-session.
- When `max_completion_tokens` / `max_tokens` / `max_output_tokens` is omitted on
  `/v1/chat/completions` or `/v1/responses`, it defaults to the remaining context
  budget (`max_model_len - prompt_tokens`); legacy `/v1/completions` defaults omitted
  `max_tokens` to `min(256, max_model_len - prompt_tokens)`.
- Physical KV pages are allocated incrementally (`128`-token / `8`-page decode
  slabs) rather than pre-allocating a request's entire `max_tokens` ceiling up front,
  preserving warm `MambaRadixCache` prefixes and gracefully finishing with `"length"`
  (`"max_tokens"` on Anthropic, `"incomplete"` on Responses) if the shared KV pool
  exhausts mid-decode.

## Readiness, shutdown, and bounded recovery

`GET /health` returns 200 while the worker is available, 503 with `status: "stopping"`
when HTTP admission closes or worker shutdown is requested, and 503 with
`status: "error"` after a worker panic. Error status takes precedence over stopping;
internal panic diagnostics remain in server logs, never in the public health body. This is **not a proof of GPU forward progress**: a stalled kernel or
collective may leave HTTP responsive. Monitor completion progress/latency and
use a bounded real inference probe appropriate to the deployment workload.

Graceful shutdown closes admission, allows up to **30 seconds for HTTP drain**,
then requests worker shutdown and joins its thread. The join deliberately waits
for safe CUDA/CPU-bridge teardown; the 30-second drain deadline is **not a total
process-exit deadline**. Timing out a blocking thread join cannot safely cancel
CUDA work or permit freeing in-flight mapped buffers.

Configure an external hard stop budget, for example Docker `--stop-timeout 45`
(or `docker stop --time 45 <container>`), or systemd `TimeoutStopSec=45s` with
`SendSIGKILL=yes`. Choose a budget compatible with your request/drain policy.
Docker health checks alone do not restart an unhealthy container; arrange the
watchdog/orchestrator policy explicitly. A manually stopped Docker container
also requires an explicit start rather than relying on its restart policy.

Before release, rehearse on an **isolated disposable instance**:

1. Send an inference request, initiate shutdown, and verify readiness becomes
   unavailable and the normal drain completes.
2. Exercise the supervisor's hard-stop path with a deliberately nonterminating
   test process or a controlled stalled instance; verify it is killed within the
   configured budget and GPU/process resources are gone before replacement.
3. Restart the pinned image/model/configuration, serve a real request, and test
   rollback to the last known-good combination. Record actual timings and status.

Do not inject GPU/rank failure into a shared production worker. This is an operator
qualification procedure, not a claim that the repository configures a supervisor
or that the hard-stop rehearsal has already passed. In-memory Responses history
is lost on restart; a partially delivered stream cannot be transparently retried.

## CLI scope and intentional omissions

The deployment settings (`--api-key`, `--api-key-env`, `--no-webui`,
`--served-model-name`, `--max-num-seqs`, `--max-model-len`) are **serve-only**.
`serve` rejects generation, benchmarking, and oracle-only options it formerly
accepted without effect. In particular, `--batch-size` is a benchmark option;
use `--max-num-seqs` for serving. `--no-thinking` is a chat option; serving callers
control thinking in their requests.

We deliberately retain the fixed FP16 target spine, automatic checkpoint
quantization detection, SDK-owned Hub credentials/cache, built-in Qwen parsers,
and validated prefill/mixed-batch tuning. No arbitrary dtype/backend selection,
remote model code, global sampler seed override, `n>1` sampling, or scheduler tuning
flags are added. TLS/CORS remain reverse-proxy responsibilities. Request logging,
watchdogs, and graceful-drain policy are separate future operational work—not
features a reverse proxy can completely replace.
