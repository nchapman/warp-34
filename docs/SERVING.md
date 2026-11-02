# Serving deployment controls

`w34 serve` targets one Qwen model on one or two GPUs, not a general-purpose model
gateway. Defaults remain loopback (`127.0.0.1:8000`), **no authentication**, 16 active
sequences, automatic KV capacity, and the existing scheduler tuning.

```sh
# Export W34_API_KEY securely through your deployment's secret manager first.
CUDA_VISIBLE_DEVICES=0 ./target/release/w34 serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ --offline \
  --served-model-name coding-model \
  --api-key-env W34_API_KEY \
  --max-num-seqs 4 --kv-pages 512 --max-model-len 8192

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $W34_API_KEY" \
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
- All inference, model-list, tokenization, token-count, and stored-Responses
  routes require authentication, including unversioned aliases (`/decisions`,
  `/systemone`, `/responses/input_tokens`, `/chat/completions/input_tokens`).
  Authentication runs before body processing, remote image fetching, or inference.
  Failures return HTTP 401 with an OpenAI error envelope, or an Anthropic envelope
  on `/v1/messages` and `/v1/messages/count_tokens`.
- **The read-only dashboard/assets, `/health`, `/metrics`, and `/api/metrics`
  remain unauthenticated**, so monitoring and the dashboard continue to work.
  They expose operational metadata, including the served model name and load.
  Protect the **entire listener with a firewall or authenticated reverse proxy**
  if that information is private. The API key is not TLS, rate limiting, or
  multi-tenant isolation. Use TLS termination for traffic outside a trusted host.
- Keys are stored as SHA-256 digests and compared in constant time. Credentials
  are not rewritten: stored Responses retain their existing per-credential scope.
  All clients using the same credential scope share that scope; this is not a
  per-user authorization system. Restart to rotate the key; stored Responses are
  in-memory and do not survive restart.

For Docker, pass `-e W34_API_KEY` before the image name and
`--api-key-env W34_API_KEY` after `serve` to forward an already-exported secret.

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
| `--max-num-seqs N` | 16 | Active sequences, **1–16**; used for both GDN state-pool planning and worker scheduling |
| `--max-model-len N` | Allocated KV token capacity | Optional total prompt + requested output admission ceiling |
| `--kv-pages N` | Automatic; speculation defaults to 256 | Shared target KV pool, 16 tokens/page; allocates GPU memory |

`--max-num-seqs` is an upper bound, not a promise that N arbitrary contexts fit at
once. GPU memory may reduce state capacity, and KV availability can serialize or
queue requests. The ingress/admission bounds remain 32; this flag is not an HTTP
connection or queue-size control. Larger than 16 requires engine graph/IO changes,
not merely a CLI override. Lower concurrency can reduce planned GDN state memory.

`--max-model-len` is an **admission policy**, not automatic KV sizing or a context
extension. It must be positive and no greater than the allocated KV pool's token
capacity; an impossible setting fails startup after capacity is known. Select a
value within the checkpoint's supported context window. It does not infer or
change RoPE/model context metadata, nor reserve that many tokens per request.

The limit checks the fully tokenized prompt (including expanded image tokens)
plus requested output tokens before worker enqueue. It applies to all generative
APIs, streaming and non-streaming, and to decision-model prefills (zero output
budget). Over-limit requests fail with HTTP 400; there is no silent truncation.
Token-count/tokenization endpoints are utilities and do not consume this budget.
Absent output budgets still use each API's existing defaults, rather than silently
shrinking the output budget to remaining capacity.

## CLI scope and intentional omissions

The five deployment settings (`--api-key`, `--api-key-env`,
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
