# warp-34

Clean-room Rust + CUDA inference engine for **Qwen3.5 hybrid models** on
RTX 3090 Ti (`sm_86`, 24 GB, 1008 GB/s), architected around **`vLLM`** from day
one.

## Core Design Principles

1. **Explicit Reference Donor per Subsystem ([`docs/ROADMAP.md`](docs/ROADMAP.md)):**
   - **`vLLM 0.31.0` (`8352b2427`)**: Primary donor for the `FP16` model spine, Marlin GEMMs (`W4A16` / `W8A16` / `W4A8` / `moe_wna16`), layer fusions (`GemmaRMSNorm`, `fused_qk_rmsnorm_rope_gate`, GDN `chunk` + `fused_recurrent`), native `MTP`, 9-step GPU sampler, and Qwen3 reasoning/tool/vision parsers.
   - **`SGLang v0.5.19` (`0bcd822377`)**: Primary donor for **Hybrid Prefix Caching (`MambaRadixCache` + `HybridReqToTokenPool` `extra_buffer` strategy)** and **`DFlash2` (`DFlashWorkerV2` `M=8` speculative verify)** — the two areas where `vLLM 0.31.0` cannot combine hybrid prefix caching with draft models or breaks on `TP=2` (`tau=1.000`).
   - **`warp3090` (`sm_86` proven IP)**: Primary donor for **TurboQuant 4-bit KV (`TQ4`)** (`block_tq4_0` 4.125 bpw, Rademacher+WHT-256 rotation, Beta(256) Lloyd-Max codebook, `FirstLast2` `FP16` boundary layers, fused `fattn_mma_tq` + GQA-8 `fattn_vec_tq`) and **Pinned-Sysmem PCIe Gen4 x8 `TP=2` AllReduce** (`~18 µs`).
2. **End-to-end `f16` activation spine:** Hidden states, residuals, norms, RoPE,
   attention, GDN, and Marlin GEMMs consume and produce `f16` directly — zero
   `f32 ↔ f16` adapter casts between ops.
3. **Continuous Oracle & `nsys` Scoreboard from Phase 0:** Every op and phase is
   gated by `w34-oracle` (`dump-diff` tensor parity, `greedy-diff` token parity,
   and `profile-diff` side-by-side `nsys` kernel timing against `vLLM` / `SGLang`).

## Workspace Layout

```
crates/
  w34-kernels/   CUDA kernels + C/Rust FFI seam (built incrementally via nvcc)
  w34-host/      Rust runtime, safetensors loader, scheduler, CLI, and server
dashboard/      React + TypeScript + Vite frontend (pnpm; embedded in w34)
docs/
  ROADMAP.md     Phased engineering roadmap & vLLM / SGLang / warp3090 source map
```

## Quick Commands

```sh
export PATH="/opt/cuda/bin:$PATH"
mise install            # pinned Rust, Python, Node, and pnpm tooling
mise run build          # build dashboard, then cargo build --release --features cuda
mise run check-quick    # fmt --check + clippy (with and without cuda)
mise run check          # fmt + clippy + release test suite
```

## Models: download on first use

```sh
./target/release/w34 serve --model nchapman/Qwen3.5-4B-W4A16-GPTQ
# Pin a revision, or require an already complete local cache:
./target/release/w34 serve --model username/modelname --revision <commit-sha>
./target/release/w34 serve --model nchapman/Qwen3.5-4B-W4A16-GPTQ --offline
```

Hub IDs automatically download the required checkpoint assets using Hugging
Face's native Rust client, then reuse the standard Hub cache. No Python or `hf`
CLI is required. Local model-directory paths still work without network access.
`HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`, saved `hf auth login` credentials, and
`HF_HUB_OFFLINE` are supported. DFlash2 has an independent `--draft-revision`.
Only supported Safetensors architectures are loaded; repository Python code is
never executed. See [model downloads, authentication, and offline behavior](docs/MODELS.md).

## Serving controls

Authentication is **optional**; local use remains unauthenticated by default.
For a deployment, export your key through a secret manager, then use:

```sh
./target/release/w34 serve --model nchapman/Qwen3.5-4B-W4A16-GPTQ \
  --api-key-env W34_API_KEY --served-model-name coding-model \
  --max-num-seqs 4 --kv-pages 512 --max-model-len 8192
```

`--api-key VALUE` is also supported, but an environment variable avoids putting
secrets in process arguments. API clients use Bearer authorization or `x-api-key`.
`--max-num-seqs` (1–16, default 16) controls active concurrency and GDN state planning;
`--max-model-len` optionally caps prompt + output tokens without changing KV allocation.
See [serving controls, authentication scope, and limits](docs/SERVING.md).

## Live dashboard and metrics

Open **`http://127.0.0.1:8000/`** while `w34 serve` is running for the built-in,
read-only dashboard: live generation throughput, engine TTFT, request load, prefix
reuse, and separate KV/GDN capacity. A warm-dark React interface is embedded in the
binary—no Node runtime, external assets, or Prometheus installation required.

**`/metrics`** exposes Prometheus counters, gauges, and latency histograms;
**`/api/metrics`** exposes the same measurements as versioned JSON. The dashboard
keeps five minutes of local history; use an external collector for persistence.
See [metrics semantics, deployment security, and development commands](docs/METRICS.md).
Dashboard, metrics, and health remain public even with API authentication enabled:
keep the listener private or protect the entire service with an authenticated TLS
reverse proxy.

For frontend development, `mise run dashboard-dev` runs Vite with a proxy to the
server on port 8000. `mise run dashboard-check` runs frontend checks and tests.
Direct Cargo builds with CUDA need `mise run dashboard-build` first; `mise run build`
and the standard check/test tasks do this automatically.

## Standalone performance comparisons

[`benchmarks/perf/`](benchmarks/perf/README.md) is a separate, engine-neutral
HTTP performance suite for warp-34, vLLM, SGLang, llama.cpp, and ninfer-3090.
It uses approximately equivalent **4-bit weights**, treats KV precision as a
separate axis, and includes single-/dual-card deployment profiles, serving load,
prefix reuse, mixed traffic, and multi-turn coding-agent replay. It does not
import development oracles. See its [plan and task ledger](benchmarks/perf/PLAN.md)
for remaining artifact/engine qualification and publication gates.

```sh
mise run perf-check  # CPU-only suite tests + experiment validation
cd benchmarks/perf
python3 -m w34perf plan suite.toml
# Start the chosen server separately, then:
python3 -m w34perf run suite.toml --engine warp34-1-fp16 --case smoke \
  --out results/warp34-smoke-001
```

## Local Docker

Requires an x86-64 host with AVX2 (`x86-64-v3`), RTX 3090 Ti GPU(s), a
CUDA 13.4-compatible NVIDIA driver, and NVIDIA Container Toolkit configured
for Docker. Building needs no GPU; running inference requires `--gpus`.

```sh
docker build -t warp-34:local .
# Optional on larger build hosts: --build-arg W34_NVCC_JOBS=16
docker run --rm warp-34:local  # print CLI usage
```

The multi-stage build compiles the locked React/Vite dashboard with the same Node
and pnpm versions pinned in mise, then uses pinned CUDA 13.4.1 + Rust 1.99.0 and copies only
`w34`, `libcudart`, `libcublas`, and `libcublasLt` into Ubuntu 24.04. No toolkit,
Python, model weights, or NVIDIA driver is bundled in the runtime image.
The locally verified image reports approximately **1.22 GB** via Docker image
inspection (size can vary with base-image/package updates).
BuildKit caches Cargo downloads and compilation across builds. This is a local
image only; there is no CI or registry publishing setup.

### Serve a cached model

For an offline deployment, mount the whole Hub cache (including `blobs`, `refs`,
and `snapshots`, so relative symlinks resolve), read-only, and pass `--offline`:

```sh
docker run --rm --name w34-local --gpus '"device=0"' \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src="$HOME/.cache/huggingface/hub",dst=/home/w34/.cache/huggingface/hub,readonly \
  warp-34:local serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ --offline --host 0.0.0.0
```

In another terminal:

```sh
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nchapman/Qwen3.5-4B-W4A16-GPTQ","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
docker stop w34-local
```

The process runs as a non-root user; mounted files must be readable and their
directories traversable. If your cache is private to your host user, add
`--user "$(id -u):$(id -g)"` before the image name; `HF_HUB_CACHE` still points
to the mounted path. To use an unpacked model directory instead, mount it at
`/models/model` and pass `--model /models/model`.

For download-on-first-use, remove `readonly` from the cache mount and omit
`--offline`. Ensure the container user can write the cache (use the host UID/GID
as above when appropriate). For private/gated repositories, pass `-e HF_TOKEN`
before the image name to forward an already-exported token; do not put its value
in command-line arguments. Online startup checks Hub metadata even for cached
weights. A read-only cache therefore requires `--offline` or a direct local path.

For two GPUs, replace `--gpus '"device=0"'` with `--gpus '"device=0,1"'` and add
`--tp 2`. Container-visible device indices are local to the selected GPUs.
The engine uses pinned host memory for TP=2; no NCCL or `--ipc=host` is needed.
For speculative decoding, mount the draft model in the same cache and append
`--speculative dflash2 --draft-model <cached-repo-id>` (or `--speculative mtp`).

Arguments after the image name go directly to `w34`, so `generate`, `bench`,
and other native commands work too. Use `--no-healthcheck` before the image
name for non-server commands. The server must bind `--host 0.0.0.0` inside the
container; the localhost-only port publication above keeps it local to the
host. The health check probes HTTP responsiveness via `/health`, not inference
worker liveness; it is not an end-to-end generation check.
The health check expects internal port 8000: change only the host side
of `-p` to use another host port, or override `--health-cmd` if changing
`serve --port`. Oracle profiling commands additionally need the development
checkout and external Python/`nsys` tooling, which are not in this image.
