# warp34

A single-binary Rust and CUDA inference engine built specifically for **Qwen 3.5, 3.6, 3.8, and Flash-Next** models on NVIDIA A100, RTX 30-series, and RTX 40-series GPUs.

[![Docker](https://github.com/nchapman/warp34/actions/workflows/docker.yml/badge.svg)](https://github.com/nchapman/warp34/actions/workflows/docker.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-sm__80%20%7C%20sm__86%20%7C%20sm__89-76b900.svg)](docs/HARDWARE.md)

No Python, no NCCL, and no runtime dependencies. Because `warp34` is purpose-built for Qwen's hybrid Gated DeltaNet + full-attention architecture, it delivers faster prefill, faster speculative decoding, and up to 262K context on consumer 24 GB GPUs—outperforming general-purpose engines like `vLLM` and `SGLang` on supported Qwen hybrid workloads.

## Highlights

- **Single binary, zero Python** — Handles Hugging Face downloads, OpenAI (`/v1/chat/completions`, `/v1/responses`, `/v1/completions`), Anthropic (`/v1/messages`), and Decision (`/v1/systemone`, `/v1/decisions`) APIs, a terminal `chat` REPL, and an embedded web dashboard (`/` & `/playground`).
- **Up to 262K context on 24 GB GPUs** — Built-in 4-bit TurboQuant KV cache (`tq4`, 3.07× compression) and hardware `fp8` KV (`sm_89+`) with automatic VRAM sizing and hybrid KV + recurrent-state prefix caching.
- **Fast prefill & lossless speculation** — `W4A8` prefill acceleration on RTX 30/40-series (zero extra weight VRAM) plus native `MTP` (`k=1..3`) and `DFlash2` block drafting—all compatible with prefix caching and multi-GPU.
- **Runs 180B `Qwen3.8-Flash-Next` on 1–2 consumer GPUs** — Memory-mapped n-gram tables, 3-tier hot/cold MoE offloading with an AVX-512/AVX2 CPU bridge, and automatic context-vs-speed sizing.
- **Multi-GPU without NCCL** — `--tp 2` and `--tp 4` automatically use NVLink P2P or 2-hop pinned PCIe transfers inside CUDA Graphs (no `--ipc=host` needed in Docker).
- **Structured outputs, Multi-LoRA, & Decision APIs** — `<think>`-aware XGrammar constraints (JSON Schema, Regex, CFG, Qwen XML tools), up to 62 hot-swappable LoRA adapters (`--lora-modules`), and single-prefill zero-decode classification/scoring.

---

## Quickstart

**Requirements:** Linux `x86_64` (AVX2/FMA), an NVIDIA A100 (`sm_80`), RTX 30-series (`sm_86`), or RTX 40-series (`sm_89`) GPU, and a CUDA 13-capable driver (R580+).

### Option A: Docker (fastest — no build tools or host CUDA toolkit needed)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```sh
docker run --rm --gpus '"device=0"' -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=warp34-models,dst=/home/warp34/.cache/huggingface/hub \
  ghcr.io/nchapman/warp34:dev \
  serve nchapman/Qwen3.5-4B-W4A16-HGPTQ --host 0.0.0.0
```

### Option B: Build from source

Requires CUDA 13.4 (`/usr/local/cuda` or `/opt/cuda`, or set `CUDA_HOME`) and [mise](https://mise.jdx.dev/) (installs pinned Rust, Node, and pnpm). Install [Git LFS](https://git-lfs.com/) only if you plan to run the oracle test suite:

```sh
git clone https://github.com/nchapman/warp34.git
cd warp34

mise install
mise run build   # tip: set WARP34_ARCH=sm_89 (or sm_86/sm_80) to compile only your GPU arch

./target/release/warp34 serve nchapman/Qwen3.5-4B-W4A16-HGPTQ
```

### Verify it works

`warp34 serve` downloads the checkpoint into `~/.cache/huggingface/hub` on first run, auto-sizes the KV cache to your GPU, and prints `Endpoint : http://127.0.0.1:8000` when ready:

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nchapman/Qwen3.5-4B-W4A16-HGPTQ",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

Open **`http://127.0.0.1:8000/`** in your browser for live telemetry, **`/playground`** for an interactive multi-API tester, and **`/connect`** for coding-agent setup snippets. Pass `--no-webui` for an API-only server.

---

## Common Recipes

> The examples below assume `warp34` is on your `PATH` (e.g., `export PATH="$PWD/target/release:$PATH"` after building from source, or replace `warp34` with `docker run --rm --gpus '"device=0"' ... ghcr.io/nchapman/warp34:dev`).

### Interactive terminal chat & benchmarking

```sh
# Multi-turn terminal REPL (omit --prompt for interactive mode)
warp34 chat nchapman/Qwen3.5-4B-W4A16-HGPTQ --speculative mtp

# One-off generation
warp34 generate nchapman/Qwen3.5-4B-W4A16-HGPTQ -p "Write a lock-free ring buffer in Rust" -n 512 --temp 0.6

# Prefill + decode benchmark with a realistic coding prompt
warp34 bench nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --speculative dflash2 --draft-model nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ \
  --prompt-len 512 --gen-len 128 --realistic-prompt
```

### Speculative decoding & multi-GPU serving

`warp34 serve` defaults to `--kv-format tq4` and auto-fits context length to your VRAM (`-c` overrides it explicitly):

```sh
# Qwen3.8-27B + DFlash2 block drafting on a single 24 GB GPU
warp34 serve nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --speculative dflash2 --draft-model nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ

# 180B Qwen3.8-Flash-Next + MTP-3 across 2x 24 GB GPUs (auto-fits 128K-262K ctx + hot experts)
warp34 serve nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ \
  --speculative mtp --mtp-k 3 --tp 2
```

### Multi-LoRA & API authentication

```sh
export WARP34_API_KEY="your-secret-key"
warp34 serve nchapman/Qwen3.5-9B-W4A16-HGPTQ \
  --lora-modules reviewer=org/my-lora-adapter \
  --api-key-env WARP34_API_KEY \
  --max-num-seqs 8 -c 64k
```

Clients authenticate with `Authorization: Bearer <key>` or `x-api-key: <key>`, and select the adapter by passing `"model": "reviewer"` (replace `org/my-lora-adapter` with any compatible Hugging Face repo or local adapter path).

### Zero-decode Decision API (`/v1/systemone`, `/v1/decisions`)

Evaluate up to 64 boolean (`noul`), categorical (`choice`), or rubric (`score`) questions over shared context (and optional images) in a **single prefill pass with zero decode tokens**. Works on specialized `GLiDE` / `d3` checkpoints or any standard Qwen model:

```sh
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "502 Bad Gateway across us-east-1 ingress after TLS certificate rotation.",
    "questions": {
      "is_outage": { "type": "noul", "instructions": "Is live production traffic impacted?" },
      "owner": {
        "type": "choice",
        "instructions": "Which team should be paged?",
        "criteria": { "network": "TLS, DNS, ingress", "storage": "Databases, disks" }
      }
    }
  }'
```

Returns calibrated probabilities and confidence scores without generating tokens:

```json
{
  "model": "nchapman/Qwen3.5-4B-W4A16-HGPTQ",
  "answers": {
    "is_outage": { "type": "noul", "noul": 0.98 },
    "owner": { "type": "choice", "choice": "network", "probabilities": { "network": 0.97, "storage": 0.03 }, "confidence": 0.94 }
  },
  "usage": { "input_tokens": 62, "output_tokens": 0, "cached_tokens": 0 }
}
```

<details>
<summary>More Docker options (offline host cache &amp; 2-GPU setup)</summary>

**Reuse an existing host Hugging Face cache offline:**

```sh
docker run --rm --gpus '"device=0"' \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_CACHE=/models \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src="$HOME/.cache/huggingface/hub",dst=/models,readonly \
  ghcr.io/nchapman/warp34:dev \
  serve nchapman/Qwen3.5-4B-W4A16-HGPTQ --offline --host 0.0.0.0
```

**Run on 2 GPUs** (no `--ipc=host` required):

```sh
docker run --rm --gpus '"device=0,1"' -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=warp34-models,dst=/home/warp34/.cache/huggingface/hub \
  ghcr.io/nchapman/warp34:dev \
  serve nchapman/Qwen3.8-27B-W4A16-HGPTQ --tp 2 --speculative mtp --host 0.0.0.0
```

Images are published to `ghcr.io/nchapman/warp34` on container-relevant pushes to `main` (`:dev`, `:sha-<commit>`) and on `v*` tags (`:latest`, `:X.Y`, `:X.Y.Z`). Build locally with `docker build --platform linux/amd64 -t warp34:local .`.

</details>

---

## Supported Models & Performance

| Family | Sizes / Checkpoints | Highlights |
|---|---|---|
| **Qwen 3.5 / 3.8 Dense** | `0.8B`, `2B`, `4B`, `9B`, `27B` | Full hybrid GDN + attention, Vision (`VL`), `MTP-3`, `DFlash2`, up to `262K` ctx |
| **Qwen 3.6 Sparse MoE** | `35B-A3B` (`256` experts, `3B` active) | Fits on **1× 24 GB GPU** with `MTP-3` + Vision (`tq4` KV) |
| **Qwen 3.8 Flash-Next** | `180B` (`512` experts, `6B` active, `qwen4_exp`) | HyperConnection, Sparse Attention (`QSA`), mmap `PLE`, 3-tier MoE on `1–4×` GPUs |
| **Decision Models** | `GLiDE-*`, `d3-*`, or any Qwen model | `CodeReadout` head + `CausalLm` fallback on `/v1/systemone` & `/v1/decisions` |
| **Quantizations** | `HGPTQ` (`W4A16`, `W2.6-RCO`), `GPTQ`, `AWQ`, `AutoRound`, `compressed-tensors`, `FP16`/`BF16` | Shared-weight `W4A8` INT8-IMMA prefill on `sm_86`/`sm_89` (zero extra weight VRAM) |

### Headline throughput (`W4A16-HGPTQ`, lossless verification)

| Model | Mode | 1× RTX 3090 Ti (`sm_86`) | 1× RTX 4090 (`sm_89`) | 2× RTX 3090 Ti (PCIe) | 2× RTX 4090 (PCIe) |
|---|---|---:|---:|---:|---:|
| **Qwen3.5-9B** | Cold 4K prefill (`W4A8`) | `8,632 tok/s` | `15,939 tok/s` | `8,905 tok/s` | `16,104 tok/s` |
| **Qwen3.5-9B** | Decode `MTP-3` (`C=1` / `C=16`) | `234` / `1,346` | `253` / `1,553` | `339` / `1,676` | `369` / `2,319` |
| **Qwen3.8-27B** | Cold 4K prefill (`W4A8`) | `2,350 tok/s` | `4,283 tok/s` | `3,094 tok/s` | `5,926 tok/s` |
| **Qwen3.8-27B** | Decode Plain / `MTP-3` / `DFlash2` | `52` / `150` / `248` | — | `85` / `252` / `365` | `97` / `246` / `328` |
| **Qwen3.6-35B-A3B** | Decode Plain / `MTP-3` | `205` / `429` | — | `262` / `553` | `315` / `672` |
| **Qwen3.8-Flash-Next** | Serving `MTP-3` (`W4` / `W2.6-RCO`) | `45` / `66 tok/s` | `58` / — | `73` / `107 tok/s` | `117` / — |

*Prefill and `9B` decode rows are end-to-end HTTP serving p50s (`benchmarks/perf`; `9B` and `27B` `TP=2` prefill use `FP16` KV, while `27B` `TP=1` prefill, `Flash-Next`, and `warp34 bench` rows use `TQ4` KV). `27B`/`35B` decode rows are `warp34 bench` peak acceptance (realistic coding prompts achieve `96–113 tok/s` on `1× 3090 Ti` and `156–184 tok/s` on `2×` GPUs for `27B MTP-3`). Full A100 (`sm_80`), `TP=4`, and head-to-head numbers vs. `vLLM`, `SGLang`, and `llama.cpp` are in [`docs/STATE.md`](docs/STATE.md) and [`benchmarks/perf/campaigns/standings.md`](benchmarks/perf/campaigns/standings.md).*

---

## Scope & Limitations

- **Qwen hybrid family only:** Built for Qwen 3.5 / 3.6 / 3.8, `Qwen3.8-Flash-Next` (`qwen4_exp`), and `GLiDE`/`d3` decision models. If you need Llama, Mistral, Gemma, or DeepSeek, use `vLLM`, `SGLang`, or `llama.cpp`.
- **Hardware & OS:** Linux `x86_64` with NVIDIA `sm_80` (A100), `sm_86` (RTX 30-series), or `sm_89` (RTX 40-series) GPUs. No macOS, Windows, AMD ROCm, or pre-Ampere (`sm_75` and older) support.
- **Single-node scaling:** Supports `TP=1`, `TP=2`, and `TP=4` on one machine; multi-node distributed inference is out of scope.

---

## Documentation

| Guide | Topic |
|---|---|
| [`docs/MODELS.md`](docs/MODELS.md) | Model downloads, `--revision` / `--offline`, supported quantizations, and KV cache policies |
| [`docs/HGPTQ.md`](docs/HGPTQ.md) | Hadamard-rotated `W4A16` / `W2.6-RCO` quantization format and parity benchmarks |
| [`docs/SERVING.md`](docs/SERVING.md) | Server flags, auth (`--api-key-env`), concurrency limits, and security |
| [`docs/GENERATIVE_APIS.md`](docs/GENERATIVE_APIS.md) | Wire contracts for OpenAI Chat/Responses/Completions and Anthropic Messages |
| [`docs/STRUCTURED_OUTPUTS.md`](docs/STRUCTURED_OUTPUTS.md) | `<think>`-aware XGrammar JSON Schema, Regex, CFG, and Qwen XML tool constraints |
| [`docs/QWEN_FLASH_NEXT.md`](docs/QWEN_FLASH_NEXT.md) | `Qwen3.8-Flash-Next` (`qwen4_exp`) architecture, `QSA`, `PLE`, and 3-tier MoE offloading |
| [`docs/DECISION_MODELS.md`](docs/DECISION_MODELS.md) | `/v1/systemone` & `/v1/decisions` schemas, `GLiDE`/`d3`/`CausalLm` profiles, and JevBench |
| [`docs/METRICS.md`](docs/METRICS.md) | Embedded web dashboard, Prometheus `/metrics`, and `/api/metrics` JSON schema |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Engine internals, donor map (`vLLM` / `SGLang` / `vllm.cpp` / `warp3090`), and invariants |
| [`docs/HARDWARE.md`](docs/HARDWARE.md) | `sm_80` / `sm_86` / `sm_89` rooflines, PCIe/NVLink collectives, and ISA traps |
| [`docs/FEATURES.md`](docs/FEATURES.md) | Complete capability inventory |
| [`benchmarks/perf/README.md`](benchmarks/perf/README.md) | Standalone HTTP benchmark harness and multi-engine campaign configs |

---

## Development & Contributing

```sh
mise run check-quick   # fmt + clippy (cuda & cpu) + rustdoc + SIMD & CUDA binary gates
mise run test-quick    # fast GPU unit test suite
mise run check         # full release gate + compute-sanitizer (requires GPUs + mise run test-prep)
```

For kernel and parity work, `warp34` also ships built-in oracle subcommands (`dump-diff`, `greedy-diff`, and `profile-diff`) to verify layer tensors, greedy/speculative parity, and `nsys` kernel timings against `vLLM` and `SGLang`.

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for bug reports, regression-first fixes, and PR verification requirements.

## License

Licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attributions.
