# warp34

The fastest single-binary inference engine for **Qwen 3.5, 3.6, 3.8, and Flash-Next** on NVIDIA GPUs—outperforming `vLLM`, `SGLang`, and `llama.cpp` across prefill, decode, continuous batching, and long-context coding agents.

[![Docker](https://github.com/nchapman/warp34/actions/workflows/docker.yml/badge.svg)](https://github.com/nchapman/warp34/actions/workflows/docker.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-sm__80%20%7C%20sm__86%20%7C%20sm__89-76b900.svg)](docs/HARDWARE.md)
[![Models on HF](https://img.shields.io/badge/Hugging%20Face-nchapman%2Fmodels-ffd21e.svg)](https://huggingface.co/nchapman/models)

Written in Rust and CUDA with zero Python and zero NCCL. Built specifically for Qwen's hybrid Gated DeltaNet + full-attention architecture on RTX 30-series (`sm_86`), RTX 40-series (`sm_89`), and A100 (`sm_80`) GPUs.

---

## Quickstart

**Requirements:** Linux `x86_64` (AVX2/FMA), an NVIDIA RTX 30-series, RTX 40-series, or A100 GPU, and a CUDA 13-capable driver (R580+).

> **VRAM note:** The commands below run `nchapman/Qwen3.5-9B-W4A16-HGPTQ` (`~8.6 GiB`, 12 GB+ GPUs). On an **8 GB GPU**, swap in `nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO` (`6.0 GiB`)—or pick any checkpoint from [Pick a Model for Your GPU](#pick-a-model-for-your-gpu).

### Option A: Docker (fastest — no build tools or host CUDA toolkit needed)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```sh
docker run --rm --gpus '"device=0"' -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=warp34-models,dst=/home/warp34/.cache/huggingface/hub \
  ghcr.io/nchapman/warp34:dev \
  serve nchapman/Qwen3.5-9B-W4A16-HGPTQ --speculative mtp
```

### Option B: Build from source

Requires CUDA 13.4 (`/usr/local/cuda` or `/opt/cuda`, or set `CUDA_HOME`) and [mise](https://mise.jdx.dev/) (installs pinned Rust, Node, and pnpm). Install [Git LFS](https://git-lfs.com/) only if you plan to run the oracle test suite:

```sh
git clone https://github.com/nchapman/warp34.git
cd warp34

mise install
mise run build   # tip: set WARP34_ARCH=sm_89 (or sm_86/sm_80) to compile only your GPU arch

./target/release/warp34 serve nchapman/Qwen3.5-9B-W4A16-HGPTQ --speculative mtp
```

### Verify it works

`warp34 serve` downloads the checkpoint into `~/.cache/huggingface/hub` on first run, auto-sizes the KV cache to your GPU, and prints `Endpoint : http://127.0.0.1:8000` when ready:

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nchapman/Qwen3.5-9B-W4A16-HGPTQ",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

Open **`http://127.0.0.1:8000/`** in your browser for live telemetry, **`/playground`** to test every API interactively, and **`/connect`** for ready-to-paste coding-agent configs. Pass `--no-webui` for an API-only server.

---

## What Makes `warp34` Great

- **Fastest Qwen hybrid inference on consumer GPUs** — Beats `vLLM`, `SGLang`, and `llama.cpp` via purpose-built CUDA kernels, shared-weight `W4A8` prefill on RTX 30/40-series (zero extra weight VRAM), and lossless `MTP-3` / `DFlash2` speculation that composes cleanly with prefix caching and multi-GPU.
- **Fit Vision, `MTP-3`, `128K` context, and LoRA on one card** — Pair our [optimized `HGPTQ` & `W2.6-RCO` checkpoints](https://huggingface.co/nchapman/models) with built-in 4-bit TurboQuant KV (`tq4`, 3.07× compression) so you never have to trade off context length, vision, or model size.
- **Zero-config VRAM tuning** — Just run `warp34 serve <model>`. The planner automatically maximizes context (up to `256K` / `262,144` tokens), selects the right KV format, and balances hot experts against context on tiered MoE models while preserving safe headroom.
- **Run 180B `Qwen3.8-Flash-Next` on 1–2 consumer GPUs** — Streams n-gram tables from disk via `mmap` (`51 GB` FP8 / `~14 GB` `lut4-gptq`), caches hot experts in VRAM, prefetches cold experts over PCIe, and evaluates misses on a 0-ULP AVX-512/AVX2 CPU bridge—reaching **`66 tok/s`** on `1× 24 GB` and **`107 tok/s`** on `2× 24 GB` (`W2.6-RCO` + `MTP-3`).
- **One server for every API & coding agent** — Drop-in OpenAI Chat, OpenAI Responses, and Anthropic Messages APIs (with streaming `<think>` reasoning, tool calls, and `<think>`-aware XGrammar structured outputs), plus an embedded web dashboard, API Playground, and `/connect` agent setup guide.
- **Single-pass classification & hot-swappable LoRAs** — Score up to 64 structured questions in one prefill (`0` decode tokens) via `/v1/systemone` & `/v1/decisions`, route up to **62 concurrent LoRA adapters** (`--lora-modules`), and scale across `2–4×` GPUs (`--tp 2` / `--tp 4`) without NCCL or `--ipc=host`.

---

## Pick a Model for Your GPU

Our [Hugging Face collection (`nchapman/models`)](https://huggingface.co/nchapman/models) hosts Hadamard-rotated `W4A16-HGPTQ` and mixed-precision `W2.6A16-HGPTQ-RCO` checkpoints tuned specifically for `warp34`. Every row below runs with zero manual KV/memory flags—just pass the model ID to `warp34 serve` (add `--speculative mtp` to enable `MTP-3`):

| GPU VRAM | Recommended Checkpoint (`nchapman/...`) | Footprint @ `128K tq4` | Why pick it |
|---|---|---:|---|
| **8 GB** *(RTX 4060)* | `Qwen3.5-9B-W2.6A16-HGPTQ-RCO` *(2.60 bpw)* | `6.0 GiB` *(~935 MiB free w/ LoRA)* | Full **Vision + `MTP-3` + `128K`** on 8 GB (`151.5 tok/s` decode w/ MTP) |
| **12 GB** *(RTX 3060 / 4070)* | `Qwen3.5-9B-W4A16-HGPTQ` *(4.00 bpw)* | `8.6 GiB` *(~2.4 GiB free w/ LoRA)* | Best quality on 12 GB (`140.5 tok/s` w/ MTP); alt: `Qwen3.8-27B-W2.2A16-HGPTQ-RCO` (`10.6 GiB` @ `64K tq4_all`) |
| **16 GB** *(RTX 4060 Ti / 4080)* | `Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO` *(2.62 bpw)* | `14.6 GiB` *(~221 MiB free w/ LoRA)* | Fast MoE pick (`190.8 tok/s` w/ MTP); dense alt: `Qwen3.8-27B-W2.6A16-HGPTQ-RCO` (`14.6 GiB`, `48.5 tok/s` w/ MTP) |
| **1× 24 GB** *(RTX 3090 / 4090)* | `Qwen3.8-27B-W4A16-HGPTQ` or `Qwen3.6-35B-A3B-W4A16-HGPTQ` | `~19–21 GiB` | Full `W4` quality + Vision + `MTP-3`/`DFlash2` (`128K–256K`); or `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`19.0 GiB`, `66 tok/s` w/ `MTP-3`) |
| **2× 24 GB** *(TP=2)* | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` | `20.8 GiB/GPU` | Flagship 180B MoE + `128K–256K` ctx + `MTP-3` (`107 tok/s` auto-fit; up to `131.6 tok/s` on a dedicated headless GPU) |

*Table throughput numbers are single-stream `MTP-3` decode measurements (`Flash-Next` on `1–2× RTX 3090 Ti`); actual speed scales with GPU memory bandwidth. Standard upstream `FP16`/`BF16`, `GPTQ`, `AWQ`, `AutoRound`, and `compressed-tensors` Qwen checkpoints also work out of the box—see [`docs/MODELS.md`](docs/MODELS.md) and [`docs/HGPTQ.md`](docs/HGPTQ.md).*

---

## APIs & Coding Agents

A single `warp34 serve` instance speaks all four API families simultaneously, with hybrid KV + recurrent-state prefix caching shared across requests:

| Protocol | Endpoints | Highlights |
|---|---|---|
| **OpenAI Chat & Completions** | `POST /v1/chat/completions`, `/v1/completions` | Streaming SSE, tool calling, `reasoning_effort`, XGrammar `response_format` (JSON Schema / Regex / CFG), `top_logprobs` |
| **OpenAI Responses** | `POST /v1/responses`, `GET/POST .../cancel` | Stateful multi-turn chaining, streaming `reasoning_text` & `function_call` deltas, structured outputs |
| **Anthropic Messages** | `POST /v1/messages`, `/v1/messages/count_tokens` | Native streaming `thinking` + `tool_use` blocks; strips Claude Code billing headers so prefix caches stay warm |
| **Decision APIs** | `POST /v1/systemone`, `/v1/decisions` | Single-prefill (`0` output tokens) calibrated probabilities for `noul`, `choice`, and `score` questions (`GLiDE`, `d3`, or any Qwen model) |
| **UI & Observability** | `GET /`, `/playground`, `/connect`, `/metrics` | Embedded dashboard, multi-API Playground, agent config generator, and Prometheus metrics |

### Connect your coding agent

Open **`http://127.0.0.1:8000/connect`** while `warp34 serve` is running for copy-paste configs tailored to your active model and context window (**Claude Code, Codex CLI, OpenCode, Pi, Aider, and Python/TS SDKs**).

For example, to point **Claude Code** directly at your local `warp34` server:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 \
ANTHROPIC_AUTH_TOKEN=warp34 \
claude --model "nchapman/Qwen3.5-9B-W4A16-HGPTQ"
```

---

## Common Recipes

> The examples below assume `warp34` is on your `PATH` (e.g., `export PATH="$PWD/target/release:$PATH"` after building from source, or replace `warp34` with `docker run --rm --gpus '"device=0"' ... ghcr.io/nchapman/warp34:dev`).

### Speculative decoding & multi-GPU serving

`warp34 serve` defaults to `--kv-format tq4` and auto-fits context length to your VRAM (`-c` overrides it explicitly):

```sh
# Qwen3.8-27B + DFlash2 block drafting on a single 24 GB GPU
warp34 serve nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --speculative dflash2 --draft-model nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ

# 180B Qwen3.8-Flash-Next + MTP-3 across 2x 24 GB GPUs (auto-fits 128K-256K ctx + hot experts)
warp34 serve nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO \
  --speculative mtp --tp 2
```

### Hot-swappable Multi-LoRA

Load up to 62 PEFT LoRA adapters alongside the base model and select any adapter per request by passing `"model": "<name>"`:

```sh
warp34 serve nchapman/Qwen3.5-9B-W4A16-HGPTQ \
  --lora-modules reviewer=org/my-lora-adapter \
  --speculative mtp
```

*(Replace `org/my-lora-adapter` with any compatible Hugging Face repo or local adapter directory.)*

### API authentication & concurrency limits

```sh
export WARP34_API_KEY="your-secret-key"
warp34 serve nchapman/Qwen3.5-9B-W4A16-HGPTQ \
  --api-key-env WARP34_API_KEY \
  --max-num-seqs 8 -c 64k
```

Clients authenticate with `Authorization: Bearer <key>` or `x-api-key: <key>`.

### Interactive terminal chat

```sh
# Multi-turn terminal REPL (pass -p "..." for a single turn)
warp34 chat nchapman/Qwen3.5-9B-W4A16-HGPTQ --speculative mtp
```

### Zero-decode Decision API (`/v1/systemone`, `/v1/decisions`)

Evaluate up to 64 boolean (`noul`), categorical (`choice`), or rubric (`score`) questions over shared context (and optional images) in a **single prefill pass with zero decode tokens**. Works on specialized `GLiDE` / `d3` checkpoints (`96.9%` JevBench, `100%` MultiQ) or any standard Qwen model:

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
  "model": "nchapman/Qwen3.5-9B-W4A16-HGPTQ",
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
  serve nchapman/Qwen3.5-9B-W4A16-HGPTQ --offline
```

**Run on 2 GPUs** (no `--ipc=host` required):

```sh
docker run --rm --gpus '"device=0,1"' -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=warp34-models,dst=/home/warp34/.cache/huggingface/hub \
  ghcr.io/nchapman/warp34:dev \
  serve nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO --tp 2 --speculative mtp
```

Images are published to `ghcr.io/nchapman/warp34` on container-relevant pushes to `main` (`:dev`, `:sha-<commit>`) and on `v*` tags (`:latest`, `:X.Y`, `:X.Y.Z`). Build locally with `docker build --platform linux/amd64 -t warp34:local .`.

</details>

---

## Performance Snapshot (`HGPTQ` / `RCO`, Lossless Verification)

| Model | Mode | 1× RTX 3090 Ti (`sm_86`) | 1× RTX 4090 (`sm_89`) | 2× RTX 3090 Ti (PCIe) | 2× RTX 4090 (PCIe) |
|---|---|---:|---:|---:|---:|
| **Qwen3.5-9B** | Cold 4K prefill (`W4A8`) | `8,632 tok/s` | `15,939 tok/s` | `8,905 tok/s` | `16,104 tok/s` |
| **Qwen3.5-9B** | Decode `MTP-3` (`C=1` / `C=16`) | `234` / `1,346` | `253` / `1,553` | `339` / `1,676` | `369` / `2,319` |
| **Qwen3.8-27B** | Cold 4K prefill (`W4A8`) | `2,350 tok/s` | `4,283 tok/s` | `3,094 tok/s` | `5,926 tok/s` |
| **Qwen3.8-27B** | Decode Plain / `MTP-3` / `DFlash2` | `52` / `150` / `248` | — | `85` / `252` / `365` | `97` / `246` / `328` |
| **Qwen3.6-35B-A3B** | Decode Plain / `MTP-3` | `205` / `429` | — | `262` / `553` | `315` / `672` |
| **Qwen3.8-Flash-Next** | Serving `MTP-3` (`W4` / `W2.6-RCO`) | `45` / `66 tok/s` | `58` / — | `73` / `107 tok/s` | `117` / — |

*Prefill and `9B` decode rows are end-to-end HTTP serving p50s (`benchmarks/perf`; `9B` and `27B` `TP=2` prefill use `FP16` KV, while `27B` `TP=1` prefill, `Flash-Next`, and `warp34 bench` rows use `TQ4` KV). `27B`/`35B` decode rows are `warp34 bench` peak acceptance (realistic coding prompts achieve `96–113 tok/s` on `1× 3090 Ti` and `156–184 tok/s` on `2×` GPUs for `27B MTP-3`). Full head-to-head numbers vs. `vLLM`, `SGLang`, and `llama.cpp` are in [`docs/STATE.md`](docs/STATE.md) and [`benchmarks/perf/campaigns/standings.md`](benchmarks/perf/campaigns/standings.md).*

---

## Scope & Limitations

- **Qwen hybrid family only:** Built for Qwen 3.5 / 3.6 / 3.8, `Qwen3.8-Flash-Next` (`qwen4_exp`), and `GLiDE`/`d3` decision models. If you need Llama, Mistral, Gemma, or DeepSeek, use `vLLM`, `SGLang`, or `llama.cpp`.
- **Hardware & OS:** Linux `x86_64` with NVIDIA RTX 30-series (`sm_86`), RTX 40-series (`sm_89`), or A100 (`sm_80`) GPUs. No macOS, Windows, AMD ROCm, or pre-Ampere (`sm_75` and older) support.
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

## Acknowledgments

`warp34` builds on ideas, kernels, and reference designs from these projects and papers:

- **[vLLM](https://github.com/vllm-project/vllm)** & **[Marlin (`IST-DASLab/marlin`)](https://github.com/IST-DASLab/marlin)** — Marlin mixed-precision GEMM kernels, FP16 Qwen model spine, `MTP`, GPU sampler pipeline, and reasoning/tool parser reference.
- **[SGLang](https://github.com/sgl-project/sglang)** — `MambaRadixCache` hybrid KV + recurrent-state prefix caching and `DFlash2` block-parallel speculative verification.
- **[`localai-org/vllm.cpp`](https://github.com/localai-org/vllm.cpp)** — Native CUDA C++ Gated DeltaNet (GDN) recurrence and chunked WY prefill kernel reference.
- **[TurboQuant (`arXiv:2504.19874`)](https://arxiv.org/abs/2504.19874)** & **[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)** — Walsh-Hadamard + Beta Lloyd-Max 4-bit KV quantization method, block format, and fused attention kernels behind `TQ4`.
- **[`DominikBucko/qwen38-flash-next-2x3090`](https://github.com/DominikBucko/qwen38-flash-next-2x3090)** & **[`Niko1221/Strata`](https://github.com/Niko1221/Strata)** — Tiered MoE offloading, hot/cold expert caching, and `Qwen3.8-Flash-Next` consumer-GPU serving reference designs.

## License

Licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attributions.
