# warp-34

`warp-34` is a Rust and CUDA inference engine built specifically for the **Qwen 3.5, 3.6, and 3.8 hybrid model family** running on **1 to 4 NVIDIA RTX 30-series (`sm_86`) and 40-series (`sm_89`) GPUs**.

If you have one or more RTX 3090s or 4090s and want to run modern Qwen models—dense hybrids, sparse MoEs, vision-language models, the `Qwen3.8-Flash-Next` (Qwen4 preview) architecture, or zero-decode decision models—`warp-34` is built to get more speed and context out of your cards than general-purpose engines like `vLLM` or `SGLang`.

### What you need to know up front

- **It is a single binary with zero Python at runtime.** `w34` handles Safetensors loading, Hugging Face Hub downloads (including Xet transfers), scheduling, speculative decoding, multi-GPU coordination, an OpenAI/Anthropic/Decision HTTP server, and an embedded live web dashboard.
- **It is purpose-built for hybrid recurrent + attention models.** Qwen 3.5/3.6/3.8 models interleave 3 layers of Gated DeltaNet (GDN linear attention) with 1 layer of full attention. `warp-34` treats both states as first-class citizens: prefix caching (`MambaRadixCache`), speculative decoding (`DFlash2` and `MTP`), and continuous batching all checkpoint and roll back GDN recurrent state and paged KV cache together atomically.
- **It fits huge contexts and huge models onto 24 GB cards.** Our **TurboQuant 4-bit KV cache (`TQ4`)** compresses full-attention KV by **3.07×** (`tq4`) to **3.88×** (`tq4_all`) directly inside the FlashAttention/paged-decode kernels, letting a single 24 GB GPU run `Qwen3.8-27B` up to **262K context** or fit `Qwen3.6-35B-A3B` comfortably on one card.
- **It scales across consumer motherboards without NCCL.** Multi-GPU (`--tp 2` and `--tp 4`) uses custom `~18 µs` collectives that automatically pick **1-hop NVLink P2P** when a bridge is present or **2-hop pinned system memory over PCIe** when P2P is disabled on consumer boards—all fully captured inside CUDA Graphs.

---

## Quickstart

### 1. Clone and build

You need an x86-64 Linux machine, an RTX 30-series or 40-series GPU, CUDA 13.4 (`/opt/cuda`), and [Git LFS](https://git-lfs.com/) (for test fixtures). [mise](https://mise.jdx.dev/) manages the pinned Rust, Node, and pnpm toolchain.

```sh
git lfs install
git clone https://github.com/nchapman/warp-34.git
cd warp-34

export PATH="/opt/cuda/bin:$PATH"
mise install
mise run build   # builds the embedded web dashboard + release binary with CUDA
```

### 2. Run a server (downloads the model automatically on first run)

Point `w34 serve` at a Hugging Face repo ID or a local directory. It will pull the required Safetensors shards into your standard Hugging Face cache (`~/.cache/huggingface/hub`) and start listening:

```sh
./target/release/w34 serve \
  --model nchapman/Qwen3.8-27B-W4A16-GPTQ \
  --speculative dflash2 --draft-model z-lab/Qwen3.5-27B-DFlash \
  --kv-format tq4 --tp 1 --port 8000
```

While it is running, open **`http://127.0.0.1:8000/`** in your browser to see the live telemetry dashboard (throughput, TTFT, prefix hit rate, and separate KV vs. GDN state capacity).

### 3. Talk to it

`w34 serve` speaks **OpenAI Chat Completions** (`/v1/chat/completions`), **OpenAI Responses** (`/v1/responses`), **OpenAI Completions** (`/v1/completions`), **Anthropic Messages** (`/v1/messages`), and **System One / Decision APIs** (`/v1/systemone`, `/v1/decisions`) out of the box:

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nchapman/Qwen3.8-27B-W4A16-GPTQ",
    "messages": [{"role": "user", "content": "Explain why Gated DeltaNet needs state checkpoints for prefix caching."}],
    "max_tokens": 256
  }'
```

For quick offline testing or benchmarking from the terminal without spinning up a server:

```sh
# One-off generation:
./target/release/w34 generate --model nchapman/Qwen3.5-4B-W4A16-GPTQ --prompt "Hello!"

# Benchmark prefill + decode with lossless speculative verification check:
./target/release/w34 bench \
  --model nchapman/Qwen3.8-27B-W4A16-GPTQ \
  --speculative dflash2 --draft-model z-lab/Qwen3.5-27B-DFlash \
  --kv-format tq4 --tp 1 --prompt-len 512 --gen-len 128 --verify-lossless
```

---

## Deep Dive 1: The Qwen 3.5 / 3.6 / 3.8 Hybrid Family

The core Qwen 3.5 family (`model_type: "qwen3_5"` and `"qwen3_5_moe"`—which also backs checkpoints labeled `Qwen3.6-35B-A3B` and `Qwen3.8-27B`) uses a 3:1 hybrid block: three Gated DeltaNet (`QwenGatedDeltaNetAttention`) linear-attention layers followed by one full-attention layer (`Qwen3NextAttention` with packed output sigmoid gating, `GemmaRMSNorm`, and partial 3D `mRoPE`).

### Supported models & formats

- **Dense hybrids:** `0.8B`, `4B`, `9B`, and `27B` (`Qwen3.8-27B`).
- **Sparse MoE hybrid:** `Qwen3.6-35B-A3B` (`256` routed experts + `1` shared expert, `3B` active parameters per token). With `--kv-format tq4`, the entire `35B-A3B` model runs on a **single 24 GB GPU** at **186.5 tok/s** (`C=1`) and **710.2 tok/s** (`C=8`), or across **2× GPUs (`TP=2`)** at **319.6 tok/s** (`C=1`, `MTP k=2`) and **1,009.7 tok/s** (`C=8`).
- **Multimodal Vision (`Qwen3.5-VL`):** Native support for `4B`, `27B`, and `35B-A3B` vision checkpoints (`Qwen3_VisionTransformer` 24/27-layer ViT, 2×2 spatial merger, 3D `mRoPE`, PNG/JPEG/PPM decoding, and content-hashed vision prefix caching).
- **Quantization:** `GPTQ`, `AWQ`, `compressed-tensors`, and `AutoRound` (`W4A16` symmetric/asymmetric and `W8A16`), plus unquantized `FP16`/`BF16`. All quantized linear layers are repacked in parallel at load time into unified Marlin tiles (`~7–10s` warm load time on 27B/35B).

### Speculative decoding (`DFlash2` and `MTP`)

Standard engines struggle to combine speculative decoding with hybrid recurrent models—for example, `vLLM 0.31.0` cannot combine hybrid prefix caching with draft models and its `DFlash2` implementation stalls on `TP=2`. `warp-34` supports two speculative engines that compose cleanly with `TP=1/2/4` and prefix caching:

1. **`DFlash2` Block-Parallel Drafter (`--speculative dflash2`):**
   - Taps multi-layer target hidden states (`TapPlan`) to condition a fast, native-`BF16` block drafter (`M=16` verify rows on `4B`, `M=8` verify rows with a `2048` sliding window on `27B`).
   - Preserves native `BF16` weights and activations for the drafter (preventing the `FP16` overflow seen on the 4B drafter) while keeping the target model on an `FP16` spine.
2. **Native Multi-Token Prediction (`--speculative mtp --mtp-k 1..3`):**
   - Uses the checkpoint's built-in 1-layer TP-sharded MTP sidecar (dense or MoE), including batched multi-sequence verify (`B = 2..4`).
   - Quantizes the MTP `draft_lm_head` on the GPU at load time into `MarlinW4A16` (`group_size = 64`), shrinking its memory bandwidth footprint by **3.56×** without touching the target model's lossless verification head.

### Reference throughput (`prompt_len=512`, `gen_len=128`, `--kv-format tq4`, verified lossless)

| Model | Mode | 1× RTX 3090 Ti | 1× RTX 4090 | 2× RTX 3090 Ti (PCIe) | 2× RTX 4090 (PCIe) | 4× RTX 3090 (`TP=4`) |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.5-4B** | `DFlash2 (M=16)` | `1,228–1,388 tok/s` | — | — | — | — |
| **Qwen3.8-27B** | Plain (`M=1`) | `51.6 tok/s` | `55.0 tok/s` | `86.4 tok/s` | `92.5 tok/s` | `129.1 tok/s` |
| **Qwen3.8-27B** | `MTP (k=3)` | `120.7 tok/s` | `125.8 tok/s` | `163.9 tok/s` | `175.4 tok/s` | `291.0 tok/s` |
| **Qwen3.8-27B** | `DFlash2 (k=7)` | `240.6 tok/s` | `253.0 tok/s` | `192.2 tok/s` | `204.3 tok/s` | `438.3 tok/s` |

*(On 2× RTX 3090 with an NVLink bridge, `27B TP=2` 512-token prefill TTFT drops by up to **79%** and `DFlash2` decode climbs to **197.0 tok/s**.)*

---

## Deep Dive 2: `Qwen3.8-Flash-Next` (The Qwen4 Architecture Preview)

Despite the `3.8` in its name, **`Qwen3.8-Flash-Next`** (`Qwen4ExpForConditionalGeneration`, `model_type: "qwen4_exp"`) is the first open-weight release of the **Qwen4** architecture. It is a **180B total parameter** checkpoint (`125B` hybrid MoE backbone with `6B` active per token + `51.2B` n-gram embedding table + `4B` MTP drafter).

Running a 180B-parameter checkpoint on 1 or 2 consumer 24 GB GPUs without swap-thrashing or stalling on PCIe transfers requires purpose-built support for all five of its new architectural primitives:

1. **HyperConnection (`GatedResidual`, `hc_count = 4`):**
   Instead of a single `[T, 2560]` residual stream, all 48 layers carry 4 parallel streams (`[T, 4, 2560]`) mixed via per-branch `GemmaRMSNorm` and low-rank (`320`-dim) input/injection projections. `warp-34` fuses the branch combine and grouped RMSNorm and runs the low-rank down + inject projections in a single dual skinny GEMM kernel (`w34_small_n_gemm_dual_f16`), eliminating 240 separate cuBLAS launches per token.
2. **Qwen Sparse Attention (`QSA`):**
   On the 12 full-attention layers, a lightweight 4-query MQA indexer scores 4:1 mean-pooled key blocks and selects at most `512` blocks (`2,048` tokens) per query to attend over in the main `TQ4`/`Q8_0`/`FP16` paged KV cache. During `MTP` drafting, steps 1..2 reuse step 0's selected block indices.
3. **N-Gram Per-Layer Embeddings (`PLE`, `51.3 GiB` across 128 FP8 shards):**
   Loading 51.3 GiB of n-gram tables into anonymous RAM alongside routed experts would exhaust a 64–128 GB workstation. Instead, `PleMmapTable` memory-maps the 128 `F8_E4M3` shards read-only from disk, computes deterministic `splitmix64` 3-gram row IDs, prefaults unique 4 KiB pages in parallel with `madvise(MADV_WILLNEED)`, gathers only the needed `2,560 B/token`, and updates the dilated short-conv state inside `StateBundlePool` (so prefix caching and speculative rollback work automatically).
4. **3-Tier 512-Expert MoE Engine (`top-10` routed + `1` shared expert per layer):**
   - **Tier 1 (GPU VRAM Hot Cache):** Hardware-aware VRAM auto-planning keeps `88–96` hot experts/layer in VRAM on `TP=1`, `112` experts/layer/rank (`224/512 = 43.75%`) on `TP=2 / EP=2`, and **all `512` experts (`128/rank = 100%`) in VRAM on `TP=4 / EP=4`** (zero CPU offload).
   - **Tier 2 (Pinned RAM + 0-ULP CPU MoE Bridge):** On 1× and 2× 24 GB setups, cold experts live in pinned system RAM (`HostExpertStore`). During prefill, an async H2D stream stages cold experts two groups ahead of GPU Grouped-MMA compute. During decode, cold experts are evaluated either via on-device GPU LRU pull (`GpuLru`) or in parallel by our lock-free **Rust CPU MoE Bridge** (`CpuHybrid`), which dispatches across physical CPU cores using `AVX-512`, `AVX2+FMA+F16C`, or portable scalar code—all sharing the exact same 16-lane pairwise reduction tree for **0-ULP bit-identity**.
5. **Dual-Stream Qwen4 `MTP` (`K = 3`):**
   Pairs the 1-layer QSA + 512-expert `INT4 g32` MTP sidecar with an online `MarlinW4A16`-quantized `draft_lm_head` and automatic VRAM residency for MTP routed experts (`100%` VRAM-resident on `TP >= 2`).

```sh
# Run Qwen3.8-Flash-Next on 1, 2, or 4 GPUs with 3-token MTP speculation:
./target/release/w34 serve \
  --model albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --speculative mtp --mtp-k 3 --kv-format tq4 --tp 2 --port 8000
```

On `TP=2` with `MTP-3`, `warp-34` reaches **92.8–136.0 tok/s** on warm interactive chat, **65.3–69.3 tok/s** on live HTTP `short-chat` serving (`4,096` TQ4 pages), and **147.2 tok/s** on `TP=4 / EP=4` (`100%` VRAM-resident). See [`docs/QWEN_FLASH_NEXT.md`](docs/QWEN_FLASH_NEXT.md) for the complete architecture guide.

---

## Deep Dive 3: Decision Models (Zero-Decode System-1 Classifiers & Scorers)

Sometimes you don't want a model to chat—you want it to read a ticket, document, or image and answer 10 structured routing, classification, or rubric-scoring questions in a single shot.

`warp-34` includes a native **Decision Engine** that evaluates up to **64 typed questions** over a shared `state` (and optional multimodal `images`) in **a single prefill forward pass with zero autoregressive decode tokens (`output_tokens = 0`)**. It returns calibrated IEEE-754 `f64` probability distributions, expected rubric scores, and analytical confidence metrics over two wire-compatible endpoints:

- **`POST /v1/systemone`** — The TypeSafe System One standard contract.
- **`POST /v1/decisions`** — An ergonomic ordered-array/object adapter (`object: "decision.list"`).

### Three question types

1. **`noul` (boolean / proposition probability):** Returns `"noul": 0.0..1.0` representing $P(\text{true})$ for yes/no or proposition checks.
2. **`choice` (categorical classification):** Evaluates named option keys (`2..64` options) and returns the winning key, full `"probabilities"` map, and calibrated `"confidence"`.
3. **`score` (ordinal rubric scoring):** Evaluates ordered rubric levels (`0..K-1`) and returns the expected value $\sum_i i \cdot p_i$, the `"legend"`, per-level `"probabilities"`, and ordinal `"confidence"`.

### Auto-detected decision model profiles

When you load a checkpoint, `warp-34` automatically inspects `config.json` and any sidecar head weights (`joint_head.safetensors` or `head.safetensors`) to select the right readout pipeline:

| Profile | Example Checkpoints | How It Evaluates Decisions |
|---|---|---|
| **`Clef`** | `Cloudflare/clef-flash` (9B), `Cloudflare/clef` (27B) | Encodes `STATE:`, optional vision media, and **all questions (`1..64`) into one joint prefill pass**, then runs `ClefJointHead` (`EvidenceRoutingLayer` + `TransformerDecoderLayer` + on-demand lexical `lm_head` priors). |
| **`Kev`** | Qwen 3.5 `KevModel` checkpoints (`head.safetensors`) | Encodes FIM/box delimiters (`<\|box_start\|>...<\|box_end\|>`), reads post-norm hidden states before `lm_head`, and scores via `PointerHead` (`f64` dot-product, skipping the full vocabulary GEMM). |
| **`Nimble`** | `Bespoke-Nimble-9B` | Compiles `SafeJson` prompts and reads calibrated last-row `lm_head` candidate logits (`A..Z`, `nimble_temperature`). |
| **`OpenJev`** | `OpenJevModel` checkpoints | Reads single-token candidate logits across `A..Z` and `a..z` (`2..52` options) with `true`-first `noul` ordering. |
| **`Lev` / `Xor`** | Dense (`Lev`) & MoE (`Xor`) decision checkpoints | Evaluates **both forward and reverse option orders (`V = 2`)** and averages probabilities in `f64` to cancel positional bias; uses a 9-level rating scale (`0..8`) for `noul` on `Lev`. |
| **`Tev1` (Default)** | `togethercomputer/Tev1-4B`, `Tev1-0.8B`, or **any standard Qwen 3.5 / 3.6 / 3.8 model** | Works on any loaded Qwen chat model with no special head required—uses `"state"`-first JSON formatting, a non-thinking `<think>\n\n</think>\n\n` prefix, and LCP shared-prefix batching across `MambaRadixCache`. |

### Example `/v1/systemone` request

```sh
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Customer reports 502 Bad Gateway across us-east-1 API ingress after TLS certificate rotation.",
    "questions": {
      "is_outage": {
        "type": "noul",
        "instructions": "Is live production traffic currently impacted?"
      },
      "owner_team": {
        "type": "choice",
        "instructions": "Which engineering team should be paged?",
        "criteria": {
          "network": "DNS, TLS, ingress, or load balancers",
          "storage": "Databases, object storage, or disks",
          "billing": "Quotas, invoices, or payments"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "Rate operational severity from 0 (cosmetic) to 3 (critical).",
        "criteria": [
          "Cosmetic issue",
          "Minor degradation with workaround",
          "Major feature unavailable",
          "Production outage affecting live traffic"
        ]
      }
    }
  }'
```

When multiple questions share the same `state` on single-question profiles (`Kev`, `Nimble`, `Tev1`, etc.), `warp-34` warms the shared prefix once into `MambaRadixCache` (KV pages + GDN state snapshot) and fans out only the short question suffixes. See [`docs/DECISION_MODELS.md`](docs/DECISION_MODELS.md) for the full wire contract and calibration math.

---

## Deep Dive 4: Why `warp-34` Is Fast (Engine Internals)

Every subsystem in `warp-34` is built against an explicit reference donor (`vLLM 0.31.0` for the `FP16` spine, Marlin GEMMs, `MTP`, 9-step GPU sampler, and parsers; `SGLang v0.5.19` for `MambaRadixCache` and `DFlash2` verification; `vllm.cpp` for native CUDA C++ GDN kernels; and `warp3090` for `TQ4` KV compression and low-latency collectives) and verified continuously against them.

1. **Paged TurboQuant 4-Bit KV Cache (`TQ4`):**
   - Full-attention KV rows (`head_dim = 256`) are rotated on the fly using a Seed-42 Rademacher sign flip + Fast Walsh-Hadamard Transform (`WHT-256`) and quantized against a 16-centroid Beta(256) Lloyd-Max codebook into `132 B` rows (`block_tq4_0`, `4.125` bits/weight).
   - `--kv-format tq4` uses a **boundary-anchored recipe** (`3.07×` compression): the first 2 and last 2 full-attention layers use rotated `Q8_0` (`272 B/row`) while the 12 interior layers use `TQ4` (`132 B/row`). `--kv-format tq4_all` compresses all layers to `TQ4` (`3.88×`).
   - Both `tq4` and `tq4_all` pass **8/8 needle-in-a-haystack tests** from `32K` through `262K` context on `Qwen3.8-27B` on a single 24 GB GPU.
2. **Hybrid Prefix Caching (`MambaRadixCache`):**
   - Caching full-attention KV pages alone is useless on a hybrid model if you have to recompute the Gated DeltaNet recurrent state from token 0.
   - `MambaRadixCache` maintains dual reference counts (`full_lock_ref` for KV pages and `mamba_lock_ref` for chunk-aligned `C = 64` GDN + PLE `StateBundle` snapshots) and uses 2-slot ping-pong track buffers to donate committed recurrent states into the radix tree—cutting warm prefix TTFT by **up to 95%** and restoring speculative draft context (`PrefixMemo`) on cache hits.
3. **Logical Page IDs Decoupled from Physical Layout (`KvLayerLayout`):**
   - Logical page allocation (`B = 16` tokens/page) is shared across namespaces while each layer resolves its own physical format (`FP16`, `BF16`, `Q8_0`, `TQ4`, `DFlashDraftContext`, or `QsaIndexer`).
4. **End-to-End `FP16` Spine + Composite CUDA Graphs:**
   - All target hidden states, norms, RoPE, attention, GDN recurrence I/O, and Marlin GEMMs consume and produce `FP16` directly (`0` `f32 ↔ f16` glue casts), while GDN `ssm_state` (`[H_v, 128, 128]`) and softmax/norm reductions stay `FP32`.
   - Decode and speculative verify steps are captured into composite CUDA Graphs with valid-row masking (`valid_token_mask` / `valid_req_mask`) and 2-slot pipelined host overlap (`~20 µs` inter-graph gap).

---

## Deep Dive 5: Serving, Structured Outputs, & Observability

### Multi-protocol HTTP APIs

A single `w34 serve` instance exposes:
- **OpenAI Chat Completions (`POST /v1/chat/completions`)** & **Completions (`POST /v1/completions`)**
- **OpenAI Responses (`POST /v1/responses`, `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`)**
- **Anthropic Messages (`POST /v1/messages`, `POST /v1/messages/count_tokens`)** — including streaming `thinking` and `tool_use` blocks
- **Decision APIs (`POST /v1/systemone`, `POST /v1/decisions`)**
- **Tokenization & Health (`POST /tokenize`, `POST /detokenize`, `GET /v1/models`, `GET /health`)**

### Reasoning, tool calls, and `<think>`-aware constrained decoding

- **9-step GPU sampler:** Handles repetition/presence/frequency penalties, `logit_bias`, `min_tokens`, `temperature`, `min_p`, `top_k`, `top_p`, Gumbel-max sampling, and `top_logprobs` entirely on the GPU.
- **Reasoning & Tool Parsers:** Native support for `<think>...</think>` blocks (`enable_thinking`, `reasoning_effort` / `output_config.effort`) and both **Qwen3-XML** and **Hermes-JSON** tool call formats (`tool_choice: "auto" | "required" | {"function": ...}`).
- **Structured Outputs (`llguidance`):** Enforces **JSON Schema**, **Regex**, **Context-Free Grammars (Lark)**, and **Structural Tags** via GPU token bitmasks. When reasoning is enabled, grammar enforcement automatically waits until `</think>` closes so the model can think freely before emitting strictly valid JSON or grammar-constrained output. And when speculative decoding rejects draft tokens, the grammar state machine rolls back in lockstep with KV and GDN state.

### Authentication & concurrency controls

Authentication is off by default for local ergonomics. For deployments, pass an environment variable containing your secret key:

```sh
export W34_API_KEY="your-secret-key"
./target/release/w34 serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ \
  --api-key-env W34_API_KEY \
  --served-model-name coding-model \
  --max-num-seqs 8 --kv-pages 2048 --max-model-len 32768
```

Clients can authenticate with either `Authorization: Bearer <key>` or `x-api-key: <key>`.

### Live dashboard & Prometheus metrics

- **`GET /`** serves the embedded warm-dark React dashboard (bundled directly into the `w34` binary—no Node runtime needed at serving time).
- **`GET /metrics`** exports Prometheus counters, gauges, and latency histograms; **`GET /api/metrics`** serves the same telemetry as versioned JSON.

---

## Docker

**Packaging choice:** CUDA libraries are shared and copied into a separate layer before the executable, so routine application updates can reuse the cached library download. Pruned static cuBLAS saved space on first install, but made every changed executable layer much larger; see [the experiment and decision record](docs/CUBLAS_PACKAGING_EXPERIMENT.md). Actual RTX 4090 validation remains pending with a separate agent.

Images are built and published to GHCR on every push to `main` (`ghcr.io/nchapman/warp-34:dev` and `:sha-<commit>`) and on `v*` release tags (`:latest`, `:X.Y.Z`, `:X.Y`). The multi-stage build ships `w34`, shared `libcudart`/`libcublas`/`libcublasLt`, and a small Ubuntu 26.04 runtime with TLS certificates and a curl healthcheck—no Python, Node, CUDA toolkit, or model weights. Embedded CUDA code is size-compressed; native `sm_86` and `sm_89` kernels are included in the same image.

**Host requirements:** Linux `amd64` with an x86-64-v3 CPU (AVX2/FMA), an RTX 30- or 40-series GPU with enough VRAM for the selected model, a CUDA 13-capable NVIDIA driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). CUDA 13.x minor-version compatibility starts at driver branch R580; a current driver is recommended because newer CUDA features/PTX can require newer drivers. The host does **not** need the CUDA toolkit installed. The container checks the CUDA 13 driver capability before startup.

```sh
docker pull ghcr.io/nchapman/warp-34:dev
# Or build locally:
docker build --platform linux/amd64 -t warp-34:local .
```

Mount your host Hugging Face cache to run offline or let it download on first use:

```sh
docker run --rm --name w34-local --gpus '"device=0"' \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_CACHE=/models \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src="$HOME/.cache/huggingface/hub",dst=/models,readonly \
  warp-34:local serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ --offline --host 0.0.0.0
```

For download-on-first-use, use a persistent writable named volume instead of the read-only bind mount, and omit `--offline`:

```sh
docker run --rm --name w34-local --gpus '"device=0"' \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=w34-models,dst=/home/w34/.cache/huggingface/hub \
  --health-start-period=10m \
  warp-34:local serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ --host 0.0.0.0
```

The image defaults to non-root `w34`. The read-only host-cache example instead uses your host UID/GID because Hub metadata can be owner-readable only; it also moves `HOME` and the cache outside `w34`'s private home directory. A new named volume inherits the image's cache-directory ownership and works with the default user. Adjust `--health-start-period` for slow downloads or large models. The default healthcheck probes port 8000; if you change the server port, also override `--health-cmd`.

For two GPUs, pass `--gpus '"device=0,1"'` and `--tp 2` (because `warp-34` uses pinned memory / P2P rather than NCCL, you don't even need `--ipc=host`).

Local rebuilds reuse BuildKit caches for pnpm, Cargo downloads, and compiled Rust/CUDA objects. CI exports ordinary build layers to the GitHub Actions cache; cache-mount contents are local to the builder and are not included in that export.

---

## Development, Verification Oracles, & Benchmarks

```sh
export PATH="/opt/cuda/bin:$PATH"
mise run build           # build dashboard + release binary (--features cuda)
mise run check-quick     # rustfmt + clippy (with and without cuda) + rustdoc check
mise run check           # fmt + clippy + rustdoc + full release GPU test suite
mise run test-cpu        # fast CPU-only unit tests
mise run coverage        # library line coverage (--lib --features cuda)
mise run perf-check      # validate the standalone HTTP benchmark suite
```

Beyond unit and integration tests, `w34` has three built-in oracle subcommands used to verify every kernel and subsystem against `vLLM` and `SGLang`:
- **`w34 dump-diff`** — Hooks per-layer tensors from `vLLM`/`SGLang` golden dumps and checks `max_abs_err`, `rel_l2`, and cosine similarity layer by layer.
- **`w34 greedy-diff`** — Verifies exact greedy token parity, lossless speculative decoding (`--verify-lossless`), and warm radix cache restoration (`--verify-radix-warm`).
- **`w34 profile-diff`** — Captures `nsys` SQLite traces of `warp-34` vs. `vLLM`/`SGLang` and prints a side-by-side `ms/round` kernel scoreboard by category (`marlin_gemm`, `full_attn`, `gdn_ssm`, `norms_rope_act`, `tp_comm`, `spec_decode`, `host_gap`).

For end-to-end HTTP load testing across single-stream decode, concurrent `C=8..16` throughput, prefix reuse, and multi-turn coding-agent replays up to `252K` context, see the standalone benchmark suite in [`benchmarks/perf/README.md`](benchmarks/perf/README.md) and the recorded standings in [`benchmarks/perf/campaigns/standings.md`](benchmarks/perf/campaigns/standings.md).

---

## Documentation Map

| Guide | What's Inside |
|---|---|
| [`docs/MODELS.md`](docs/MODELS.md) | Model downloads, `--revision` / `--offline`, supported architectures, quantizations, and KV policies |
| [`docs/QWEN_FLASH_NEXT.md`](docs/QWEN_FLASH_NEXT.md) | Full architecture & 3-tier offloading reference for `Qwen3.8-Flash-Next` (`qwen4_exp`) |
| [`docs/DECISION_MODELS.md`](docs/DECISION_MODELS.md) | `/v1/systemone` and `/v1/decisions` schemas, supported profiles (`Clef`, `Kev`, `Nimble`, `Tev1`, `OpenJev`, `Lev`/`Xor`), and confidence math |
| [`docs/GENERATIVE_APIS.md`](docs/GENERATIVE_APIS.md) | Wire contracts for `/v1/chat/completions`, `/v1/responses`, and Anthropic `/v1/messages` |
| [`docs/SERVING.md`](docs/SERVING.md) | Server flags, authentication (`--api-key-env`), concurrency limits, and image security |
| [`docs/METRICS.md`](docs/METRICS.md) | Live web dashboard, Prometheus `/metrics`, and `/api/metrics` JSON schema |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Subsystem architecture, donor map (`vLLM` / `SGLang` / `vllm.cpp` / `warp3090`), and seam contracts |
| [`docs/HARDWARE.md`](docs/HARDWARE.md) | RTX 30-series (`sm_86`) and 40-series (`sm_89`) rooflines, PCIe/NVLink topology, and ISA traps |
| [`docs/STATE.md`](docs/STATE.md) | Current verified hardware benchmark matrix and engineering state |
| [`docs/contracts/`](docs/contracts/) | Frozen precision, KV layout, GDN state-transaction, and multi-GPU collective contracts |
