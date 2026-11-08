# Historical mixed-KV single-GPU 27B tooling checks

**Historical evidence:** these completed checks used TQ4 for warp-34. The new
cross-engine baseline uses FP16 KV throughout; see [automation](AUTOMATION.md).
Do not relabel the results below as FP16 results. Commands here preserve the
original configurations.

The primary user-facing configuration is **one 24GB RTX 3090, a 4K context,
one active evaluation request, and 10 examples per task**, across warp-34,
llama.cpp, vLLM and SGLang. These are short workflow checks, not comparative
quality scores or throughput measurements. The previous [TP2 recipes](servers.md)
and [qualification evidence](QUALIFICATION.md) remain historical records.

All commands bind to localhost, expose the alias `warp34-quality`, and restrict
CUDA visibility to GPU 0. Run one engine at a time. Do not silently add another
GPU, move transformer layers to CPU, or shrink context to make an unsuccessful
configuration appear successful. Record startup failures separately from retries.

## Verified outcome

All four serving paths completed both native tasks: **80 evaluated requests**,
10 distinct documents per task per engine, no empty responses, and all eight
native result/sample bundles produced. This does not mean every answer was
correct. SGLang's successful run is explicitly **plain**, not MTP.

| Engine | Successful mode | Observed GPU allocation |
|---|---|---:|
| warp-34 | MTP-3, TQ4 | 22,164 MiB (21.6 GiB) |
| llama.cpp / Unsloth | Native MTP, max 3 draft tokens | 18,278 MiB (17.8 GiB) |
| vLLM | MTP-3, 0.95 GPU utilization | 23,032 MiB (22.5 GiB) |
| SGLang | Plain, 0.95 static memory fraction | 23,639 MiB (23.1 GiB) |

These are point-in-time `nvidia-smi` allocation snapshots, not peak-memory or
memory-efficiency benchmarks; they include differently sized cache reserves.
GPU 1 remained unused. Servers were stopped after verification and the pod GPU
process list was empty. Local GPUs were not used.

## Checkpoints and builds

| Engine | Checkpoint | Exact revision |
|---|---|---|
| warp-34 (merged branch `9b04799`, engine tree `f6ab96e`) | `nchapman/Qwen3.8-27B-W4A16-HGPTQ` | `1de743eedefdb7422e6b171174c7080204f5b005` |
| vLLM 0.31.0 / SGLang 0.5.19 | `nchapman/Qwen3.8-27B-W4A16-GPTQ` | `0df81eadcfd59601cb82ccece2f16ea7d39244bf` |
| llama.cpp v0.9.0 (`cd959bae99ff4bed22bef2604f45241c09bbb8c8`) | `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |

The user-selected llama.cpp reference is **Unsloth UD-Q4_K_XL**, not ggml-org's
GGUF. Use `Qwen3.8-27B-UD-Q4_K_XL.gguf` and its same-repository native MTP export
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`. The MTP file is the model's built-in head, not a
separately trained DFlash2 drafter. No abliterated/heretic checkpoints are used.

Quantization and KV policies differ by engine, so this is a practical migration
configuration, not a matched-quant quality comparison. Preserve each engine's
package freeze, command, artifact identity and logs as in the main README.

## warp-34

Set `HGPTQ` to the pinned local snapshot path above:

```sh
CUDA_VISIBLE_DEVICES=0 ./target/release/w34 serve \
  --model "$HGPTQ" --offline --tp 1 --kv-format tq4 --kv-pages 256 \
  --max-num-seqs 4 --speculative mtp --mtp-k 3 \
  --served-model-name warp34-quality --host 127.0.0.1 --port 18034
```

`--max-num-seqs 4` reserves internal state capacity; the client still submits
only one request at a time. The server reports a 4,096-token context limit.

## vLLM

Use the isolated installation in [servers.md](servers.md), with `GPTQ` pointing
to the regular-GPTQ snapshot above:

```sh
CUDA_VISIBLE_DEVICES=0 /workspace/vllm-venv/bin/vllm serve "$GPTQ" \
  --served-model-name warp34-quality --dtype half --tensor-parallel-size 1 \
  --language-model-only --max-model-len 4096 --max-num-seqs 1 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.95 \
  --mamba-ssm-cache-dtype float32 --generation-config vllm --seed 34 \
  --cudagraph-capture-sizes 1 2 4 8 --host 127.0.0.1 --port 18034 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The first attempt at 0.90 utilization failed its KV budget check (0.84 GiB
available versus 1.09 GiB needed for 4K). The 0.95 retry preserved the 4K context
and MTP-3 and completed both short tasks. The failed receipt remains separate.

## SGLang — explicitly plain decoding

The single-GPU native-MTP attempt at 0.95 static memory fraction failed before
serving: target plus draft weights left no KV-cache budget (the loader estimated
more than 0.996 merely to obtain any cache room). The fallback below removes MTP,
not context or GPU-residency requirements. This is **not** an MTP qualification.

```sh
CUDA_VISIBLE_DEVICES=0 SGLANG_MAMBA_CONV_DTYPE=float16 \
  /workspace/sglang-venv/bin/python -m sglang.launch_server \
  --model-path "$GPTQ" --served-model-name warp34-quality \
  --dtype float16 --tp-size 1 --context-length 4096 \
  --mamba-ssm-dtype float32 --mem-fraction-static 0.95 \
  --chunked-prefill-size 1024 --max-running-requests 1 --max-mamba-cache-size 8 \
  --cuda-graph-bs-decode 1 2 4 8 --cuda-graph-bs-prefill 512 1024 \
  --reasoning-parser qwen3 --enable-cache-report --enable-metrics \
  --random-seed 34 --host 127.0.0.1 --port 18034
```

## llama.cpp / Unsloth

Build the pinned upstream tag with CUDA for the 3090 (`sm_86`):

```sh
git clone --branch v0.9.0 --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build llama.cpp/build --target llama-server -j 8
```

Set `GGUF` to the pinned Unsloth snapshot directory:

```sh
CUDA_VISIBLE_DEVICES=0 llama.cpp/build/bin/llama-server \
  --model "$GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf" --alias warp34-quality \
  --ctx-size 4096 --parallel 1 --n-gpu-layers all --split-mode none --fit off \
  --host 127.0.0.1 --port 18034 --metrics --jinja --threads 16 \
  --spec-type draft-mtp --spec-draft-model "$GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf" \
  --spec-draft-ngl all --spec-draft-n-max 3
```

`--fit off` and explicit all-layer offload prevent automatic fitting from
silently changing this configuration. Check target/draft offload logs, not just
CUDA availability. The HTTP listener can start before model loading completes;
verify `/health` is ready before invoking lm-eval.

A separate startup-only check with `--verbosity 5` confirmed the loader reported
**66/66 layers offloaded to GPU** for both target and MTP contexts. It also
reported normal CPU-mapped model buffers (682.03 MiB target, 521.00 MiB draft),
plus small host output/compute buffers. Thus "all-layer offload" does **not** mean
zero CPU-side model memory; these buffers are explicitly retained in the evidence.
The startup-only check did not add evaluation requests or replace the original run.

## Same upstream client for all four

With the isolated client environment installed and the server ready:

```sh
export NLTK_DATA="$HOME/nltk_data"
OUT="benchmarks/quality/results/$(date -u +%Y%m%dT%H%M%SZ)-engine-27b-single-smoke10"
test ! -e "$OUT" && mkdir -p "$OUT" || exit 1
benchmarks/quality/.venv/bin/lm-eval run \
  --config benchmarks/quality/lm-eval.yaml --limit 10 \
  --output_path "$OUT/upstream" > "$OUT/lm-eval.log" 2>&1
```

The shared YAML now records a 4K context; this is not client-side token counting
or truncation with `tokenizer_backend: null`. Upstream IFEval/GSM8K generation
budgets, few-shot setup, filtering and scoring are unchanged. Qwen thinking is
disabled by the same native request parameter for each server. Do not substitute
provider-recommended sampling settings for upstream task settings.

## Evidence and limitations

Native results and run metadata are copied unchanged under `qualification/`.
Complete raw bundles (including sample JSONL, server logs, metrics, effective
4K configs and package freezes) are retained under gitignored `results/` and
on the pod under `/workspace/`, using these directory names:

- `quality-27b-single-mtp3-smoke10-001`
- `quality-vllm-27b-single-mtp3-smoke10-001` — failed 0.90 KV-budget attempt
- `quality-vllm-27b-single-mtp3-smoke10-002` — successful 0.95 retry
- `quality-sglang-27b-single-mtp3-smoke10-001` — failed MTP fit attempt
- `quality-sglang-27b-single-plain-smoke10-001` — successful plain fallback
- `quality-llama-27b-single-mtp3-smoke10-001`

`qualification/single-gpu-evidence.sha256` records hashes of these bundles,
including failed attempts. The llama.cpp metadata includes GGUF and executable
SHA-256 hashes; its offload-check command/log is retained separately. A lost SSH
build-tracking session was recovered by stopping the orphaned build and resuming
incrementally; no llama.cpp source modifications were made.

warp-34 reports 20 completed requests and zero failures. vLLM reports 20 successes
(18 stop, two length finishes; zero abort/error finishes). SGLang logs 21 HTTP-200
chat calls: its one built-in startup warmup plus the 20 evaluation requests.
llama.cpp logged 20 completed slots with `truncated=0`, no remaining processing
or deferred requests, and nonzero native-MTP accepted-token counters. Its default
metrics do not provide the same HTTP-error/finish-reason counters as vLLM; do not
claim equivalent telemetry coverage. Upstream token-budget finishes remain scored
unchanged. vLLM and SGLang emitted cleanup warnings during deliberate shutdown,
after scoring completed; logs are preserved.

These observations qualify only these pinned short serving paths. They do not
establish close quality parity, validate longer contexts or concurrency, or
constitute a full-dataset evaluation. Unsloth's template/quantization and each
server's KV/speculation implementation differ. MTP did not fit in the tested
SGLang configuration; that is not a claim that every SGLang 27B/MTP configuration
is impossible on 24GB.
