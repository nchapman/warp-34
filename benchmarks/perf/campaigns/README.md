# Comparison campaigns

Active configurations follow the [checkpoint policy](../../MODELS.md): nchapman
HGPTQ targets versus official-base cyankiwi AWQ competitors. Changed profiles
require new qualification; archived metrics do not transfer to the new weights.

## 27B FP16 — refreshed configuration, historical results

- Configuration: [27b-fp16-first.toml](27b-fp16-first.toml)
- Machine: one RTX 3090 Ti, GPU 0; engines run serially.
- Models: nchapman HGPTQ for warp34; cyankiwi AWQ for vLLM/SGLang, pinned
  in the TOML. This is a practical cross-quant comparison, not identical weights.
- Precision: INT4 weights, FP16 activations/KV/convolution history,
  FP32 recurrent SSM; no speculation or CPU offload.
- Three measured repetitions, one warmup per case/repetition.

| Case | Work per repetition | Output ceiling |
|---|---|---:|
| `decode-c1` | 4 sequential requests | 256 tokens |
| `throughput-c8` | 16 requests, concurrency 8 | 256 tokens |
| `agent-c1` | 2 sequential four-turn agent sessions | 128 tokens/turn |
| `prefill-primed-c1` | 4 primed requests, ~3k input tokens | 128 tokens |

Each engine receives **96 measured requests** across 12 trials. Warmup and
priming are retained separately. SGLang uses two profiles: eight active requests
with radix caching off for unique-prompt decode/throughput, then its cached
two-active profile for the C1 agent/prefill cases. Its eight-active cached mode
OOMed during C8 with the historical checkpoint; that attempt is retained
separately and says nothing about AWQ capacity. Current candidate commands are
in the TOML; original commands remain in archived receipts.

### Run a new comparison

From the repository root, launch the selected profile's `launch_command`.
Run one engine at a time, wait for readiness, then:

```sh
cd benchmarks/perf
python3 -m warp34perf inventory /absolute/path/to/pinned/cyankiwi/snapshot \
  --out results/cyankiwi-27b.inventory.json
python3 -m warp34perf preflight campaigns/27b-fp16-first.toml \
  --engine sglang27-fp16-uncached --inventory results/cyankiwi-27b.inventory.json \
  --out results/my-comparison-sglang-preflight.json
python3 -m warp34perf run campaigns/27b-fp16-first.toml \
  --engine sglang27-fp16-uncached --gpus 1 --case decode-c1 --case throughput-c8 \
  --out results/my-comparison-sglang
python3 -m warp34perf report results/my-comparison-sglang
```

Repeat with `sglang27-fp16-cached --case agent-c1 --case prefill-primed-c1` for
the cached SGLang cases, then `vllm27-fp16` and `warp34-fp16` without case filters.
Use fresh output paths for each profile and a separate HGPTQ inventory for
warp34. Do not reuse retired GPTQ inventories for either checkpoint.

### Reported metrics

- **Aggregate output tok/s:** total server-reported completion tokens divided by
  total measured duration across the three repetitions, including request,
  prefill and agent think-time overhead.
- **TTFT:** median client-observed first-output latency over measured requests.
- **TPOT estimate:** median per-request output-span / (`completion_tokens - 1`).
- **Agent session:** median complete four-turn session duration.

Raw requests, SSE chunks, manifests, summaries, receipts and startup logs stay
under `benchmarks/perf/results/27b-fp16-first-2026-10-30/`. GPU clocks, power,
utilization, memory and temperature are sampled once per second in `gpu.csv`.
