# Comparison campaigns

## 27B FP16 — first comparison

- Configuration: [27b-fp16-first.toml](27b-fp16-first.toml)
- Results: [27b-fp16-first.md](27b-fp16-first.md)
- Machine: one RTX 3090 Ti, GPU 0; engines run serially.
- Model: `nchapman/Qwen3.8-27B-W4A16-HGPTQ`, snapshot
  `0df81eadcfd59601cb82ccece2f16ea7d39244bf`.
- Precision: GPTQ INT4 weights, FP16 activations/KV/convolution history,
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
OOMed during C8; that attempt is retained separately. Exact commands are in
the TOML.

### Reproduce

From the repository root, launch the selected profile's `launch_command`.
Run one engine at a time, wait for readiness, then:

```sh
cd benchmarks/perf
python3 -m w34perf preflight campaigns/27b-fp16-first.toml \
  --engine sglang27-fp16-uncached --inventory qualification/27b-gptq-0df.inventory.json \
  --out results/my-comparison-sglang-preflight.json
python3 -m w34perf run campaigns/27b-fp16-first.toml \
  --engine sglang27-fp16-uncached --gpus 1 --case decode-c1 --case throughput-c8 \
  --out results/my-comparison-sglang
python3 -m w34perf report results/my-comparison-sglang
```

Repeat with `sglang27-fp16-cached --case agent-c1 --case prefill-primed-c1` for
the cached SGLang cases, then `vllm27-fp16` and `warp34-fp16` without case filters.
Use fresh output paths for each profile.

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
