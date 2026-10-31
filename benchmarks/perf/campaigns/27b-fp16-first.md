# First 27B FP16 comparison — 2026-10-30

One RTX 3090 Ti, newer `0df81ead…` GPTQ weights, FP16, no speculation.
Three repetitions and **96 measured requests per engine**. See the [campaign](README.md) for settings and reproduction.

## Aggregate output throughput

| Workload | warp-34 | vLLM 0.31.0 | SGLang 0.5.19 |
|---|---:|---:|---:|
| Sequential decode C1 | 45.5 tok/s | 50.7 tok/s | 51.7 tok/s |
| Throughput C8 | 287.0 tok/s | 309.1 tok/s | 226.6 tok/s |
| Four-turn agent C1 | 39.4 tok/s | 41.1 tok/s | 45.7 tok/s |
| Primed ~3k prefill C1 | 44.7 tok/s | 42.9 tok/s | 50.6 tok/s |

## Median first-output latency

| Workload | warp-34 | vLLM | SGLang |
|---|---:|---:|---:|
| Sequential decode C1 | 182.4 ms | 138.7 ms | 146.1 ms |
| Throughput C8 | 733.8 ms | 910.1 ms | 774.3 ms |
| Four-turn agent C1 | 227.3 ms | 452.8 ms | 185.7 ms |
| Primed ~3k prefill C1 | 35.0 ms | 492.9 ms | 81.1 ms |

## Median complete agent session

- **warp-34: 11.90 s**
- **vLLM: 8.71 s**
- **SGLang: 7.91 s**

## Run settings and evidence

- All engines ran serially on GPU 0; 256-token decode/throughput ceilings and 128-token agent/prefill ceilings.
- SGLang uses cache off for the unique-prompt C1/C8 cases, then cache on for the C1 agent/prefill cases. Its eight-active cached attempt OOMed and is retained, not mixed into the comparison.
- warp-34 uses its built-in ChatML renderer; vLLM and SGLang use the artifact template.
- Throughput = total reported output tokens / total measured duration over all three repetitions.
- TTFT = median client first-output latency across measured requests; agent time includes all four turns and configured think time.
- [Machine-readable results and evidence hashes](27b-fp16-first.results.json); [CSV](27b-fp16-first.results.csv).
- Full raw bundles, logs, preflight receipts and one-second GPU samples: `benchmarks/perf/results/27b-fp16-first-2026-10-30/`.
