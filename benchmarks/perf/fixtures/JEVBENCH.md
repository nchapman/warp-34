# JevBench decision-model workload fixtures

This directory contains two `kind = "decision"` (`POST /v1/systemone`) benchmark
fixtures derived from the public MIT-licensed
[JevBench](https://github.com/fstandhartinger/jevbench) suite (data card:
[https://benchmarkheaven.com/jev-models/data](https://benchmarkheaven.com/jev-models/data))
at commit `609a664e81f46e03731600bd7bd9f1b7a61257c5`:

| Upstream file | SHA-256 | Items |
|---|---|---:|
| `datasets/public/original.jsonl` | `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180` | 72 |
| `datasets/public/easy.jsonl` | `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b` | 48 |
| `datasets/public/hard.jsonl` | `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb` | 111 |

## 1. Single-question suite (`jevbench_public.json`)

Contains **128** single-question `/v1/systemone` requests:

- All **72** items from `datasets/public/original.jsonl` (`policy`, `intent`,
  `ordinal`, `extraction`, `adequacy`, `routing`; 24 `noul`, 36 `choice`, 12
  `score`).
- **16** items from `datasets/public/easy.jsonl` (`extraction`, `fact`,
  `intent`, `tool_selection`; 4 per family).
- **40** items from `datasets/public/hard.jsonl` (`adversarial`, `ambiguous`,
  `judge_hard`, `long_policy`, `multi_hop`, `probability`, `routing_hard`,
  `temporal_numeric`, `tradeoff`, `trap`; 4 per family).

Each item preserves its upstream `id`, `tier`, `family`, `state`,
`questions.decision` (`type`, `instructions`, `criteria`), `labels`, and
ground-truth `expected.decision`.

## 2. Multi-question shared-state suite (`jevbench_multiquestion.json`)

Contains **16** multi-question `/v1/systemone` requests (**4** questions per
request = **64** decisions total), constructed by grouping 4 consecutive items
from `datasets/public/original.jsonl` into a shared structured `state` object
and asking all 4 typed questions in a single `/v1/systemone` request. This
exercises:

- `vllm-sr/d3-{turbo,ultra}` `prefix_bidir` state-prefix reuse,
- `fastino/GLiDE-*` and `CausalLm` synchronous multi-query wave submission, and
- `order_passes = 2` forward+reverse option permutation averaging.
