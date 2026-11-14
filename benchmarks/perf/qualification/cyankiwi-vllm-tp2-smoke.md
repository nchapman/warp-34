# cyankiwi AWQ / vLLM TP2 serving smoke

This is **startup and HTTP generation compatibility only**, not a throughput,
quality, MTP acceptance, prefix-hit, or concurrent-serving qualification.

## Result

All four configurations started and served two successful chat completions:

| Checkpoint | Plain | Native MTP-3 |
|---|---|---|
| `cyankiwi/Qwen3.8-27B-AWQ-INT4@e3a700906e6a73ba4290b1873dc2ba8198527ab5` | pass | pass |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit@f1e8415c4f29508fe281da4e256df26573213bfd` | pass | pass |

[Raw receipt](cyankiwi-vllm-tp2-smoke.json) retains exact commands and HTTP replies.
Both responses in every configuration were `4` (two completion tokens including
termination). Success checks required model discovery, nonempty response content,
and positive completion-token usage. The checkpoint revision is a pinned local
snapshot path, not a moving Hub `main`.

## Setup

- Engine: locally installed vLLM **0.31.0**, no source patches for this probe.
- Hardware: **2× RTX 3090 Ti**, `CUDA_VISIBLE_DEVICES=0,1`, TP2, serial server runs.
- Precision: FP16 activations/native KV, FP32 Mamba SSM.
- Limits: 16,384 context; 8 configured sequences; 2,048 batched tokens; GPU budget
  0.90; CUDA graph capture sizes 1, 2, 4, 8.
- Prefix caching and prompt-token details explicitly enabled. These very short
  requests reported zero cached tokens; they do not exercise useful prefix reuse.
- MTP probe adds `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
- Offline HF/Transformers; HTTP port 18836; `/v1/models` checked before generation.
- Request, repeated sequentially twice per server:

```json
{
  "model": "cyankiwi-smoke",
  "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
  "max_tokens": 32,
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Local startup logs and the probe script are retained under
`benchmarks/perf/results/cyankiwi-vllm-tp2-smoke/` (ignored raw artifacts).
Each owned server process group was stopped before the next run.

## Limits

Do not transfer this result to TP1, SGLang, DFlash2, 32K contexts, long prompts,
MTP acceleration, or quality-suite recipes with different launch settings.
Campaign profiles remain `unverified` until their exact workloads/configurations
are requalified. The previous deprecated-GPTQ/uncensored head-to-head numbers
are not cyankiwi results; fresh practical HGPTQ-versus-AWQ measurements are still
required. See the [checkpoint policy](../../MODELS.md).
