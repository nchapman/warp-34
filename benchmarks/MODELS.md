# Benchmark checkpoint policy

## Active model selection

- **warp-34 quantized targets:** `nchapman/*-HGPTQ*` (normally `W4A16-HGPTQ`).
  Do not substitute the deprecated `nchapman/*-W4A16-GPTQ` exports.
- **vLLM / SGLang quantized Qwen baselines:** cyankiwi quants of the **official
  Qwen base**, not uncensored, abliterated, or otherwise fine-tuned derivatives.
- Unquantized upstream models used for correctness/oracle tests remain valid.
  Format-loader coverage is not a benchmark model endorsement.
- Pin the checkpoint and tokenizer revisions. Keep the API `model` alias,
  launch path, quantization metadata, and receipt consistent. Never inherit
  warp-34's artifact hashes or HGPTQ metadata into a competitor profile.

| Family | vLLM / SGLang checkpoint | Revision | Weight format |
|---|---|---|---|
| Qwen3.8-27B | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | `e3a700906e6a73ba4290b1873dc2ba8198527ab5` | AWQ-calibrated asymmetric INT4, group 32; compressed-tensors `pack-quantized` |
| Qwen3.6-35B-A3B | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | `f1e8415c4f29508fe281da4e256df26573213bfd` | Asymmetric AWQ INT4, group 32; AWQ `gemm` packaging |

The 27B index contains 15 unquantized MTP tensors. The 35B index contains 2,321
MTP tensors, including quantized routed experts; attention and shared-expert
projections are excluded from quantization. The model card's statement that a
base was trained with MTP is **not** proof of serving compatibility. Test each
engine/checkpoint/speculation combination before claiming support.

Download the exact competitor snapshots before using offline launch profiles:

```sh
hf download cyankiwi/Qwen3.8-27B-AWQ-INT4 --revision e3a700906e6a73ba4290b1873dc2ba8198527ab5
hf download cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit --revision f1e8415c4f29508fe281da4e256df26573213bfd
mise run test-prep-all  # pinned warp-34 HGPTQ and correctness fixtures
```

Use the pins in each campaign for warp-34; `tools/prep_test_models.py` prepares
the core test pins. The quality suite has its own explicitly documented HGPTQ
pin in [quality/AUTOMATION.md](quality/AUTOMATION.md).

## Serving smoke evidence

Both pins passed [vLLM 0.31.0 TP2 plain/MTP-3 startup and short generation](perf/qualification/cyankiwi-vllm-tp2-smoke.md)
on 2× RTX 3090 Ti. This is not a throughput, quality, cache-reuse, TP1, or SGLang
qualification. Updated campaign and quality-suite configurations still require
exact-setting qualification.

## Comparing results

HGPTQ versus cyankiwi AWQ is a **practical deployment comparison**, not an
identical-weight engine benchmark. Quantization, KV precision, memory capacity,
and speculation can all affect throughput and generated tokens. Report those
settings and qualify output quality separately. Do not call a configuration the
"strongest" without testing the relevant supported alternatives.

**The multi-agent vLLM comparisons recorded with commit `f8bf8a3` are withdrawn
as approved baselines.** The 35B run used an uncensored derivative, and the 27B
run used a deprecated in-house GPTQ export. Their speedup and cache-hit ratios
must not be attributed to cyankiwi or used to claim that the corrected baseline
has been beaten. Fresh measurements are required.

Keep archived results, logs, and qualification receipts byte-for-byte faithful
to the checkpoints actually used. Historical references in those records are
not runnable defaults or endorsements. Do not search-and-replace model IDs in
old evidence. Updated profiles stay unverified until their exact configuration
is requalified; a short serving smoke test is not a quality or throughput result.

The CPU-only model-policy tests in `perf/tests/test_model_policy.py` and
`quality/tests/test_runner.py` guard active configuration selection and metadata.
