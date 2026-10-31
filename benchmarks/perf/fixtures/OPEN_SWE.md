# Open-SWE-derived workload fixtures

These are **synthetic performance replays assembled from real tool observations**,
not faithful task trajectories and not a SWE correctness benchmark. They combine
material from different repositories, repeat corpus material at the largest size,
and retain one initial issue prompt. Generated responses are preserved in history;
recorded tool observations are appended regardless of the generated plan. No tool
commands execute. Every agent-replay request uses `tool_choice: "none"`;
standalone text cold-read requests send no tools and omit `tool_choice`.

## Attribution and source inventory

Source: NVIDIA, [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces),
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Repository excerpts retain the upstream licenses listed below (as reported by the
dataset). Changes: selection, concatenation, line-boundary slicing, repository
headers, repeated blocks, and synthetic tool-call wrappers/system prompt.
No endorsement by NVIDIA or upstream authors is implied.

Rows were retrieved through the Hugging Face datasets-server `/rows` API. The
original extraction did **not** record a dataset commit; these checked-in fixture
bytes, not a claim of reproducible retrieval from a moving dataset, are the frozen
workload. Config/split/row offsets and instance IDs identify the sampled pool:

| Config / split / offset | Instance ID | Repository / reported license |
|---|---|---|
| v2.0 / claudecode / 0 | raimon49_pip-licenses_pr64 | [raimon49/pip-licenses](https://github.com/raimon49/pip-licenses) / MIT |
| v2.0 / openhands / 0 | saltstack_libnacl_pr89 | [saltstack/libnacl](https://github.com/saltstack/libnacl) / Apache-2.0 |
| v2.0 / openhands / 1 | marcosschroh_dataclasses-avroschema_pr780 | [marcosschroh/dataclasses-avroschema](https://github.com/marcosschroh/dataclasses-avroschema) / MIT |
| v1.2 / claudecode / 0 | demisto_demisto-sdk_pr58 | [demisto/demisto-sdk](https://github.com/demisto/demisto-sdk) / MIT |
| v2.0 / claudecode / 1 | tefra_xsdata_pr902 | [tefra/xsdata](https://github.com/tefra/xsdata) / MIT |
| v2.0 / claudecode / 2 | more-itertools_more-itertools_pr630 | [more-itertools/more-itertools](https://github.com/more-itertools/more-itertools) / MIT |
| v2.0 / claudecode / 3 | pyvisa_pyvisa_pr550 | [pyvisa/pyvisa](https://github.com/pyvisa/pyvisa) / MIT |
| v2.0 / claudecode / 4 | gruebel_pycep_pr180 | [gruebel/pycep](https://github.com/gruebel/pycep) / Apache-2.0 |
| v2.0 / claudecode / 5 | pyapp-kit_app-model_pr213 | [pyapp-kit/app-model](https://github.com/pyapp-kit/app-model) / BSD-3-Clause |
| v2.0 / claudecode / 6 | maxcountryman_flask-login_pr403 | [maxcountryman/flask-login](https://github.com/maxcountryman/flask-login) / MIT |
| v2.0 / claudecode / 7 | k2-fsa_sherpa-onnx_pr710 | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) / Apache-2.0 |
| v2.0 / claudecode / 8 | fonttools_fonttools_pr1676 | [fonttools/fonttools](https://github.com/fonttools/fonttools) / MIT |
| v2.0 / claudecode / 9 | fonttools_fonttools_pr3092 | [fonttools/fonttools](https://github.com/fonttools/fonttools) / MIT |
| v2.0 / claudecode / 10 | django-crispy-forms_django-crispy-forms_pr1116 | [django-crispy-forms/django-crispy-forms](https://github.com/django-crispy-forms/django-crispy-forms) / MIT |

The initial user prompt comes from `raimon49_pip-licenses_pr64`. Warm observations
were selected from 220–1100-token tool results; larger observations were concatenated
into a line-sliced cold corpus. Source-to-excerpt offsets were not retained, so this
inventory describes the source pool rather than a complete per-line mapping.

## Token lengths and measurement limits

Calibrated with the tokenizer and Jinja template from Qwen3.8-27B GPTQ revision
`0df81eadcfd59601cb82ccece2f16ea7d39244bf`, tokenizer truncation disabled. Synthetic
assistant responses were used only to estimate progression. Actual prompt lengths
are measured from each engine's usage and vary with generated responses.

| Fixture | Turns | Estimated first → last prompt tokens |
|---|---:|---:|
| `open_swe_agent_64k.json` | 9 | 15,508 → 59,222 |
| `open_swe_agent_120k.json` | 10 | 26,672 → 118,004 |
| `open_swe_agent_252k.json` | 9 | 54,515 → 251,999 |
| `open_swe_cold_read_60k.json` | 1 | 57,589 (repeat context twice: 115,083) |

The size names describe capacity tiers, **not exact 65,536/131,072/262,144-token
prompts**. The separate `needle_gate` uses a fixed 262,144-token window. Agent
outputs are capped at 64 tokens to emphasize prefill/cache costs rather than long
answers; reasoning may be truncated. Omitting follow-up `user` messages preserves
Qwen's last-query rendering boundary. It does not itself prove a cache hit.

Warm/cold classification uses server-reported cached tokens: warm is a positive
cache hit with fewer than 4,096 uncached tokens; cold is at least 4,096 uncached
tokens. Unknown cache usage remains unknown. Cold prefill tok/s divides uncached
tokens by client TTFT (including scheduling/transport), not isolated GPU time.
Enable vLLM `--enable-prompt-tokens-details` or SGLang `--enable-cache-report` for
cache accounting. Earlier retained runs without these flags have `n/a` cache data.

## Frozen file SHA-256 (raw bytes)

```
eefb7e864637a288c4131e13546a134c0bbc1c3d0d3e049dc2d0836b16bef69c  open_swe_agent_64k.json
dbf090d4517d8234e5d78f6df9a97d1c98b5fb18782e9e0fbbb8a6f01b901334  open_swe_agent_120k.json
efda6ce29ded5ead263a7e3747c923ccfbd4c228160a67832b80a56e94bd0530  open_swe_agent_252k.json
bc31dab01fda96255aa0fbc46fd348458842997c79fb3cb9151fdda98ef974bd  open_swe_cold_read_60k.json
```

`w34perf` additionally stores a canonical-JSON fixture digest in each run's config.
