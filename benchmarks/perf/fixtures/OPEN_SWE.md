# Open-SWE-derived workload fixtures

This directory contains two distinct classes of `nvidia/Open-SWE-Traces` replay
fixtures:

1. **Intact single-instance SWE trajectory replays (`open_swe_intact_16k.json`):**
   Pinned to dataset commit `4fa4c2ae9afc6b397bc50391d43450c79993c138` and
   reproducible via [`tools/extract_open_swe_intact.py`](../../../tools/extract_open_swe_intact.py).
   Each entry in `trajectories` is a single unresolved-free (`resolved == 1`),
   permissive-licensed repository task replayed in its exact chronological turn and
   tool-call order (no cross-repository concatenation, line slicing, or repeated
   padding).
2. **Assembled long-context stress fixtures (`open_swe_agent_{64k,120k,252k}.json`,
   `open_swe_cold_read_60k.json`):** Synthetic capacity/prefill stress assemblies
   built from real tool observations across multiple repositories (with repeated
   corpus blocks at the largest size and one initial issue prompt).

In both classes, generated assistant responses are preserved in conversation history
and recorded tool observations are appended deterministically without executing
commands. Every `agent_replay` request sends `tool_choice: "none"`; standalone text
cold-read requests send no tools and omit `tool_choice`. Replaying recorded
observations against newly generated responses measures **serving latency, batching,
speculation, and prefix-cache reuse**, not autonomous SWE task solve rate.

## Intact single-instance pool (`open_swe_intact_16k.json`)

Source: NVIDIA, [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
at commit `4fa4c2ae9afc6b397bc50391d43450c79993c138`, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Upstream repository
excerpts retain the permissive licenses reported by the dataset. Changes from raw
dataset rows: mapping assistant `tool_calls` + `tool` messages into deterministic
replay turns (`user` on turn 0, `user: null` tool-loop follow-ups thereafter) and
appending a two-line replay note to the upstream harness `system` prompt so models
running with `tool_choice: "none"` analyze replayed observations concisely. No
endorsement by NVIDIA or upstream authors is implied.

Calibrated with the tokenizer and Jinja chat template from
`nchapman/Qwen3.8-27B-W4A16-HGPTQ` (`ae51987e3339a2a70070702548571e10fe8bcbc2`,
`enable_thinking = false`, ~240 generated tokens/turn):

| Config / split / offset | Instance ID (`trajectory_uuid`) | Repository / license | Turns | Est. first → last prompt tokens |
|---|---|---|---:|---:|
| v2.0 / claudecode / 38 | `h5netcdf_h5netcdf_pr218` (`c6f7bf75-47ea-4fb1-94ef-596bc5fc4ca2`) | [h5netcdf/h5netcdf](https://github.com/h5netcdf/h5netcdf) / BSD-3-Clause | 12 | 641 → 8,982 |
| v2.0 / claudecode / 94 | `oauthlib_oauthlib_pr336` (`c27dc049-8c82-4cc4-a1cf-c2be6e289df0`) | [oauthlib/oauthlib](https://github.com/oauthlib/oauthlib) / BSD-3-Clause | 7 | 519 → 6,590 |
| v2.0 / opencode / 7 | `app-sre_qontract-reconcile_pr702` (`4d9878cc-6c0c-414b-a1a7-fb879bc8b532`) | [app-sre/qontract-reconcile](https://github.com/app-sre/qontract-reconcile) / Apache-2.0 | 15 | 2,521 → 10,082 |
| v2.0 / opencode / 78 | `more-itertools_more-itertools_pr742` (`11c08d8c-f852-415d-b779-21f43a511e5d`) | [more-itertools/more-itertools](https://github.com/more-itertools/more-itertools) / MIT | 12 | 2,558 → 9,906 |
| v2.0 / openhands / 2 | `python-tap_tappy_pr69` (`cc7c58e8-a5e6-4911-8dd1-cfb1b042ce24`) | [python-tap/tappy](https://github.com/python-tap/tappy) / BSD-2-Clause | 15 | 2,194 → 12,312 |
| v2.0 / openhands / 84 | `fonttools_fonttools_pr2353` (`55b9abbe-6e3f-4fa1-891b-f6eb659855a2`) | [fonttools/fonttools](https://github.com/fonttools/fonttools) / MIT | 12 | 2,156 → 10,115 |

## Assembled long-context stress fixtures (`open_swe_agent_{64k,120k,252k}.json`, `open_swe_cold_read_60k.json`)

Rows for the earlier assembled long-context fixtures were retrieved through the
Hugging Face datasets-server `/rows` API without recording a dataset commit; those
checked-in fixture bytes are the frozen workload:

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

Long-context stress tiers calibrated with Qwen3.8-27B GPTQ revision
`0df81eadcfd59601cb82ccece2f16ea7d39244bf` (tokenizer truncation disabled):

| Fixture | Turns | Estimated first → last prompt tokens |
|---|---:|---:|
| `open_swe_intact_16k.json` (6 intact trajectories) | 7–15 | 519 → 12,312 |
| `open_swe_agent_64k.json` | 9 | 15,508 → 59,222 |
| `open_swe_agent_120k.json` | 10 | 26,672 → 118,004 |
| `open_swe_agent_252k.json` | 9 | 54,515 → 251,999 |
| `open_swe_cold_read_60k.json` | 1 | 57,589 (repeat context twice: 115,083) |

The size names describe capacity tiers, **not exact 16,384/65,536/131,072/262,144-token
prompts**. The separate `needle_gate` uses a fixed 262,144-token window. Omitting
follow-up `user` messages preserves Qwen's `last_query` rendering boundary; it does
not itself prove a cache hit.

Warm/cold classification uses server-reported cached tokens: warm is a positive
cache hit with fewer than 4,096 uncached tokens; cold is at least 4,096 uncached
tokens. Unknown cache usage remains unknown. Cold prefill tok/s divides uncached
tokens by client TTFT (including scheduling/transport), not isolated GPU time.
Enable vLLM `--enable-prompt-tokens-details` or SGLang `--enable-cache-report` for
cache accounting. Earlier retained runs without these flags have `n/a` cache data.

## Frozen file SHA-256 (raw bytes)

```
cc722396e99de4b5bc399588025f26ccb2fc7b7f3644b33cc003a886bef6cfb6  open_swe_intact_16k.json
eefb7e864637a288c4131e13546a134c0bbc1c3d0d3e049dc2d0836b16bef69c  open_swe_agent_64k.json
dbf090d4517d8234e5d78f6df9a97d1c98b5fb18782e9e0fbbb8a6f01b901334  open_swe_agent_120k.json
efda6ce29ded5ead263a7e3747c923ccfbd4c228160a67832b80a56e94bd0530  open_swe_agent_252k.json
bc31dab01fda96255aa0fbc46fd348458842997c79fb3cb9151fdda98ef974bd  open_swe_cold_read_60k.json
```

`w34perf` additionally stores a canonical-JSON fixture digest in each run's config.
