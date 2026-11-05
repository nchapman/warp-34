# warp-34 — REST API Landscape (Research Note)

> Research note dated 2026-10-31. Survey only — no implementation commitment.
> Two independent use cases are covered: **§1** which generative-serving API shapes
> to support beyond classic OpenAI chat completions, and **§2** the System One
> decision-model API family, including the model-side readout mechanics we would
> have to implement to serve one. Local evidence comes from the pinned oracles
> (`vLLM 0.31.0` installed tree, `SGLang v0.5.19` Docker image, vendored
> `llama.cpp` @ `cb7934c52`, vendored `vllm.cpp`) plus public web sources.

---

## §1 Generative serving: API shapes beyond chat completions

### 1.1 Current warp-34 surface

`crates/w34-host/src/server/http.rs`:

| Route | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming SSE, tool calls, `image_url` vision, `reasoning_content` |
| `POST /v1/completions` | legacy text completion |
| `GET /v1/models`, `POST /tokenize`, `POST /detokenize` | |
| `GET /health`, `GET /metrics` | |

### 1.2 Oracle support matrix

| API family | warp-34 | vLLM 0.31.0 | SGLang v0.5.19 | llama.cpp | vllm.cpp (vendored) |
|---|---|---|---|---|---|
| OpenAI Chat Completions `/v1/chat/completions` | ✅ | ✅ full | ✅ full | ✅ full (parallel + `none` `tool_choice`, `reasoning_format`) | ✅ (tools, SSE) |
| OpenAI Responses `/v1/responses` | ❌ | ✅ + `GET /v1/responses/{id}` + `/cancel`, ungated | ✅ same three routes | ✅ + `/v1/responses/input_tokens` | ❌ |
| Anthropic Messages `/v1/messages` | ❌ | ✅ + `/v1/messages/count_tokens` | ✅ + count_tokens | ✅ + count_tokens (tools need `--jinja`) | ❌ |
| Legacy `/v1/completions` | ✅ | ✅ | ✅ | ✅ | ✅ |

Extras: vLLM adds Cohere v2 chat + batch; SGLang adds native `/generate`, an
**Ollama-compatible `/api/chat`**, embeddings/classify/rerank; llama.cpp adds
`/props`, `/infill`, `/rerank`, resumable `/v1/stream`, GCP/Vertex aliases, and
per-family token-count endpoints (`/v1/chat/completions/input_tokens`,
`/v1/responses/input_tokens`, `/v1/messages/count_tokens`).

**Consensus: all three actively-maintained oracles now ship all three families.**
The two new families map exactly onto the two flagship agent CLIs:
**Responses ← Codex**, **Anthropic Messages ← Claude Code**.

### 1.3 Family B: OpenAI Responses (Codex)

vLLM reference: `vllm/entrypoints/openai/responses/` (`api_router.py:48,80,110`,
`protocol.py`, `serving.py`, `streaming_events.py`, `harmony_utils.py`). SGLang:
`python/sglang/srt/entrypoints/openai/serving_responses.py` ("responses API with
reasoning support"). llama.cpp: `routes.post_responses_oai` (`tools/server/server.cpp:269`).

Wire essentials:

- `input`: string **or** item list — message items plus agent history items
  (`function_call`, `function_call_output`), reasoning items.
- `instructions` field (system-prompt analog), `tools` + `tool_choice`,
  `reasoning: {effort}`, `previous_response_id`, `store`, `background`.
- Streaming: typed SSE events with `event: <type>` framing —
  `response.created/in_progress/completed`, `output_item.added/done`,
  `content_part.added/done`, `output_text.delta/done`,
  `reasoning_text.delta/done`, `function_call_arguments.delta/done`.
- **Statelessness suffices for Codex**: it sends `store: false` and replays the
  full item history each turn; `GET/cancel` routes only matter for background
  mode. No server-side response store needed.
- vLLM's `harmony_utils.py` handles the harmony prompt format Codex-style
  reasoning models expect.

### 1.4 Family C: Anthropic Messages (Claude Code)

vLLM reference: `vllm/entrypoints/anthropic/` — a clean, self-contained
`{protocol.py, serving.py}` pair translating Anthropic↔OpenAI internally. SGLang:
`python/sglang/srt/entrypoints/anthropic/` (thinking blocks incl.
`redacted_thinking`, server-tool types, Anthropic-shaped error envelopes).
llama.cpp: `routes.post_anthropic_messages` (`tools/server/server.cpp:273`) —
**battle-tested against real Claude Code traffic** (an actual
`x-anthropic-billing-header: cc_version=2.1.101…` request is handled in
`tools/server/server-chat.cpp:323`; conversion incl. thinking blocks at
`server-chat.cpp:408-535`).

Wire essentials:

- `system` as string **or** block array with `cache_control` (must be accepted;
  can be a billing no-op), **`max_tokens` required**, `stop_sequences` naming,
  `top_p`/`top_k`, `thinking: {type: enabled, budget_tokens ≥ 1024}`.
- `x-api-key` + `anthropic-version` headers (accept/ignore gracefully).
- SSE framing with named events: `message_start`, `content_block_start`,
  `content_block_delta` (`text_delta`, `input_json_delta`, `thinking_delta`),
  `content_block_stop`, `message_delta`, `message_stop`, `ping`.
- Tool use streams arguments as `input_json_delta`; results return as
  `tool_use` blocks / `tool_result` blocks.
- **`/v1/messages/count_tokens`** is called by Claude Code for context
  management — needs only a tokenizer-count path. All three oracles ship it.
- Anthropic-shaped error envelopes (`{type: "error", error: {type, message}}`).

### 1.5 Cross-oracle divergences to respect

- **Reasoning field naming forked**: vLLM 0.31 emits `reasoning` (input
  normalization renames legacy `reasoning_content` → `reasoning`,
  `chat_completion/protocol.py:528-544`); SGLang and llama.cpp emit
  `reasoning_content` (the DeepSeek de-facto standard). Recommendation: emit
  `reasoning_content`, accept both on input.
- Clients poke `/v1/models` (Codex validates the model id) and the count_tokens
  endpoints above.

### 1.6 Donor mapping & recommendation

| Piece | Donor | Why |
|---|---|---|
| Responses schema/events | `vLLM 0.31` `entrypoints/openai/responses/` | most complete reference incl. harmony |
| Anthropic Messages translation | `vLLM 0.31` `entrypoints/anthropic/` | self-contained 2-file translator |
| C++ cross-check for both | vendored `llama.cpp tools/server/` | only non-Python reference implementing all three; Claude-Code-verified |

Both new families are thin wire-translation layers over the existing request
path — the engine-side hard parts (Qwen3Parser reasoning/tools, tool
constraints, SSE infra) already exist. Supporting endpoints to add alongside:
`/v1/messages/count_tokens` (+ optionally the llama.cpp-style
`…/input_tokens` counters).

---

## §2 System One decision models and `/v1/systemone`

### 2.1 What they are

**TypeSafe** (typesafe.ai) launched the category on **2026-09-15** with **Jev**,
their flagship closed model ("Introducing System One models and Jev"). The name
is Kahneman's *Thinking, Fast and Slow* System 1: fast, intuitive judgment. A
System One model evaluates a `state` against typed questions and returns
**calibrated probabilities, not generated text** — no chain-of-thought, no JSON
parsing, one forward pass per question. Training calibrates probabilities
against outcomes (RLCD — Reinforcement Learning for Calibrated Decisions).

Why it took off (Sept–Oct 2026): routing, guardrails, and agent-loop decisions
are the majority of "LLM calls" in production pipelines, and paying
generation-latency/token cost for what is logically an `if` is wasteful —
Jev-class decisions run ~100–210 ms with usage-style billing (~$0.042/M input
tokens reported). Simon Willison's 2026-09-21 writeup mainstreamed it; open
clones appeared within ~3 days.

### 2.2 The two wire formats

There are now **two competing standard shapes** for the same semantics:

| | TypeSafe System One `POST /v1/systemone` | OpenAI Decisions `POST /v1/decisions` |
|---|---|---|
| Payload to judge | `state`: string, object, or array | `input`: string or message list (`input_text` / `input_image` parts) |
| Questions | named **map**, 1–64 (128 on some servers) | ordered **array** with `name`, 1–128 |
| Binary | `noul` → `noul`: P(yes) | `predicate` → `probability` |
| Choice | `criteria`: map label→description (≤255) | `choices`: array of `{value: string\|bool, description}` |
| Score | `criteria`: 2–10 level descriptions | `levels`: array of `{label, description}` |
| Grounding | `box` (Jev 1.14, opt-in header) | `box` |
| Images | `images` array / `assets` map + `TypeSafe-Danger-Zone-Images` header | `input_image` parts |
| Answers | keyed by question id | array in request order |
| Reference model | `jev-latest` / `jev-1.13.0` | `gpt-6-luna` |

Both answers include per-option `probabilities` (summing to 1) and
`confidence` for choice/score (noul/predicate omit it — a two-outcome
distribution is fully described by the single probability). `score` is the
probability-weighted level index (can land between levels). `usage.output_tokens`
is always ~0. Non-streaming in every implementation surveyed.

Request/response example (canonical System One, as served by llama.cpp
`tools/server/README.md:1670+`):

```jsonc
// POST /v1/systemone
{ "state": "Customer message: I was charged twice for my order last week.",
  "questions": {
    "route":   { "type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": null, "shipping": null, "technical": null} },
    "angry":   { "type": "noul", "instructions": "Is the customer angry?" },
    "urgency": { "type": "score", "instructions": "How urgent is this?",
                 "criteria": ["can wait", "this week", "today", "right now"] } } }

// response
{ "model": "openjev",
  "answers": {
    "route":   { "type": "choice", "choice": "billing",
                 "probabilities": {"billing": 0.9998, "shipping": 0.0001, "technical": 0.0001},
                 "confidence": 0.9997 },
    "angry":   { "type": "noul", "noul": 0.6328 },
    "urgency": { "type": "score", "score": 2.0858,
                 "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"},
                 "probabilities": {"0": 0.0023, "1": 0.116, "2": 0.6753, "3": 0.2064},
                 "confidence": 0.673 } },
  "usage": { "input_tokens": 239, "output_tokens": 0 } }
```

`state` may also be a chat-message array (or `{"messages": [...]}`) — then
`image_url` content parts are extracted as images, exactly like chat
completions.

### 2.3 Who serves it

| Runtime | Support | Notes |
|---|---|---|
| **TypeSafe hosted** | origin | `api.typesafe.ai/v1/systemone`, `typesafe-sdk`, OpenAPI spec; SDK honors `TYPESAFE_BASE_URL` — the interop lever every local server pulls |
| **Ollama ≥0.35** (2026-09-29) | `/v1/systemone` | launched with nimble + tev1 4B/0.8B; 9+ models by 0.40 (clef, clef-flash, laya…); TypeSafe SDK works unchanged against localhost |
| **vLLM ≥0.32** | `/v1/systemone` (PR #59299) + OpenAI `/v1/decisions` (PR #60465) | modular read-strategy registry (Qwen3/3.5…); RFC #59365 debates per-request strategies vs format purity. **Our pinned 0.31.0 predates it** — one version behind |
| **llama.cpp** (vendored) | `/v1/systemone` | `a4cb4c61f` (#29818: laya, julia-1, lev, openjev, kev), `46ca246de` (#29844 nimble), `99b95488c` (#29831 clef); cross-question shared-prefix batching; per-family readouts |
| **vllm.cpp** (vendored fork) | `/v1/systemone` + nonstandard `/permute`, `/separate`; C ABI `vllm_decide` (v29) | ports: kev, laya, nimble, tev1, xor, GLiNER2.5-Decide, cua-s1-forms (`/v1/score`) — see `docs/FEATURES.md:216-224` in that tree |
| **SGLang v0.5.19** | ❌ | no systemone route |
| **LiteLLM** | both routes | translates between the two formats; ~10 providers (TypeSafe, OpenAI, Cloudflare, Ollama, vLLM, Databricks `ai_decide`, Azure `Microsoft-Decision-1`, Fireworks `rune`, …) |
| **ollaya** | `/v1/systemone` | "Ollama for decision models", 29 families |
| Gateways | partial | new-api relay requested (issue #7726, closed as not planned at the time) |

### 2.4 Model families and readout mechanics (what we would implement)

All local evidence: llama.cpp `tools/server/server-decision.{h,cpp}` (802 lines),
vendored vllm.cpp `src/vllm/model_executor/models/{kev,laya,nimble}_registry.cpp`,
`decision_scorer.cpp`, `clm_head.cpp`, and specs `.agents/specs/{kev,laya,nimble,tev1,xor,gliner2.5-decide}.md`.

A decision model = **backbone forward + a readout that turns logits or hidden
states at specific token positions into a per-option score**, then
`softmax(scores / T_calib)`. The families differ in backbone, readout position,
and calibration:

| Model | Who | Backbone | Readout | Options max | Quirks |
|---|---|---|---|---|---|
| **Jev** | TypeSafe (closed) | unknown | hosted only | — | the reference |
| **openjev** | openjev/openjev (HF, CC BY-NC 4.0) | ~27B Qwen-family | logits at single-token option letters `A-Z,a-z` | 52 | one forward pass per question; noul options ordered **true-first**; multimodal (≤8 images via `images` field or chat-message `image_url` parts, OpenJev multimodal API extension); 84.0% vs Jev 85.4% on its 10k benchmark; shuffle-flip rate 2.3% |
| **lev** | — | — | logits at single-token codes `A..Z`, then `AA..ZZ` | ~255 (single-token codes only) | **dual-variant**: every choice evaluated twice (forward + reversed option order), probabilities averaged — cancels first-label position bias; **noul via a 9-level rating scale**, answer = expected value over levels; prompt convention: JSON keys sorted recursively |
| **nimble** | Bespoke Labs | Qwen3.5-9B + rank-16 LoRA (248 modules **incl. GDN projections**) merged offline at convert | candidate-letter logits, `softmax(logits/T)`, one last-row forward per question | 26 (2,048-token prompt budget) | **one prompt lists ALL questions**; T=1.0 (v1) / T=2.179 (v2); vllm.cpp reports answer parity ≤1e-12 vs the openjev `scoring.answer` reference |
| **kev** | jaredpalmer/kev | Qwen3.5-0.8B-Base + rank-16 LoRA (merged at convert) + **PointerHead**; no LM head (hidden-state extraction) | q/k attention head (Linear 1024→256, scale 1/√256) over hidden states at `<\|box_end\|>` marker tokens (option ends); question read at last token; T=2.406 | 255 | state JSON rendered to YAML-ish text (`kev/api.py render`); special tokens in input text neutralized |
| **laya** | Convai | **ModernBERT-large encoder** (non-generative) + transformer decision head | one score per question-type column at CLS-style positions | 255 | prompt layout `[cls] question [sep] ([marker] option)* [sep] state [sep]`; truncation to the trained head budget (`max_head_tokens`, `max_option_tokens` from metadata, options shrunk evenly); must fit one batch |
| **clef** | Cloudflare (open, Workers AI) | joint decision model | **all questions + options in ONE prompt** with `<<clef:question>>` / `<<clef:option>>` markers; pieces tokenized separately; per-piece decision order feeds a joint head | 255 | decides questions jointly (not independent); head returns NaN if it can't evaluate; clef-flash ~38.8 ms; vision; a clef server is decision-only (no generation) |
| **xor** | juspay | **Qwen3.6-35B-A3B MoE** + decision head, merged BF16 | one token logit per candidate option | — | forward+reverse option order + calibration (same bias trick as lev); multimodal ≤8 images; SGLang parity is the reference oracle |
| **tev1** | Together AI (4B / 0.8B) | Qwen3.5 SFT (text-generation class) | generative scoring: `max_tokens=1` + `logprob_token_ids` at answer letters, the model's own `decide.py` prompt | — | tied Jev (93.0%) on 440 BFCL tool-routing decisions in third-party testing |
| **GLiNER2.5-Decide** | fastino | DeBERTa-v3-large encoder + classification head | span-extraction heritage; softmax/sigmoid head | — | served on the same `/v1/systemone` lane in vllm.cpp |
| others | pplx-decider (Perplexity), winnow, coba, julia-1, cua-s1-forms (trycua, `/v1/score`), Microsoft-Decision-1, Databricks `ai_decide`, strands-decider, Fireworks `rune` (preview), OpenAI `gpt-6-luna` (`/v1/decisions`) | | | | ollaya counts 29 families total |

Note the backbone overlap with our target spine: Qwen3.5 dense 0.8B/4B/9B
(kev, tev1, nimble), Qwen3.6-35B-A3B MoE (xor), ~27B Qwen-family (openjev) —
including **GDN-hybrid backbones whose decision LoRAs touch the DeltaNet
projections** (nimble). The readout is the only new math.

### 2.5 Serving machinery common to all families (llama.cpp reference)

From `tools/server/server-decision.cpp`:

- **Prompt construction**: a per-model `systemone` chat template (GGUF
  `llama_model_chat_template(model, "systemone")`) receiving
  `{id, type, instructions, state, options[{key, description, label}]}` —
  each question becomes one task/prompt sharing the `state`.
- **Calibration temperatures travel in model metadata**: GGUF keys
  `{arch}.decision.temperature.{bucket}` where buckets depend on question type
  and option count (openjev: `2`, `3_5`, `6_10`, `11`; lev: `small` ≤8,
  `mid` ≤26, `large`). Probabilities are
  `softmax(scores / T)` per variant, variants averaged. *"Not guaranteed to be
  calibrated for your data"* — they are fit constants, not physics.
- **Confidence formulas (published by TypeSafe)**, `server-decision.cpp:693+`:
  - choice: `(p_max − 1/k) / (1 − 1/k)` (rescaled margin over uniform)
  - score: `1 − E[i − mode] / E_uniform[|i − center|]` (mean distance to the
    mode relative to a uniform distribution's)
  - noul: none (single number is complete)
- **Shared-prefix batching** (`server_decision_group_tasks`, `:775+`): the
  questions of a request share the state prefix; tasks are grouped into slot
  batches and deduplicated by longest common prefix (`n_tokens_shared`,
  `get_common_prefix`), so N questions over one state cost ~1 prefill + N short
  suffixes. This maps directly onto our paged KV + `MambaRadixCache` prefix
  sharing (and is the exact optimization the kev spec defers — "kev
  reprocesses state per question" — so it is a legitimate differentiation).
- **Usage**: `input_tokens` = prompt tokens summed over questions,
  `output_tokens` = 0 (hosted Jev reports small nonzero values; treat as
  billing flavor).
- **Non-streaming** everywhere; multimodal via the OpenJev extension (≤8 data
  URLs, images placed before state in the prompt).

### 2.6 Fit for warp-34

- **Serving profile is prefill-only**: one shared prefill + single-position
  logit/hidden reads. No decode loop, no KV growth beyond the shared prefix,
  no sampling. Our engine already computes everything needed; the new pieces
  are (a) wire endpoints, (b) per-family `systemone` templates + JSON→text
  state rendering, (c) readout heads (pointer/CLS/joint) with
  hidden-state-extraction forwards, (d) calibration + confidence math.
- **If we serve both wire formats**, follow LiteLLM's precedent: one core,
  `/v1/systemone` (TypeSafe shape) and optionally `/v1/decisions` (OpenAI
  shape) as a thin adapter. TypeSafe's is the one the open ecosystem
  (Ollama/llama.cpp/ollaya/SDKs) already duplicated; OpenAI's is the one with
  first-party backing. The vLLM RFC (#59365) is the live standardization fight.
- **License note**: OpenJev weights are CC BY-NC 4.0 (non-commercial);
  clef/tev1/others vary. We would serve user-supplied weights — worth a
  per-family license column before any bundled-model story.
- **Donor pins if we build it**: llama.cpp `tools/server/server-decision.cpp`
  @ `cb7934c52` (readouts, calibration, shared-prefix batching, README wire
  docs) and vendored vllm.cpp specs `.agents/specs/{kev,laya,nimble,tev1,xor}.md`
  (per-model parity methodology with golden tests vs the Python references).
  Hosted-API schema: `docs.typesafe.ai/api/system-one` (OpenAPI).
