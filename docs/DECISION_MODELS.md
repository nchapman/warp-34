# Decision-Model Serving Specification (`/v1/systemone`, `/v1/decisions`)

## 1. Objective & Donor Provenance

Decision models ("System-1 classifiers / structured scorers") evaluate one or more typed decision questions (`noul`, `choice`, `score`) over a shared `state` in **a single prefill forward pass with zero autoregressive decode tokens**, returning calibrated probability distributions, expected scores, and analytical confidence metrics.

`warp-34` exposes two first-class HTTP endpoints backed by one unified decision engine:

1. **`POST /v1/systemone`** (and `POST /systemone`) — TypeSafe System One wire contract (primary donor standard across `llama.cpp`, `vllm.cpp`, `vLLM` `generative_scoring`, and `Ollama`).
2. **`POST /v1/decisions`** (and `POST /decisions`) — Ergonomic ordered-array Decisions adapter (`object: "decision.list"`, `decisions[]` + `answers{}` with `"noul" | "choice" | "score"`), translating deterministically onto the same execution & calibration pipeline.

### Explicit Donors
- **`llama.cpp` (`tools/server/server-decision.{h,cpp}`)**:
  - Request validation, `noul` / `choice` / `score` option compilation, `kev` text flattening & `<|tag|>` -> `<¦tag¦>` special-token sanitization, `lev` dual-variant forward/reverse option-order bias cancellation & 9-level `noul` rating scale, temperature bucket lookup, TypeSafe confidence formulas, and LCP shared-prefix task grouping.
- **`vllm.cpp` (`/home/nchapman/Code/warptoo/vendor/vllm.cpp`)**:
  - `decision_scorer.cpp` — deterministic Python-compatible JSON serialization (`", "` / `": "`, `ensure_ascii=False`, finite-float check) and strict field validation.
  - `nimble_inference.cpp` — `NimbleModel` (`Bespoke-Nimble-9B`) schema prompt contract, `SafeJson` (`<` -> `\u003c`, `>` -> `\u003e`), and `nimble_temperature` calibration.
  - `tev1_inference.cpp` — `Tev1` (`togethercomputer/Tev1-4B`, `Tev1-0.8B`) & general Qwen3.5 generative-scoring prompt contract (`state`-first JSON ordering + `enable_thinking=false` `<think>\n\n</think>\n\n` boundary).
  - `kev.h`, `kev_head.cpp`, `kev_registry.cpp` — `KevModel` Qwen3.5 FIM/box delimiter encoding (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`), pre-`lm_head` hidden-state readout, `head.safetensors` `PointerHead` (`q`, `k` `[head_dim, hidden_size]`), and perturbation-gated golden tests.
  - `xor_inference.h` — `XorModel` Qwen3.5 MoE dual-order logit readout.
- **`Cloudflare/clef-flash` & `Cloudflare/clef` (`joint_schema_model.py`, `joint_head_config.json`, `joint_head.safetensors`)**:
  - Single-pass joint-schema record encoding (`STATE:` + optional `<|vision_start|><|image_pad|>...<|vision_end|>\n` media block + `SCHEMA FIELDS:` with tracked `question_span` and `option_spans` + `JOINT SCHEMA DECISIONS:`), sorted-key compact JSON `render()`, `JointSchemaHead` (`EvidenceRoutingLayer` cross-attention + `TransformerDecoderLayer` joint field self/cross-attention + lexical `lm_head.weight` prior & residual cosine scorer), and `ConfidenceKind::MaxProbability` (`max_i p_i`).

---

## 2. Supported Model Family Scope on `warp-34`

`warp-34`'s inference spine is purpose-built for causal Qwen 3.5 / 3.6 / 3.8 hybrid GDN + paged full-attention (+ MoE + Vision) models. Decision profiles are auto-detected at model load time from `config.json`, `tokenizer_config.json`, and safetensors shards:

| Profile | Detection Rule | Backbone | Readout Mechanism | Max Options & Special Rules |
|---|---|---|---|---|
| **`Clef`** | `architectures` contains `"ClefModel"`, `decision.type == "clef"`, or `joint_head.safetensors` + `joint_head_config.json` present alongside Qwen3.5 weights | Qwen 3.5 VL / dense (`Cloudflare/clef-flash` 9B, `Cloudflare/clef` 27B) | **Full-sequence post-final-`GemmaRMSNorm` hidden states** from **one joint prefill pass** across all questions + `ClefJointHead` (`EvidenceRoutingLayer` + `TransformerDecoderLayer` + on-demand `pread` option-token lexical embeddings from `lm_head.weight`) | **2–64 options**, **1–64 joint questions** in one pass; `noul` `true`-first; `choice` keys sorted in prompt & permuted back to caller order; `ConfidenceKind::MaxProbability`; native multimodal `images` (`processor_config.json` `shortest_edge` respected). |
| **`Kev`** | `architectures` contains `"KevModel"`, `decision.type == "kev"`, or `head.safetensors` (`q.weight`, `q.bias`, `k.weight`, `k.bias`) present alongside Qwen3.5 weights | Qwen 3.5 dense / VL | **Post-final-`GemmaRMSNorm` hidden states** at `<\|box_end\|>` (`opt_idx[k]`) and `<\|fim_suffix\|>` (`decide_idx`) + `f64`-accumulated `PointerHeadForward` (`lm_head` GEMM skipped) | **2–64 options**; `kev_head_dim` (default `256`), `kev_temperature` (default `1.0`); `<\|...\|>` -> `<¦...¦>` sanitization. |
| **`Nimble`** | `architectures` contains `"NimbleModel"`, `decision.type == "nimble"`, or `config.json` defines `nimble_temperature` | Qwen 3.5 dense (`Bespoke-Nimble-9B`) | **Last-row `lm_head` candidate-token logits** at single-token codes `A..Z` | **2–26 options** (`parallel_schema.py` one-letter contract); `SafeJson` escaping; `nimble_temperature` calibration. |
| **`OpenJev`** | `architectures` contains `"OpenJevModel"` or `decision.type == "openjev"` | Qwen 3.5 dense | **Last-row `lm_head` candidate-token logits** at single-token letters `A..Z`, `a..z` | **2–52 options**; `noul_true_first = true`; RFC 8259 special-token delimiter escaping. |
| **`Lev` / `Xor`** | `architectures` contains `"XorModel"` / `"LevModel"` or `decision.type` in `{"lev", "xor"}` | Qwen 3.5 dense (`lev`) or Qwen 3.5 MoE (`xor`) | **Last-row `lm_head` candidate-token logits** at single-token codes `A..Z` | **2–26 options**; **2 variants (forward + reverse option order)** averaged to cancel position bias; **9-level rating scale (`0..8`)** for `noul` on `Lev`; `<\|...\|>` sanitization. |
| **`Tev1` (Default Qwen Generative Scorer)** | Standard `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` (`Tev1-4B`, `Tev1-0.8B`, or any loaded Qwen 3.5 / 3.6 / 3.8 model without a specialized head) | Qwen 3.5 / 3.6 / 3.8 dense or MoE | **Last-row `lm_head` candidate-token logits** at single-token labels `A..X` | **2–24 non-blank options** (`togethercomputer/tev1` `examples/decide.py`); `"state"` serialized first in user JSON (`{"state": ..., "question": ..., "options": [...]}`) + `<think>\n\n</think>\n\n` assistant prefix; RFC 8259 special-token delimiter escaping; `T = 1.0` default. |

### Explicit Scope Boundary (Non-Qwen Encoder Backbones)
Bidirectional encoder-only decision models (`laya` on ModernBERT, `GLiNER2.5` on DeBERTa) use non-causal encoder attention + `[CLS]`/`[MASK]`/`[SEP]` heads outside `warp-34`'s Qwen hybrid GDN/paged-FA spine. Loading a non-Qwen encoder checkpoint continues to fail fast in `ModelConfig::from_json` with an explicit unsupported-architecture error.

---

## 3. Wire Schemas

### 3.1 `POST /v1/systemone` (TypeSafe System One Contract)

#### Request
```json
{
  "model": "optional-model-id",
  "state": " Arbitrary text, JSON object, JSON array, or {\"messages\": [...]} ",
  "images": ["data:image/png;base64,..."],
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does the customer report a production outage?",
      "criteria": {
        "true": "Active production outage affecting live traffic.",
        "false": "Non-urgent inquiry, feature request, or staging issue."
      }
    },
    "route_team": {
      "type": "choice",
      "instructions": "Which engineering team should own this ticket?",
      "criteria": {
        "storage": "Database, disk, or snapshot failures",
        "network": "DNS, TLS, routing, or load-balancer issues",
        "billing": "Invoices, quotas, or payment methods"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "Rate operational severity from 0 (cosmetic) to 3 (critical).",
      "criteria": [
        "Cosmetic or documentation issue",
        "Minor degradation with workaround",
        "Major feature unavailable",
        "Complete production outage"
      ]
    }
  }
}
```

#### Validation Rules
- `state`: required, non-null (`string`, `object`, or `array`; string or serialized representation must be non-blank).
- `images`: optional array (`0..=4` entries) of `"data:image/{png,jpeg,x-portable-pixmap};base64,..."` strings, objects (`{"url": ...}`, `{"image_url": ...}`, or `{"media_type": ..., "data": ...}`), or public `http(s)://` URLs when `--allow-remote-images` is enabled (SSRF-guarded). Requires a vision-capable loaded checkpoint (`has_vision_weights == true`); non-empty `videos` fail fast with HTTP 400.
- `questions`: required non-empty JSON object (`1..=64` entries; key order preserved deterministically):
  - `type == "noul"`: `instructions` required (non-blank); `criteria` optional object with optional `"true"` and `"false"` string descriptions (defaults: `"Yes"` / `"No"`, with `false` ordered before `true` except on `Clef` and `OpenJev`; `Clef` defaults to `"The proposition is true or the answer is yes."` / `"The proposition is false or the answer is no."`).
  - `type == "choice"`: `instructions` required; `criteria` required non-empty object (`2..=max_options` entries, keys non-blank, values `string` or `null`; `Tev1` requires non-blank descriptions per `decide.py`).
  - `type == "score"`: `instructions` required; `criteria` required array of `2..=max_options` level descriptions (`string`, non-blank for `Tev1`), indexed `"0"`, `"1"`, ..., `"K-1"`.

#### Response (`200 OK`)
```json
{
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.9642
    },
    "route_team": {
      "type": "choice",
      "choice": "network",
      "probabilities": {
        "storage": 0.0211,
        "network": 0.9684,
        "billing": 0.0105
      },
      "confidence": 0.9526
    },
    "severity": {
      "type": "score",
      "score": 2.891,
      "legend": {
        "0": "Cosmetic or documentation issue",
        "1": "Minor degradation with workaround",
        "2": "Major feature unavailable",
        "3": "Complete production outage"
      },
      "probabilities": {
        "0": 0.002,
        "1": 0.015,
        "2": 0.073,
        "3": 0.910
      },
      "confidence": 0.9273
    }
  },
  "usage": {
    "input_tokens": 1420,
    "output_tokens": 0,
    "cached_tokens": 960
  }
}
```

### 3.2 `POST /v1/decisions` (Ergonomic Array/Object Adapter)

Translates ordered-array or simplified decision payloads (`state` or `context`, `decisions[]` or `questions`) onto the exact same internal `prepare_decision_batch` + `MambaRadixCache` execution pipeline:

- **Request**:
  - `model`: optional string
  - `state` (or `context` alias): string or arbitrary JSON object/array
  - `decisions` (or `questions`): non-empty array (`1..=64`) of `{ "id" (or "name"): "...", "type": "noul" | "choice" | "score", "instructions" (or "question"): "...", "criteria" (or "options"): ... }`, or canonical `questions` object
- **Response (`200 OK`)**:
  ```json
  {
    "object": "decision.list",
    "model": "Qwen3.5-9B-W4A16-GPTQ",
    "decisions": [
      {
        "id": "is_urgent",
        "type": "noul",
        "noul": 0.9642
      },
      {
        "id": "route_team",
        "type": "choice",
        "choice": "network",
        "probabilities": {"storage": 0.0211, "network": 0.9684, "billing": 0.0105},
        "confidence": 0.9526
      },
      {
        "id": "severity",
        "type": "score",
        "score": 2.891,
        "legend": {"0": "...", "1": "...", "2": "...", "3": "..."},
        "probabilities": {"0": 0.002, "1": 0.015, "2": 0.073, "3": 0.910},
        "confidence": 0.9273
      }
    ],
    "answers": {
      "is_urgent": {"type": "noul", "noul": 0.9642},
      "route_team": {"type": "choice", "choice": "network", "probabilities": {"storage": 0.0211, "network": 0.9684, "billing": 0.0105}, "confidence": 0.9526},
      "severity": {"type": "score", "score": 2.891, "legend": {"0": "...", "1": "...", "2": "...", "3": "..."}, "probabilities": {"0": 0.002, "1": 0.015, "2": 0.073, "3": 0.910}, "confidence": 0.9273}
    },
    "usage": {
      "input_tokens": 1420,
      "output_tokens": 0,
      "cached_tokens": 960
    }
  }
  ```

---

## 4. Calibration & TypeSafe Confidence Mathematics

All probability and confidence computations execute in IEEE-754 `f64` (`server-decision.cpp:696-778`):

1. **Per-variant temperature-scaled softmax & variant averaging**:
   Given `V` variants (`V = 2` for `Lev`/`Xor` `choice` with `K > 1`, `V = 1` otherwise) and temperature `T > 0`:
   $$p^{(v)}_i = \frac{\exp\big((s^{(v)}_i - \max_j s^{(v)}_j) / T\big)}{\sum_{j=0}^{K-1} \exp\big((s^{(v)}_j - \max_m s^{(v)}_m) / T\big)}$$
   Variant `v = 1` reverses option order back to canonical index `i`:
   $$p_i = \frac{1}{V} \sum_{v=0}^{V-1} p^{(v)}_{\pi_v(i)}, \quad \pi_0(i) = i,\ \pi_1(i) = K - 1 - i$$
   Note: For `Kev`, `kev_temperature` divides logits once (`T = kev_temperature`), matching `kev_registry.cpp:277`. Any `NaN` or non-finite logit fails fast with HTTP 500 rather than emitting corrupt probabilities.

2. **`noul` readout**:
   - Standard (`Kev`, `Nimble`, `OpenJev`, `Tev1`): `noul = p[idx("true")]`.
   - `Lev` / `Xor` 9-level rating scale (`K = 9`): `noul = \sum_{i=0}^{8} p_i \cdot \frac{i}{8}`.

3. **`choice` readout & per-family confidence (`ConfidenceKind`)**:
   - `choice = options[argmax_i p_i].key` (ties broken by lowest index `i` in canonical option order).
   - **`ConfidenceKind::NormalizedEntropy` (`Nimble`, `Tev1`)**:
     $$\text{confidence}_{\text{entropy}}(p) = \begin{cases} 1.0 & K < 2 \\ \text{clamp}_{[0,1]}\!\left(1.0 + \frac{\sum_{i: p_i > 0} p_i \ln p_i}{\ln K}\right) & K \ge 2 \end{cases}$$
   - **`ConfidenceKind::MarginAndModeDistance` (`Kev`, `OpenJev`, `Lev`, `Xor`)**, with uniform baseline $u = 1 / K$:
     $$\text{confidence}_{\text{choice}}(p) = \begin{cases} 1.0 & K < 2 \\ \text{clamp}_{[0,1]}\!\left(\frac{\max_i p_i - u}{1.0 - u}\right) & K \ge 2 \end{cases}$$
   - **`ConfidenceKind::MaxProbability` (`Clef`)**:
     $$\text{confidence}_{\max}(p) = \max_{0 \le i < K} p_i$$

4. **`score` readout & per-family confidence**:
   - Expected score: $\text{score} = \sum_{i=0}^{K-1} i \cdot p_i$.
   - **`ConfidenceKind::NormalizedEntropy` (`Nimble`, `Tev1`)**: uses $\text{confidence}_{\text{entropy}}(p)$ above.
   - **`ConfidenceKind::MarginAndModeDistance` (`Kev`, `OpenJev`, `Lev`, `Xor`)** (`vllm.cpp` `decision_scorer.cpp`), with mode $m = \arg\max_i p_i$:
     $$s = \sum_{i=0}^{K-1} p_i \cdot |i - m|, \qquad \text{confidence}_{\text{score}}(p) = \begin{cases} 1.0 & K < 2 \\ \text{clamp}_{[0,1]}\!\left(1.0 - \frac{s}{K - 1}\right) & K \ge 2 \end{cases}$$
   - **`ConfidenceKind::MaxProbability` (`Clef`)**: uses $\text{confidence}_{\max}(p) = \max_{0 \le i < K} p_i$.

---

## 5. Engine Seam & `MambaRadixCache` Shared-Prefix Batching

### 5.1 Why `MambaRadixCache` Is Uniquely Advantageous on Qwen 3.5 / 3.6 / 3.8
Every prompt contract across `Kev`, `Nimble`, `Tev1`, `OpenJev`, `Lev`, and `Xor` places `state` in the leading prefix before the per-question suffix:
- `Kev`: `[<|fim_prefix|>] state_tokens [<|fim_middle|>]` (shared) + `instr_tokens [<|box_start|>] opts... [<|fim_suffix|>]`
- `Nimble`: `<|im_start|>system\n...\n<|im_start|>user\n{"context": <state>, "schema": [...]}\n\nRequested field: ` (99%+ shared!) + `"<field>"<|im_end|>...`
- `Tev1`: `<|im_start|>system\n...\n<|im_start|>user\n{"state": <state>, "question": ` (shared!) + `"<instr>", "options": [...]}`

Because Qwen 3.5 / 3.6 / 3.8 interleaves 3 GDN recurrent layers per 1 full-attention layer, sharing KV alone is invalid without also restoring recurrent GDN `ssm_state` (`FP32`) + `conv_state` (`FP16`). `warp-34`'s `MambaRadixCache` + `HybridReqToTokenPool` already checkpoints **both** 16-token paged KV blocks and 64-token GDN state snapshots!

### 5.2 Execution Flow (`EngineCommand::ScoreDecisionBatch`)
1. **Prompt Compilation & Tokenization**:
   - Compile all `(question_idx, variant_idx)` tasks into token ID vectors (+ `MultimodalInput` on VL requests) and attach each task's `DecisionReadout`:
     - `DecisionReadout::CandidateLogits { token_ids: Vec<u32> }` (`Nimble`, `Tev1`, `OpenJev`, `Lev`, `Xor`)
     - `DecisionReadout::KevPointer { option_positions: Vec<usize>, decide_position: usize }` (`Kev`)
2. **Longest-Common-Prefix (`LCP`) Shared-State Warmup**:
   - When a batch has `>= 2` tasks sharing `LCP >= 64` tokens (at least one `MambaRadixCache` GDN checkpoint boundary, leaving `>= 1` suffix token on task 0), `EngineWorker` evaluates **Task 0** first with `mamba_branching_seqlen` (honored by both `SchedulerV1::schedule_step` and multimodal `ScheduledVision::step_prefill`) and immediately inserts its prefix + GDN snapshots into `MambaRadixCache`.
   - Tasks `1..M` then hit `MambaRadixCache` on the entire 64-token-aligned `[images + system + state]` prefix (skipping redundant ViT encoding on cached images and reporting exact `cached_tokens`) and prefill **only their uncached question suffixes** together in batched multi-sequence forward pass(es) respecting `max_num_batched_tokens`!
   - Repeated `/v1/systemone` or `/v1/decisions` calls over the same `state` hit `MambaRadixCache` even on Task 0!
3. **Zero-Decode Readout Seams in `Qwen3_5Engine`**:
   - **`CandidateLogits`**: Runs `forward_batch_tokens(..., selected_rows = &[last_row_per_seq])` — `lm_head` executes **only** on the `B` final sequence rows (never on prompt interior rows), gathers the `K` candidate token logits per task on host, releases sequence KV/GDN slots (committing prefix snapshots to `MambaRadixCache`), and performs **zero decode steps** (`output_tokens = 0`).
   - **`KevPointer` (`forward_batch_hidden_rows`)**: Adds a clean pre-`lm_head` exit after final `GemmaRMSNorm` in `Qwen3_5Engine` that gathers `selected_rows` (`option_positions` + `decide_position`, `[K + 1, hidden_size]` `FP16`) directly from `normed_hidden` (already replicated across ranks after `all_reduce_rmsnorm` under `TP=2`!), skips the `248,320`-vocab `lm_head` GEMM completely, converts the `(K + 1) * hidden_size` elements to `f32`, and runs `PointerHeadForward` (`f64` dot-product accumulation).
   - **`Clef` (`ClefJointHeadGpu` + `op_clef.cu`, `STACK:0`, GPU-only at runtime)**: Runs a single joint prefill pass (including `ScheduledVision` 3-plane `mRoPE` + ViT embedding splice when `images` are supplied), streams each post-final-`GemmaRMSNorm` prefill chunk directly on Rank 0 GPU (`hidden_norm` + `FP32` span-mean accumulation + cuBLAS `FP16` `memory_projection` into `[seq_len, width]` `FP16` `d_memory`, with **zero `[seq_len, hidden_size]` host/device materialization**), fetches option-span lexical rows on demand via `SafetensorsArchive::read_row_f32` (`pread` + bounded row cache), and executes `ClefJointHeadGpu::finalize` (cuBLAS `FP16` Tensor-Core GEMMs + 4-warp online-softmax `w34_clef_mha_attn_f16` across `EvidenceRoutingLayer` & `TransformerDecoderLayer` + batched `residual_scorer`) on Rank 0's CUDA stream. Host `Vec<f32>` head weights (`~466 MiB`) are dropped immediately after `FP16` `DeviceArena` upload; safe-scalar `ClefJointHead::forward` is retained strictly under `#[cfg(test)]` as the PyTorch reference oracle.

---

## 6. Verification & Parity Gates

1. **Donor Golden Unit Tests (`cargo test --lib`)**:
   - **`Clef` `JointSchemaHead` & `encode_record` Goldens (`clef::tests::*`)**: Verify exact PyTorch `JointSchemaHead` logit parity (`< 2e-6` CPU `f32`, plus `ClefJointHeadGpu` `FP16` vs `FP32` golden parity and 4-chunk split-prefill invariance `< 1e-4`), cross-question field self-attention perturbation gate, sorted-key `clef_render_value`, and exact 274-token text / 341-token multimodal token-ID & `(question_span, option_spans)` parity against `Cloudflare/clef-flash` `joint_schema_model.py`.
   - **`Kev` PointerHead & Sequence Encoding**: Port PyTorch golden vectors and all three silent-bug perturbation gates from `vllm.cpp` (`kev_head_goldens.inc` / `test_kev.cpp`: exact logit match `< 1e-5`, swapped `q/k` rejection, omitted-bias rejection, wrong `1/sqrt(head_dim)` scale rejection, `<|tag|>` -> `<¦tag¦>` sanitization, and YAML-style `decision_kev_render`).
   - **`Nimble` & `Tev1` Prompt + Scorer Goldens**: Port prompt-rendering and logit-to-answer golden checks from `vllm.cpp` (`test_nimble.cpp`, `test_tev1_systemone.cpp`: `SafeJson` `<`/`>` escaping, `state`-first JSON serialization, `noul`/`choice`/`score` validation errors, blank-description rejection on `Tev1`).
   - **Confidence, Multimodal `images`, & `Lev`/`Xor` Dual-Variant Math**: Verify `entropy_confidence` (uniform -> `0.0`, one-hot -> `1.0`), `margin_choice_confidence`, `mode_distance_score_confidence` (`1 - s/(K - 1)`), `MaxProbability`, multimodal `parse_decision_images` validation, `Xor`/`Lev` forward+reverse variant averaging, and `Lev` 9-level `noul` expected value against `vllm.cpp` `decision_scorer.cpp` / `test_xor.cpp`.
2. **GPU & Live HTTP Integration Tests (`crates/w34-host/tests/decision_models.rs`, `--features cuda`)**:
   - **`ClefModel` Joint Multi-Question + Multimodal `images` `TP=1` & `TP=2` Parity & Speed**: Verify `ClefJointHeadGpu` + `Qwen3VisionEncoder` over `/v1/decisions` and `/v1/systemone` across both `TP=1` and `TP=2` (caller-order `choice` probability preservation, visual counterfactual sensitivity, and `TP=1` vs `TP=2` parity `< 3e-3`), plus live `Cloudflare/clef-flash` (9B) parity & latency vs PyTorch CUDA `joint_schema_model.py` (`85.1 ms` `TP=2` / `109.8 ms` `TP=1` HTTP vs `185.0 ms` in-process PyTorch CUDA at 428 text tokens = **`2.17×` / `1.68×` faster**; `86.2 ms` `TP=2` / `97.2 ms` `TP=1` HTTP vs `164.6 ms` PyTorch CUDA on 309-token multimodal PNG = **`1.91×` / `1.69×` faster**; `565.2 ms` `TP=2` / `669.0 ms` `TP=1` HTTP vs `1,024.0 ms` PyTorch CUDA at 3,254 tokens = **`1.81×` / `1.53×` faster**).
   - **`TP=2` Sharded-`lm_head` Multimodal `MambaRadixCache` Reuse**: Verify `Tev1` (`CandidateLogits`) with `images` under `TP=2` (`load_for_serving` rank-0/rank-1 `KvPool` & `StateBundlePool` lockstep clamping + `ScheduledVision` `mamba_branching_seqlen` prefix donation yielding `cached_tokens >= 64`).
   - **`KevModel` `head.safetensors` GPU Parity**: Load `KevPointerHead` (`q.weight`, `q.bias`, `k.weight`, `k.bias`) from `head.safetensors` alongside `Qwen3.5-4B-W4A16-GPTQ` and verify `/v1/systemone` (with `MambaRadixCache` shared-state reuse via `max_prefix_lookup_len`) matches an uncached full-prefill GPU reference to `< 1e-4`.
   - **Live `/v1/systemone` & `/v1/decisions` across `Clef`, `Tev1`, `Kev`, `Lev`, and `Nimble`**:
     - Multi-question requests (`noul` + `choice` + `score`) over a shared `state`: verify exact 146-token `tev1_goldens.inc` Qwen3.5 token-ID parity, special-token injection immunity, semantically accurate classifications, `usage.output_tokens == 0`, and `usage.cached_tokens > 0` proving `MambaRadixCache` `mamba_branching_seqlen` shared-prefix reuse across sub-queries.
     - `/v1/decisions` ordered-array adapter (`object: "decision.list"`, duplicate-id HTTP 400 validation, `decisions[]` + `answers{}` parity, multimodal `images` forwarding).
