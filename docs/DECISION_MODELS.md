# Decision-Model Serving Specification (`/v1/systemone`, `/v1/decisions`)

## 1. Objective & Donor Provenance

Decision models ("System-1 classifiers / structured scorers") evaluate one or more typed decision questions (`noul`, `choice`, `score`) over a shared `state` in **a single prefill forward pass with zero autoregressive decode tokens**, returning calibrated probability distributions, expected scores, and analytical confidence metrics.

`warp-34` exposes two first-class HTTP endpoints backed by one unified decision engine:

1. **`POST /v1/systemone`** (and `POST /systemone`) — TypeSafe System One wire contract.
2. **`POST /v1/decisions`** (and `POST /decisions`) — Ergonomic ordered-array Decisions adapter (`object: "decision.list"`, `decisions[]` + `answers{}` with `"noul" | "choice" | "score"`), translating deterministically onto the same execution & calibration pipeline.

### Explicit Donors
- **`fastino/GLiDE-*` (`GLiDE-mini` 0.8B, `GLiDE-small` 2B, `GLiDE-base` 4B, `GLiDE-large` 9B — `infer.py`, `decision_config.json`, `readout.safetensors`)**:
  - `CodeReadoutHead` (`readout.weight` `[num_codes, hidden_size]` + optional `readout.bias` `[num_codes]`, `lm_head.weight` omitted from checkpoints).
  - `quick-empty-think-medium-v1` / `glide-v1` ChatML prompt ending at `<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: (`, with `(A) key: desc` option lines and `noul` ordered `[true, false]` (`"Yes"`, `"No"`).
  - `noncausal_full_attention` (unmasked full-attention layers) and `midloop` damped-Euler recurrence (`h <- h + step_scale * (f(h) - h)` across `start_layer..=end_layer` for `loops` iterations with GDN `conv_state`/`ssm_state` snapshot & restore on loop `0..loops-1`).
  - `ConfidenceKind::Top1Top2Margin` (`top1 - top2`, or `|p_true - 0.5| * 2` for `noul`; `score` returns integer `score` + continuous `expected_level`).
- **`vllm-sr/d3-*` (`d3-flash` 0.8B, `d3-turbo` 4B, `d3-ultra` 9B — `infer.py`, `decision_config.json`, `readout.safetensors`)**:
  - `d3-code-readout-v1` plain-text prompt (`State:\n{state}\n\nQuestion ({type}): {instructions}\nOptions:\n{code}: {opt}\n...\n\nAnswer:`), with `noul` ordered `[false, true]` (`"No / false"`, `"Yes / true"`).
  - `state_once: true` + `prompt_layout: "header-v1"` (`Questions in this request:\n1. (type) ...\n\nState:\n...` followed by all question blocks in a single sequence, tapping each question's final `Answer:` token).
  - `prefix_bidir: true` (full bidirectional attention across the `[0, prefix_bidir_len)` state prefix, and causal attention across the `[prefix_bidir_len, seq_len)` question suffix via exact boundary chunk splitting).
  - `temperature_by_questions` bucket lookup (`max_questions` -> `temperature`).
  - `ConfidenceKind::NormalizedEntropy` (`1 - H(p) / ln(K)`).
- **`CausalLm` (`togethercomputer/Tev1-*` & general Qwen 3.5 / 3.6 / 3.8 generative scoring)**:
  - Last-row `lm_head` candidate-token logits at single-token labels `A..X` after `</think>\n\nAnswer: `.

---

## 2. Supported Model Family Scope on `warp-34`

`warp-34`'s inference spine is purpose-built for Qwen 3.5 / 3.6 / 3.8 hybrid GDN + paged full-attention (+ MoE + Vision) models. Decision profiles are auto-detected at model load time from `decision_config.json`, `config.json`, and `readout.safetensors`:

| Profile | Detection Rule | Backbone | Readout Mechanism | Max Options & Special Rules |
|---|---|---|---|---|
| **`CodeReadout` (`GLiDE`)** | `readout.safetensors` / `decision_config.json` with `prompt_format` in `{"quick-empty-think-medium-v1", "glide-v1"}` (default `CodeReadoutStyle::Glide`) | Qwen 3.5 hybrid (`fastino/GLiDE-mini`, `GLiDE-small`, `GLiDE-base`, `GLiDE-large`) | Post-final-`GemmaRMSNorm` hidden state at the final prompt token projected via `CodeReadoutHead` (`readout.weight` `[num_codes, hidden_size]`); `lm_head` optional (`None`) | **2–`num_codes` options** (up to 64); `noul` `true`-first; optional `noncausal_full_attention` and `midloop` damped-Euler recurrence; `ConfidenceKind::Top1Top2Margin`; optional `order_passes: 2` forward+reverse option permutation averaging. |
| **`CodeReadout` (`d3`)** | `readout.safetensors` / `decision_config.json` with `prompt == "d3"`, `format_id == "d3-code-readout-v1"`, `state_once`, or `prefix_bidir` (`CodeReadoutStyle::D3`) | Qwen 3.5 hybrid (`vllm-sr/d3-flash`, `d3-turbo`, `d3-ultra`) | Post-final-`GemmaRMSNorm` hidden states at each question's `Answer:` token projected via `CodeReadoutHead`; `lm_head` optional (`None`) | **2–`num_codes` options** (up to 64); `noul` `false`-first; `state_once` multi-question single-prefill tap; `header-v1` question summary prefix; `prefix_bidir` bidirectional state attention; `temperature_by_questions`; `ConfidenceKind::NormalizedEntropy`; optional `order_passes: 2`. |
| **`CausalLm` (Default Qwen Generative Scorer)** | Standard `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` without `readout.safetensors` | Qwen 3.5 / 3.6 / 3.8 dense or MoE | Last-row `lm_head` candidate-token logits at single-token labels `A..X` | **2–24 options**; `"state"` serialized first in user JSON (`{"state": ..., "question": ..., "options": [...]}`) + `<think>\n\n</think>\n\nAnswer: ` assistant prefix; `ConfidenceKind::NormalizedEntropy`; optional `order_passes: 2`. |

### Explicit Scope Boundaries
- **PPLX v1.2 (`perplexity-ai/pplx-decider-v1.2`)**: Checkpoints with `joint_config.json`, `format_version >= 2`, or `format_id == "joint-question-inline-readout-v1"` are explicitly deferred and fail fast at load time with a clear error message.
- **Non-Qwen Encoder Backbones**: Bidirectional encoder-only models (`ModernBERT`, `DeBERTa`) fail fast in `ModelConfig::from_json`.

---

## 3. Wire Schemas

### 3.1 `POST /v1/systemone` (TypeSafe System One Contract)

#### Request
```json
{
  "model": "optional-model-id",
  "state": "Arbitrary text, JSON object, or JSON array",
  "images": ["data:image/png;base64,..."],
  "order_passes": 1,
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
- `order_passes`: optional integer (`1` or `2`, default `1`). When `order_passes == 2`, both `CodeReadout` and `CausalLm` evaluate forward (`0..K`) and reversed (`K-1..=0`) option orders and average the resulting probability distributions to cancel option-order bias.
- `questions`: required non-empty JSON object (`1..=64` entries; key order preserved deterministically):
  - `type == "noul"`: `instructions` required (non-blank); `criteria` optional object with optional `"true"` and `"false"` string descriptions (defaults: `"Yes"` / `"No"`, with `true` ordered first on `GLiDE` and `false` ordered first on `d3` and `CausalLm`).
  - `type == "choice"`: `instructions` required; `criteria` required non-empty object (`2..=max_options` entries, keys non-blank, values `string` or `null`).
  - `type == "score"`: `instructions` required; `criteria` required array of `2..=max_options` level descriptions (`string`), indexed `"0"`, `"1"`, ..., `"K-1"`.

### 3.2 `POST /v1/decisions` (Ergonomic Array/Object Adapter)

Translates ordered-array or simplified decision payloads (`state` or `context`, `order_passes`, `decisions[]` or `questions`) onto the exact same internal `prepare_decision_batch` + execution pipeline.

---

## 4. Engine Seams (`noncausal_full_attention`, `prefix_bidir`, `midloop`, Optional `lm_head`)

1. **Optional `lm_head` & `language_model.*` Weight Prefixes**:
   - `SafetensorsArchive::resolve_name` resolves bare `language_model.layers.*` / `language_model.embed_tokens.*` / `language_model.norm.*` tensor names used by `GLiDE` and `d3` checkpoints alongside standard `model.language_model.*` and `model.layers.*` prefixes.
   - `Qwen3_5ModelRank::lm_head` is `Option<LinearProj>`. When `tie_word_embeddings == false` and no `lm_head` tensor exists in the checkpoint, `warp-34` skips allocating the vocab projection (~1–2 GiB saved) and suppresses `selected_rows` (`emit_mask = false`) on `HiddenPositions`-only prefill batches.
2. **`noncausal_full_attention` & `prefix_bidir` State-Once Execution**:
   - Full-attention layers use `W34_ATTN_MASK_SYMMETRIC_WINDOW` (`sliding_window = 0`) when `cfg.noncausal_full_attention` is true or when `ScheduledBatchSlice::prefix_bidir` is true for the current prefill chunk.
   - When `prefix_bidir_len = Some(P)` is set on a `d3` `state_once` request, `SchedulerV1::schedule_step` and `ScheduledVision::step_prefill` split prefill chunks at the exact token boundary `P` (`prefix_bidir = true` for `start_pos + len <= P`, `prefix_bidir = false` for `start_pos >= P`) and disable `MambaRadixCache` prefix donation for bidirectional prefixes.
3. **`midloop` Damped-Euler Recurrence (`w34_midloop_damped_euler_f16`)**:
   - When `cfg.midloop` (`start_layer`, `end_layer`, `passes > 1`) is configured on a `GLiDE` model, `Qwen3_5ModelRank::run_decoder_layers_into` snapshots `x_0` and each GDN layer's `(conv_state, ssm_state)` before `start_layer`.
   - After each pass through `start_layer..=end_layer`, it applies `w34_midloop_damped_euler_f16` (`h <- x_i + (1 / passes) * (f(x_i) - x_i)` with `alpha = 1.0 / passes as f32`); on passes `0..passes - 1`, it updates `x_{i+1} <- h`, restores GDN states, and re-evaluates `start_layer.input_layernorm` before the next pass.
   - On the final pass (`pass_idx == passes - 1`), GDN states and KV pages remain at their post-pass values and the following layer (`end_layer + 1` or final norm) normalizes the damped-Euler updated residual.
