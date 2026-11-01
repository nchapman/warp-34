# Generative APIs Specification (`/v1/responses`, `/v1/messages`, Token Counting)

## 1. Objective & Donor Provenance

This specification defines `warp-34`'s native support for first-tier agentic generative protocols alongside `/v1/chat/completions`:

1. **OpenAI Responses API**:
   - `POST /v1/responses` (sync JSON & typed SSE stream)
   - `GET /v1/responses/{response_id}` / `DELETE /v1/responses/{response_id}` / `POST /v1/responses/{response_id}/cancel` (bounded in-memory `ResponseStore` when `store` is enabled)
   - `POST /v1/responses/input_tokens` (and `/responses/input_tokens`)
   - `POST /v1/chat/completions/input_tokens` (and `/chat/completions/input_tokens`)
2. **Anthropic Messages API**:
   - `POST /v1/messages` (sync JSON & typed SSE stream)
   - `POST /v1/messages/count_tokens`

### Explicit Donors
- **`vLLM 0.31.0` (`8352b2427`)**:
  - `vllm/entrypoints/openai/responses/{protocol.py,serving.py}` — Responses wire types, `sequence_number` SSE ordering, output item lifecycle (`reasoning`, `message`, `function_call`), `previous_response_id` state chaining, and `usage` accounting (`cached_tokens`, `reasoning_tokens`).
  - `vllm/entrypoints/anthropic/{protocol.py,serving.py}` — Anthropic Messages wire types, content-block normalization, `x-anthropic-billing-header` `cch=` stripping for prefix-cache stability, `usage.input_tokens = max(0, prompt_tokens - cache_read_input_tokens)`, SSE block lifecycle, and Anthropic error envelopes.
- **`llama.cpp` (`tools/server/`)**:
  - `server-chat.cpp` (`server_chat_convert_responses_to_chatcmpl`, `server_chat_convert_anthropic_to_oai`) and `server-context.cpp:5813-5853` (`handle_count_tokens`) — lightweight request canonicalization onto the internal chat/Jinja/parser/grammar spine and exact token-count response payloads.

---

## 2. Internal Architecture & Shared Pipeline (`crates/w34-host/src/server/`)

All generative endpoints translate into `warp-34`'s single canonical prompt-preparation and execution spine so token counts, `MambaRadixCache` prefix reuse, `Qwen3Parser` incremental `<think>` / `<tool_call>` parsing, vision token expansion, and `TokenBitmaskConstraint` grammar enforcement behave identically across protocols:

```text
  POST /v1/chat/completions ──┐
  POST /v1/responses        ──┼──► CanonicalChatRequest ──► prepare_chat_prompt()
  POST /v1/messages         ──┘         │                        │
                                        │                        ├─► Token-count routes return prompt_ids.len()
                                        ▼                        ▼
                               resolve_request_constraint()  EngineHandle::submit_streaming()
                                        │                        │
                                        └───────────┬────────────┘
                                                    ▼
                                     Qwen3StreamParser + Protocol Formatter
                                  (Chat SSE | Responses SSE | Anthropic SSE)
```

### Module Layout
- `crates/w34-host/src/server/mod.rs` — Axum router registration, auth middleware (`Authorization: Bearer <token>` and Anthropic `x-api-key: <token>` accepted equivalently when `--api-key` is configured; `anthropic-version` and `anthropic-beta` headers tolerated), shared `PreparedPrompt` builder, and `/v1/chat/completions` + `*/input_tokens`.
- `crates/w34-host/src/server/responses.rs` — OpenAI Responses request/response types, bounded `ResponseStore` (`Mutex<VecDeque<StoredResponse>>`, capacity 256), `previous_response_id` history stitching, sync JSON builder, and 0-indexed `sequence_number` SSE state machine.
- `crates/w34-host/src/server/anthropic.rs` — Anthropic Messages request/response types, `cch=` billing-header normalization, `thinking` / `tool_use` / `tool_result` / `image` block translation, sync JSON builder, typed Anthropic SSE state machine, `/v1/messages/count_tokens`, and Anthropic error envelopes.

### Invariants Preserved Across All Protocols
1. **Stateful `tool_choice: "none"` safety invariant**: Whenever `tools` is non-empty and `tool_choice` resolves to `"none"`, `resolve_request_constraint()` attaches `TokenBitmaskConstraint::from_disallowed_tokens` banning `<tool_call>` token emission (`op_sampler.cu` fast masked greedy / full masked sampler), while keeping tool schemas rendered in the system prompt so `MambaRadixCache` prefixes remain byte-identical across turns toggling `tool_choice`.
2. **Remote image SSRF gate**: Remote `http://` / `https://` image URLs in Responses `input_image` and Anthropic `image.source.type == "url"` flow through the existing `resolve_image_bytes(allow_remote_images)` check (rejected with HTTP 400 unless `--allow-remote-images` is enabled; `data:image/...;base64,...` and Anthropic `source.type == "base64"` always work offline).
3. **Incremental `Qwen3StreamParser` reuse**: Both `/v1/responses` and `/v1/messages` consume `StreamDelta { reasoning_content, content }` chunks during generation and `parse_assistant_output()` at completion to emit structured tool-call items/blocks and accurate `reasoning_tokens` counts.

---

## 3. Token-Counting Endpoints

All three token-counting routes execute `prepare_chat_prompt()` (including Jinja chat template rendering, tool schema injection, `enable_thinking` prompt suffixing, and multimodal `<|vision_start|><|image_pad|>...<|vision_end|>` token expansion) and return immediately without allocating KV/SSM blocks or scheduling a GPU forward pass.

| Route | Request Body | Success Response (`200 OK`) | Error Envelope |
|---|---|---|---|
| `POST /v1/chat/completions/input_tokens`<br>`POST /chat/completions/input_tokens` | Same schema as `POST /v1/chat/completions` (`messages`, optional `tools`, `chat_template_kwargs`, multimodal parts) | `{"object": "response.input_tokens", "input_tokens": <u64>}` | OpenAI `{"error": ...}` |
| `POST /v1/responses/input_tokens`<br>`POST /responses/input_tokens` | Same schema as `POST /v1/responses` (`input`, `instructions`, `previous_response_id`, `tools`, `reasoning`) | `{"object": "response.input_tokens", "input_tokens": <u64>}` | OpenAI `{"error": ...}` |
| `POST /v1/messages/count_tokens` | Same schema as `POST /v1/messages` (`messages`, optional `system`, `tools`, `thinking`; `max_tokens` optional on `count_tokens`) | `{"input_tokens": <u64>}` | Anthropic `{"type": "error", "error": ...}` |

---

## 4. OpenAI Responses API (`/v1/responses`)

### 4.1 Request Schema & Translation

| Field | Accepted Shapes | Canonical Mapping |
|---|---|---|
| `model` | `Option<String>` | Validated against loaded model ID/alias when strict model matching applies; echoed in response (`served_model_name` default). |
| `instructions` | `Option<String>` | Prepended as leading `role: "system"` message. |
| `previous_response_id` | `Option<String>` | Looks up prior `StoredResponse` in `ResponseStore` matching both the 128-bit CSPRNG `resp_<32 hex>` bearer capability token and the request's `Authorization` / `x-api-key` credential partition (`404 not_found_error` if unknown or mismatched) and prepends its normalized input + assistant output turns before current `input`. |
| `input` | `String` or `Vec<ResponseInputItem>` | `String` -> single `user` message. Array items support:<br>- `type: "message"` (`role`: `user`/`assistant`/`system`/`developer`, `content`: string or typed parts `input_text`, `output_text`, `input_image`); `developer` maps to `system`.<br>- Bare `{"role": ..., "content": ...}` chat-shaped items.<br>- `type: "function_call"` (`call_id`, `name`, `arguments`) -> merged onto preceding/new `assistant` message `tool_calls`.<br>- `type: "function_call_output"` (`call_id`, `output`: string or content array) -> `role: "tool"` message with `tool_call_id = call_id`.<br>- `type: "reasoning"` (`content` / `summary`) -> attached as `reasoning_content` on the associated `assistant` turn. |
| `tools` | Flat `{"type":"function","name":...,"description":...,"parameters":...,"strict":...}` or Chat-nested `{"type":"function","function":{...}}` | Normalized to `ChatTool`; non-`function` built-in tools (`web_search_preview`, `file_search`, `code_interpreter`) rejected with HTTP 400 `invalid_request_error`. |
| `tool_choice` | `"none"` \| `"auto"` \| `"required"` \| `{"type":"function","name":"..."}` (or nested `function.name`) | Mapped to internal `ToolChoice`; `"none"` with tools activates `<tool_call>` ban bitmask; `"required"` or named function activates tool-call JSON grammar when `thinking` is disabled. |
| `max_output_tokens` | `Option<usize>` | Mapped to `max_tokens` (reject `< 1` with HTTP 400). |
| `reasoning` | `Option<{ effort: Option<String>, summary: Option<String> }>` | `effort == Some("none")` sets `enable_thinking = false`; `"low" \| "medium" \| "high"` sets `enable_thinking = true`; omitted follows model default. |
| `text.format` | `{"type": "text"}` \| `{"type": "json_object"}` \| `{"type": "json_schema", "name": "...", "schema": {...}, "strict": ...}` | Mapped directly to `ResponseFormat` -> `llguidance` `TokenBitmaskConstraint` (validating mutual exclusion with active thinking when strict JSON grammar is requested, or applying post-`</think>` / `enable_thinking=false` per existing `/v1/chat/completions` behavior). |
| `temperature`, `top_p`, `top_k`, `stop` | Optional sampling params | Mapped directly to `SamplingParams`. |
| `stream` | `bool` (default `false`) | Selects sync JSON vs typed `text/event-stream` SSE. |
| `store` | `bool` (default `false`) | Explicit opt-in persistence: requires a non-empty `Authorization` or `x-api-key` header (HTTP 400 otherwise) and stores the completed response under a 128-bit CSPRNG `resp_<32 hex>` ID in `ResponseStore`, partitioned by credential scope with per-scope eviction quotas. |
| `parallel_tool_calls`, `metadata`, `truncation`, `user` | Optional metadata/control fields | Echoed in `ResponseObject` (`metadata` default `{}`). |

### 4.2 Sync Response Object (`object: "response"`)

```json
{
  "id": "resp_<hex>",
  "object": "response",
  "created_at": 1760000000,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": 512,
  "model": "Qwen3.5-9B-W4A16-GPTQ",
  "output": [
    {
      "id": "rs_<hex>",
      "type": "reasoning",
      "status": "completed",
      "summary": [],
      "content": [{"type": "reasoning_text", "text": "..."}]
    },
    {
      "id": "msg_<hex>",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "...", "annotations": []}]
    },
    {
      "id": "fc_<hex>",
      "type": "function_call",
      "status": "completed",
      "call_id": "call_<hex>",
      "name": "read_file",
      "arguments": "{\"path\":\"src/lib.rs\"}"
    }
  ],
  "output_text": "...",
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {"effort": null, "summary": null},
  "store": true,
  "temperature": 0.0,
  "text": {"format": {"type": "text"}},
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 128,
    "input_tokens_details": {"cached_tokens": 64},
    "output_tokens": 42,
    "output_tokens_details": {"reasoning_tokens": 18},
    "total_tokens": 170
  },
  "metadata": {}
}
```

- When generation stops due to `FinishReason::Length`, `status` is `"incomplete"` and `incomplete_details` is `{"reason": "max_output_tokens"}`.
- Empty `reasoning` items are omitted when no reasoning text was emitted; empty `message` items are omitted when the turn consists solely of `function_call` items (unless neither text nor tool calls were produced, in which case an empty `output_text` message item is emitted).

### 4.3 Streaming SSE State Machine (`POST /v1/responses` with `"stream": true`)

Every SSE frame sets `event: <type>` and carries `"type": "<type>"` plus a strictly monotonic 0-indexed `"sequence_number"` in its JSON `data:`:

1. `response.created` (`sequence_number: 0`, `response.status = "in_progress"`, `output: []`, `usage: null`)
2. `response.in_progress` (`sequence_number: 1`, `response.status = "in_progress"`)
3. **Reasoning item** (opened lazily on first non-empty `reasoning_content` delta):
   - `response.output_item.added` (`output_index`, `item: { id: "rs_...", type: "reasoning", status: "in_progress", summary: [], content: [] }`)
   - `response.reasoning_part.added` (`item_id`, `output_index`, `content_index: 0`, `part: { type: "reasoning_text", text: "" }`)
   - `response.reasoning_text.delta` (`item_id`, `output_index`, `content_index: 0`, `delta`) — repeated per chunk
   - `response.reasoning_text.done` (`item_id`, `output_index`, `content_index: 0`, `text: full_reasoning`)
   - `response.reasoning_part.done` (`item_id`, `output_index`, `content_index: 0`, `part: { type: "reasoning_text", text: full_reasoning }`)
   - `response.output_item.done` (`output_index`, `item: { ..., status: "completed", content: [...] }`)
4. **Assistant text message item** (opened lazily on first non-empty `content` delta, closing any open reasoning item first):
   - `response.output_item.added` (`output_index`, `item: { id: "msg_...", type: "message", status: "in_progress", role: "assistant", content: [] }`)
   - `response.content_part.added` (`item_id`, `output_index`, `content_index: 0`, `part: { type: "output_text", text: "", annotations: [] }`)
   - `response.output_text.delta` (`item_id`, `output_index`, `content_index: 0`, `delta`, `logprobs: []`) — repeated per chunk
   - `response.output_text.done` (`item_id`, `output_index`, `content_index: 0`, `text: full_text`, `logprobs: []`)
   - `response.content_part.done` (`item_id`, `output_index`, `content_index: 0`, `part: { type: "output_text", text: full_text, annotations: [] }`)
   - `response.output_item.done` (`output_index`, `item: { ..., status: "completed", content: [...] }`)
5. **Function-call item(s)** (emitted for each parsed `ToolCall` from `Qwen3StreamParser`, closing any open reasoning/message item first):
   - `response.output_item.added` (`output_index`, `item: { id: "fc_...", type: "function_call", status: "in_progress", call_id: "call_...", name, arguments: "" }`)
   - `response.function_call_arguments.delta` (`item_id`, `output_index`, `call_id`, `delta: arguments_json`)
   - `response.function_call_arguments.done` (`item_id`, `output_index`, `call_id`, `name`, `arguments: arguments_json`)
   - `response.output_item.done` (`output_index`, `item: { ..., status: "completed", arguments: arguments_json }`)
6. **Terminal event**:
   - `response.completed` (with full `response` object including `output` and `usage`) on normal/tool stop,
   - `response.incomplete` (`status: "incomplete"`, `incomplete_details: {"reason": "max_output_tokens"}`) on `FinishReason::Length`, or
   - `response.failed` (`status: "failed"`, `error: {"code": "server_error", "message": "..."}`) on engine failure.
   - Stream closes cleanly after the terminal event (no `[DONE]` frame, matching the OpenAI Responses specification).

---

## 5. Anthropic Messages API (`/v1/messages`)

### 5.1 Request Schema & Translation

| Field | Accepted Shapes | Canonical Mapping |
|---|---|---|
| `model` | `String` | Validated/echoed in response. |
| `max_tokens` | `usize` (required on `/v1/messages`, optional on `/v1/messages/count_tokens`) | Must be `>= 1` (HTTP 400 `invalid_request_error` if `0` or missing on `/v1/messages`). |
| `system` | `Option<String \| Vec<AnthropicSystemBlock>>` | Text blocks concatenated with `"\n\n"` into leading `system` message; `cache_control` accepted and ignored; leading `x-anthropic-billing-header: ...` lines have `cch=[0-9a-fA-F]+;` stripped (matching `vLLM` `serving.py` so Claude Code per-request billing hashes do not invalidate `MambaRadixCache` prefixes!). |
| `messages` | `Vec<AnthropicMessage>` (`role`: `"user"` \| `"assistant"`, non-empty) | Consecutive same-role turns (e.g. multiple `tool_result` blocks in `user` content) map cleanly:<br>- `text` -> text content<br>- `image` (`source.type == "base64"` with `media_type` + `data`, or `source.type == "url"` with `url`) -> `data:{media_type};base64,{data}` or remote URL via `resolve_image_bytes(allow_remote_images)`<br>- `thinking` (`thinking`, optional `signature`) -> assistant `reasoning_content`<br>- `redacted_thinking` (`data`) -> ignored/preserved safely without corrupting prompt<br>- `tool_use` (`id`, `name`, `input`: JSON object) -> assistant `tool_calls` entry (`arguments = serde_json::to_string(&input)`)<br>- `tool_result` (`tool_use_id`, `content`: string or array of `text`/`image` blocks, optional `is_error`) -> `role: "tool"` message(s) (`tool_call_id = tool_use_id`), preserving any embedded images in a follow-up user multimodal turn if present. |
| `tools` | `Option<Vec<{ name, description, input_schema, cache_control, type }>>` | Mapped to `ChatTool` (`parameters = input_schema`). Non-custom server tools (e.g. `computer_20250124`, `bash_20250124` without `input_schema`) return HTTP 400 `invalid_request_error` unless `input_schema` is provided. |
| `tool_choice` | `{"type": "auto"}` \| `{"type": "any"}` \| `{"type": "none"}` \| `{"type": "tool", "name": "..."}` (plus optional `disable_parallel_tool_use`) | `auto` -> `Auto`; `any` -> `Required`; `none` -> `None` (with stateful `<tool_call>` ban bitmask!); `tool` -> named function. |
| `thinking` | `{"type": "enabled", "budget_tokens": N}` \| `{"type": "disabled"}` \| `{"type": "adaptive"}` | `enabled`: validates `budget_tokens >= 1024` and `budget_tokens < max_tokens` (HTTP 400 `invalid_request_error` otherwise) and rejects forced `tool_choice` (`any` or `tool`) with HTTP 400 `invalid_request_error` per Anthropic API spec; sets `enable_thinking = true`. `disabled` sets `enable_thinking = false`. `adaptive` sets `enable_thinking = true` unless forced `tool_choice` is active. |
| `output_config.format` | `Option<{ type: "json_schema", schema: {...} }>` | Mapped to `ResponseFormat::JsonSchema` -> `TokenBitmaskConstraint`. |
| `stop_sequences` | `Option<Vec<String>>` | Mapped to `SamplingParams::stop` and tracks matched `stop_sequence` for `stop_reason: "stop_sequence"`. |
| `temperature`, `top_p`, `top_k`, `stream`, `metadata` | Optional parameters | Mapped to `SamplingParams` (`temperature` validated in `[0.0, 2.0]`, `top_p` in `(0.0, 1.0]`). |

### 5.2 Sync Response Object (`type: "message"`)

```json
{
  "id": "msg_<hex>",
  "type": "message",
  "role": "assistant",
  "model": "Qwen3.5-9B-W4A16-GPTQ",
  "content": [
    {
      "type": "thinking",
      "thinking": "...",
      "signature": "w34-thinking-v1"
    },
    {
      "type": "text",
      "text": "..."
    },
    {
      "type": "tool_use",
      "id": "toolu_<hex>",
      "name": "read_file",
      "input": {"path": "src/lib.rs"}
    }
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 64,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 64,
    "output_tokens": 42
  }
}
```

- **Anthropic `usage` invariant** (`vLLM` `anthropic/serving.py:66-96`): Anthropic SDKs compute total prompt tokens as `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`. Therefore `cache_read_input_tokens = cached_tokens`, `cache_creation_input_tokens = 0`, and `input_tokens = prompt_tokens.saturating_sub(cached_tokens)`.
- **`stop_reason` mapping**:
  - Parsed tool call(s) present -> `"tool_use"`
  - `FinishReason::Length` -> `"max_tokens"`
  - Matched user `stop_sequences` entry -> `"stop_sequence"` (with `stop_sequence: Some(matched)`)
  - Normal EOS -> `"end_turn"`

### 5.3 Streaming SSE State Machine (`POST /v1/messages` with `"stream": true`)

Every frame uses `event: <event_name>\ndata: <json>\n\n`:

1. `event: message_start`
   - `data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","model":"...","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":<uncached>,"cache_creation_input_tokens":0,"cache_read_input_tokens":<cached>,"output_tokens":0}}}`
2. **Content blocks** (`index` increments `0, 1, 2, ...` across emitted blocks):
   - **Thinking block** (opened lazily on first non-empty `reasoning_content` delta):
     - `event: content_block_start` -> `{"type":"content_block_start","index":i,"content_block":{"type":"thinking","thinking":""}}`
     - `event: content_block_delta` -> `{"type":"content_block_delta","index":i,"delta":{"type":"thinking_delta","thinking":"..."}}`
     - Before closing: `event: content_block_delta` -> `{"type":"content_block_delta","index":i,"delta":{"type":"signature_delta","signature":"w34-thinking-v1"}}`
     - `event: content_block_stop` -> `{"type":"content_block_stop","index":i}`
   - **Text block** (opened lazily on first non-empty `content` delta, closing any open thinking block first):
     - `event: content_block_start` -> `{"type":"content_block_start","index":i,"content_block":{"type":"text","text":""}}`
     - `event: content_block_delta` -> `{"type":"content_block_delta","index":i,"delta":{"type":"text_delta","text":"..."}}`
     - `event: content_block_stop` -> `{"type":"content_block_stop","index":i}`
   - **Tool-use block(s)** (one block per parsed `ToolCall`, closing any open thinking/text block first; `input` parsed to JSON object or `{}` fallback):
     - `event: content_block_start` -> `{"type":"content_block_start","index":i,"content_block":{"type":"tool_use","id":"toolu_...","name":"...","input":{}}}`
     - `event: content_block_delta` -> `{"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":"{...}"}}`
     - `event: content_block_stop` -> `{"type":"content_block_stop","index":i}`
3. `event: message_delta`
   - `data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":<N>}}`
4. `event: message_stop`
   - `data: {"type":"message_stop"}`
5. Mid-stream engine error:
   - `event: error` -> `{"type":"error","error":{"type":"api_error","message":"..."}}`

### 5.4 Error Envelopes

- **Anthropic endpoints (`/v1/messages`, `/v1/messages/count_tokens`)**:
  ```json
  {
    "type": "error",
    "error": {
      "type": "invalid_request_error",
      "message": "..."
    }
  }
  ```
  Status-to-type mapping: `400 -> invalid_request_error`, `401 -> authentication_error`, `404 -> not_found_error`, `429/503 -> overloaded_error`, `500 -> api_error`.
- **OpenAI endpoints (`/v1/responses*`, `/v1/chat/completions*`)**:
  ```json
  {
    "error": {
      "message": "...",
      "type": "invalid_request_error",
      "param": null,
      "code": null
    }
  }
  ```

---

## 6. Verification Matrix

1. **Unit tests (CPU, `cargo test --lib`)**:
   - Request translation & validation for `/v1/responses` (string vs typed items, flat vs nested tools, `previous_response_id` stitching & 404, `tool_choice: "none"` bitmask activation, `text.format` JSON schema constraint, `reasoning.effort` mapping).
   - Request translation & validation for `/v1/messages` (`system` blocks + `cch=` stripping, `thinking.budget_tokens` bounds & forced-`tool_choice` conflict 400, `tool_use` + `tool_result` multi-turn round-trip, `image` base64 + SSRF remote-URL gate, `usage` cached-token subtraction, Anthropic error envelope JSON).
   - SSE frame sequence verification for both protocols (monotonic 0-indexed `sequence_number` on `/v1/responses`; exact `message_start -> content_block_start -> content_block_delta -> content_block_stop -> message_delta -> message_stop` ordering on `/v1/messages` across thinking + text + tool calls).
2. **End-to-end HTTP + GPU tests (`--features cuda`)**:
   - Live Axum router backed by `Qwen3.5-4B-W4A16-GPTQ` exercising sync + streaming `/v1/responses` and `/v1/messages` (plain text, `<think>` reasoning blocks, structured tool calls, `tool_choice: "none"` safety, JSON schema `text.format` / `output_config`, `previous_response_id` multi-turn cache hit, and exact agreement between `/v1/responses/input_tokens`, `/v1/messages/count_tokens`, `/v1/chat/completions/input_tokens`, and actual generation `usage`).
