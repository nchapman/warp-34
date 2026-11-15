export type HttpMethod = "GET" | "POST" | "DELETE";

export type EndpointKind =
  | "openai-chat"
  | "openai-responses"
  | "anthropic-messages"
  | "openai-completions"
  | "decisions"
  | "utility";

export interface TurnMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_call_id?: string;
  tool_calls?: unknown[];
  reasoning_content?: string;
  rawContentBlocks?: unknown[];
}

export interface EndpointPreset {
  id: string;
  label: string;
  description: string;
  buildBody: (model: string) => Record<string, unknown>;
}

export interface EndpointSpec {
  id: string;
  group: "Generative" | "Decision & Scoring" | "Tokenizer & Counts" | "System";
  method: HttpMethod;
  path: string;
  label: string;
  kind: EndpointKind;
  conversational: boolean;
  supportsStream: boolean;
  presets: EndpointPreset[];
}

export interface SseFrame {
  index: number;
  elapsedMs: number;
  event: string | null;
  data: string;
}

export interface ParsedToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface ParsedApiResult {
  reasoning: string;
  content: string;
  toolCalls: ParsedToolCall[];
  finishReason: string | null;
  responseId: string | null;
  streamError: string | null;
  usage: {
    inputTokens: number | null;
    cachedTokens: number | null;
    outputTokens: number | null;
    reasoningTokens: number | null;
  };
}

// 8x8 warm-amber checkerboard PNG data URI for offline multimodal Vision testing
const SAMPLE_IMAGE_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12P4//8/A00wCgD/fQH5n3b68AAAAABJRU5ErkJggg==";

export const ENDPOINTS: EndpointSpec[] = [
  {
    id: "chat-completions",
    group: "Generative",
    method: "POST",
    path: "/v1/chat/completions",
    label: "POST /v1/chat/completions",
    kind: "openai-chat",
    conversational: true,
    supportsStream: true,
    presets: [
      {
        id: "standard",
        label: "Standard Chat (SSE)",
        description: "Streaming OpenAI Chat Completion with usage reporting",
        buildBody: (model) => ({
          model,
          messages: [
            {
              role: "system",
              content: "You are a concise systems engineering assistant.",
            },
            {
              role: "user",
              content:
                "Explain why warp-level reductions use __shfl_down_sync instead of shared memory.",
            },
          ],
          max_tokens: 256,
          temperature: 0,
          stream: true,
          stream_options: { include_usage: true },
          chat_template_kwargs: { enable_thinking: false },
        }),
      },
      {
        id: "thinking",
        label: "Reasoning / Thinking",
        description: "Streams reasoning_content followed by final answer",
        buildBody: (model) => ({
          model,
          messages: [
            {
              role: "user",
              content:
                "A 24 GiB GPU holds 18.4 GiB of weights and uses 21,120 bytes/token for KV cache. How many full 16-token pages fit in 4.0 GiB of KV budget?",
            },
          ],
          max_tokens: 512,
          temperature: 0.6,
          top_p: 0.95,
          stream: true,
          stream_options: { include_usage: true },
          chat_template_kwargs: { enable_thinking: true },
        }),
      },
      {
        id: "tools",
        label: "Tool Calling",
        description: "Grammar-constrained function tool invocation",
        buildBody: (model) => ({
          model,
          messages: [
            {
              role: "user",
              content:
                "Check the GPU temperature and free VRAM on device 0 before launching the job.",
            },
          ],
          tools: [
            {
              type: "function",
              function: {
                name: "query_gpu_status",
                description: "Read live telemetry for a CUDA device index",
                parameters: {
                  type: "object",
                  properties: {
                    device_index: { type: "integer", minimum: 0 },
                    include_clocks: { type: "boolean" },
                  },
                  required: ["device_index"],
                  additionalProperties: false,
                },
              },
            },
          ],
          tool_choice: "auto",
          max_tokens: 256,
          temperature: 0,
          stream: true,
          stream_options: { include_usage: true },
          chat_template_kwargs: { enable_thinking: false },
        }),
      },
      {
        id: "json-schema",
        label: "Structured Output (JSON Schema)",
        description: "Strict token-bitmask JSON Schema enforcement",
        buildBody: (model) => ({
          model,
          messages: [
            {
              role: "user",
              content:
                "Produce a kernel launch configuration for a 4096x4096 FP16 GEMM.",
            },
          ],
          response_format: {
            type: "json_schema",
            json_schema: {
              name: "kernel_launch",
              strict: true,
              schema: {
                type: "object",
                properties: {
                  kernel_name: { type: "string" },
                  grid_dim: {
                    type: "array",
                    items: { type: "integer" },
                  },
                  block_dim: {
                    type: "array",
                    items: { type: "integer" },
                  },
                  shared_mem_bytes: { type: "integer" },
                },
                required: [
                  "kernel_name",
                  "grid_dim",
                  "block_dim",
                  "shared_mem_bytes",
                ],
                additionalProperties: false,
              },
            },
          },
          max_tokens: 256,
          temperature: 0,
          stream: false,
          chat_template_kwargs: { enable_thinking: false },
        }),
      },
      {
        id: "vision",
        label: "Multimodal Vision (Data URL)",
        description:
          "Inline PNG data URL + text prompt for Vision-enabled checkpoints",
        buildBody: (model) => ({
          model,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: { url: SAMPLE_IMAGE_DATA_URL },
                },
                {
                  type: "text",
                  text: "Describe the colors and brightness of this sample image.",
                },
              ],
            },
          ],
          max_tokens: 128,
          temperature: 0,
          stream: true,
          stream_options: { include_usage: true },
          chat_template_kwargs: { enable_thinking: false },
        }),
      },
    ],
  },
  {
    id: "responses",
    group: "Generative",
    method: "POST",
    path: "/v1/responses",
    label: "POST /v1/responses",
    kind: "openai-responses",
    conversational: true,
    supportsStream: true,
    presets: [
      {
        id: "responses-stream",
        label: "Responses Stream (Stateful)",
        description:
          "OpenAI Responses API with store:true for previous_response_id chaining",
        buildBody: (model) => ({
          model,
          instructions: "Answer concisely with concrete technical details.",
          input: [
            {
              role: "user",
              content:
                "What is the difference between GDN recurrent state and paged KV cache?",
            },
          ],
          max_output_tokens: 256,
          temperature: 0,
          store: true,
          stream: true,
        }),
      },
      {
        id: "responses-tools",
        label: "Responses Function Tool",
        description: "Flat function tool declaration on /v1/responses",
        buildBody: (model) => ({
          model,
          input: [
            {
              role: "user",
              content: "Run the sanitizer gate on the paged_fa2 target.",
            },
          ],
          tools: [
            {
              type: "function",
              name: "run_test_target",
              description: "Execute a Cargo integration test binary",
              parameters: {
                type: "object",
                properties: {
                  target: { type: "string" },
                  sanitize: { type: "boolean" },
                },
                required: ["target", "sanitize"],
                additionalProperties: false,
              },
            },
          ],
          max_output_tokens: 256,
          temperature: 0,
          stream: false,
        }),
      },
    ],
  },
  {
    id: "anthropic-messages",
    group: "Generative",
    method: "POST",
    path: "/v1/messages",
    label: "POST /v1/messages",
    kind: "anthropic-messages",
    conversational: true,
    supportsStream: true,
    presets: [
      {
        id: "messages-stream",
        label: "Anthropic Messages (SSE)",
        description: "Native /v1/messages with cache_read_input_tokens usage",
        buildBody: (model) => ({
          model,
          system: "You are a precise Rust + CUDA compiler engineer.",
          messages: [
            {
              role: "user",
              content:
                "Give a 3-bullet checklist for avoiding register spills in CUDA kernels.",
            },
          ],
          thinking: { type: "disabled" },
          max_tokens: 256,
          temperature: 0,
          stream: true,
        }),
      },
      {
        id: "messages-thinking",
        label: "Extended Thinking",
        description:
          "Anthropic thinking block (budget_tokens < max_tokens) + text streaming",
        buildBody: (model) => ({
          model,
          messages: [
            {
              role: "user",
              content:
                "Compare PCIe pinned-sysmem two-hop vs NVLink P2P for 10 KB all-reduce payloads.",
            },
          ],
          thinking: {
            type: "enabled",
            budget_tokens: 1024,
          },
          max_tokens: 2048,
          stream: true,
        }),
      },
    ],
  },
  {
    id: "completions",
    group: "Generative",
    method: "POST",
    path: "/v1/completions",
    label: "POST /v1/completions",
    kind: "openai-completions",
    conversational: false,
    supportsStream: true,
    presets: [
      {
        id: "raw-completion",
        label: "Raw Prompt + Logprobs",
        description: "Legacy /v1/completions with top token logprobs",
        buildBody: (model) => ({
          model,
          prompt: 'pub unsafe extern "C" fn w34_all_reduce_sum_f16(',
          max_tokens: 64,
          temperature: 0,
          logprobs: 3,
          stream: false,
        }),
      },
    ],
  },
  {
    id: "decisions",
    group: "Decision & Scoring",
    method: "POST",
    path: "/v1/decisions",
    label: "POST /v1/decisions",
    kind: "decisions",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "decisions-multi",
        label: "Multi-Decision Batch (noul + choice)",
        description: "Shared-prefix evaluation over noul and choice fields",
        buildBody: (model) => ({
          model,
          context:
            "PR #412 modifies crates/w34-kernels/csrc/ops/op_paged_attn.cu to add FP8 E4M3 tail anchors on sm_89+. All 117 compute-sanitizer runs pass.",
          decisions: [
            {
              id: "requires_gpu_ci",
              type: "noul",
              question: "Does this change require GPU hardware verification?",
              criteria: {
                true: "Touches CUDA kernels, FFI, or device memory layout",
                false: "Documentation or pure CPU change only",
              },
            },
            {
              id: "risk_tier",
              type: "choice",
              question: "Classify the deployment risk tier.",
              options: {
                low: "Isolated surgical change with full sanitizer coverage",
                medium: "Cross-subsystem runtime change",
                high: "Unverified kernel or ABI change",
              },
            },
          ],
        }),
      },
      {
        id: "decisions-calibrated",
        label: "Calibrated 2-Pass + Score Field",
        description:
          "Bidirectional order_passes:2 option-order calibration with ordinal score rubric",
        buildBody: (model) => ({
          model,
          order_passes: 2,
          context:
            "Incident report: Node 0 GPU temperature is 64C, ECC error count is 0, NVLink P2P bandwidth measured at 478 GB/s unidirectional.",
          decisions: [
            {
              id: "node_health_score",
              type: "score",
              question: "Rate the hardware readiness of Node 0 for production.",
              criteria: [
                "Critical hardware fault or ECC errors present",
                "Degraded thermal or interconnect performance",
                "Healthy thermal, ECC, and NVLink P2P metrics",
              ],
            },
          ],
        }),
      },
    ],
  },
  {
    id: "systemone",
    group: "Decision & Scoring",
    method: "POST",
    path: "/v1/systemone",
    label: "POST /v1/systemone",
    kind: "decisions",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "systemone-canonical",
        label: "SystemOne Canonical Schema",
        description: "Canonical state + questions object format",
        buildBody: (model) => ({
          model,
          state:
            "Customer ticket: 'Our TP=2 node on 2x A100-SXM4 shows NV12 topology in nvidia-smi topo -m.'",
          questions: {
            p2p_capable: {
              type: "noul",
              instructions: "Can this node use NVLink P2P device-VRAM staging?",
              criteria: {
                true: "NVLink P2P topology is present between ranks",
                false: "PCIe host-staging required",
              },
            },
          },
        }),
      },
    ],
  },
  {
    id: "tokenize",
    group: "Tokenizer & Counts",
    method: "POST",
    path: "/tokenize",
    label: "POST /tokenize",
    kind: "utility",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "tokenize-prompt",
        label: "Encode Text -> Token IDs",
        description: "Tokenizes raw text using the loaded HF tokenizer",
        buildBody: () => ({
          prompt: "<|im_start|>user\nHello from warp-34!<|im_end|>",
        }),
      },
    ],
  },
  {
    id: "detokenize",
    group: "Tokenizer & Counts",
    method: "POST",
    path: "/detokenize",
    label: "POST /detokenize",
    kind: "utility",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "detokenize-ids",
        label: "Decode Token IDs -> Text",
        description: "Decodes token IDs back into UTF-8 text",
        buildBody: () => ({
          tokens: [9707, 504, 49689, 12, 18, 19, 0],
        }),
      },
    ],
  },
  {
    id: "chat-input-tokens",
    group: "Tokenizer & Counts",
    method: "POST",
    path: "/v1/chat/completions/input_tokens",
    label: "POST /v1/chat/completions/input_tokens",
    kind: "utility",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "chat-count",
        label: "Count Chat Prompt Tokens",
        description: "Applies ChatML template + tools and counts input tokens",
        buildBody: (model) => ({
          model,
          messages: [
            { role: "system", content: "You are helpful." },
            {
              role: "user",
              content: "How many tokens is this formatted turn?",
            },
          ],
          chat_template_kwargs: { enable_thinking: false },
        }),
      },
    ],
  },
  {
    id: "responses-input-tokens",
    group: "Tokenizer & Counts",
    method: "POST",
    path: "/v1/responses/input_tokens",
    label: "POST /v1/responses/input_tokens",
    kind: "utility",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "responses-count",
        label: "Count Responses Input Tokens",
        description: "Counts input tokens for an OpenAI /v1/responses payload",
        buildBody: (model) => ({
          model,
          instructions: "You are helpful.",
          input: [{ role: "user", content: "Count these input tokens." }],
        }),
      },
    ],
  },
  {
    id: "anthropic-count-tokens",
    group: "Tokenizer & Counts",
    method: "POST",
    path: "/v1/messages/count_tokens",
    label: "POST /v1/messages/count_tokens",
    kind: "utility",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "messages-count",
        label: "Count Anthropic Message Tokens",
        description: "Counts input_tokens for an Anthropic /v1/messages body",
        buildBody: (model) => ({
          model,
          system: "You are helpful.",
          messages: [{ role: "user", content: "Count these input tokens." }],
        }),
      },
    ],
  },
  {
    id: "models",
    group: "System",
    method: "GET",
    path: "/v1/models",
    label: "GET /v1/models",
    kind: "utility",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "list-models",
        label: "List Models & LoRA Adapters",
        description: "Returns base model and registered LoRA adapters",
        buildBody: () => ({}),
      },
    ],
  },
  {
    id: "health",
    group: "System",
    method: "GET",
    path: "/health",
    label: "GET /health",
    kind: "utility",
    conversational: false,
    supportsStream: false,
    presets: [
      {
        id: "health-check",
        label: "Health & Readiness Probe",
        description: "Returns 200 OK when worker is healthy and accepting work",
        buildBody: () => ({}),
      },
    ],
  },
];

export function emptyParsedResult(): ParsedApiResult {
  return {
    reasoning: "",
    content: "",
    toolCalls: [],
    finishReason: null,
    responseId: null,
    streamError: null,
    usage: {
      inputTokens: null,
      cachedTokens: null,
      outputTokens: null,
      reasoningTokens: null,
    },
  };
}

/**
 * Extracts a structured `TurnMessage[]` list from a raw request JSON object
 * for `/v1/chat/completions`, `/v1/responses`, or `/v1/messages`.
 * Non-text blocks (e.g. `image_url`, `tool_use`, `tool_result`) and assistant
 * `tool_calls` / `reasoning_content` are preserved on the `TurnMessage` so
 * editing other turns never strips multimodal or tool metadata.
 */
export function extractMessagesFromBody(
  kind: EndpointKind,
  body: Record<string, unknown>,
): TurnMessage[] {
  if (kind === "openai-chat") {
    const raw = Array.isArray(body.messages) ? body.messages : [];
    return raw.map(normalizeTurnMessage);
  }
  if (kind === "anthropic-messages") {
    const out: TurnMessage[] = [];
    if (typeof body.system === "string" && body.system.length > 0) {
      out.push({ role: "system", content: body.system });
    }
    const raw = Array.isArray(body.messages) ? body.messages : [];
    for (const item of raw) {
      out.push(normalizeTurnMessage(item));
    }
    return out;
  }
  if (kind === "openai-responses") {
    const out: TurnMessage[] = [];
    if (typeof body.instructions === "string" && body.instructions.length > 0) {
      out.push({ role: "system", content: body.instructions });
    }
    if (typeof body.input === "string") {
      out.push({ role: "user", content: body.input });
    } else if (Array.isArray(body.input)) {
      for (const item of body.input) {
        out.push(normalizeTurnMessage(item));
      }
    }
    return out;
  }
  return [];
}

function normalizeTurnMessage(item: unknown): TurnMessage {
  if (!item || typeof item !== "object") {
    return { role: "user", content: "" };
  }
  const obj = item as Record<string, unknown>;
  const rawRole = typeof obj.role === "string" ? obj.role : "user";
  const role: TurnMessage["role"] =
    rawRole === "system" ||
    rawRole === "user" ||
    rawRole === "assistant" ||
    rawRole === "tool"
      ? rawRole
      : "user";
  let content = "";
  let rawContentBlocks: unknown[] | undefined;
  if (typeof obj.content === "string") {
    content = obj.content;
  } else if (Array.isArray(obj.content)) {
    const textParts: string[] = [];
    const nonTextBlocks: unknown[] = [];
    for (const part of obj.content) {
      if (typeof part === "string") {
        textParts.push(part);
      } else if (
        part &&
        typeof part === "object" &&
        (part as { type?: unknown }).type === "text" &&
        typeof (part as { text?: unknown }).text === "string"
      ) {
        textParts.push((part as { text: string }).text);
      } else {
        nonTextBlocks.push(part);
      }
    }
    content = textParts.join("\n");
    if (nonTextBlocks.length > 0) {
      rawContentBlocks = nonTextBlocks;
    }
  }
  const turn: TurnMessage = { role, content };
  if (rawContentBlocks) {
    turn.rawContentBlocks = rawContentBlocks;
  }
  if (typeof obj.tool_call_id === "string") {
    turn.tool_call_id = obj.tool_call_id;
  }
  if (Array.isArray(obj.tool_calls)) {
    turn.tool_calls = obj.tool_calls;
  }
  if (typeof obj.reasoning_content === "string") {
    turn.reasoning_content = obj.reasoning_content;
  }
  return turn;
}

/**
 * Applies an updated `TurnMessage[]` array back onto a cloned request JSON object,
 * preserving any non-text blocks (`image_url`, `tool_use`, `tool_result`) and
 * assistant `tool_calls` / `reasoning_content`.
 */
export function applyMessagesToBody(
  kind: EndpointKind,
  body: Record<string, unknown>,
  messages: TurnMessage[],
): Record<string, unknown> {
  const next = { ...body };
  if (kind === "openai-chat") {
    next.messages = messages.map((m) => {
      const msgObj: Record<string, unknown> = {
        role: m.role,
        content:
          m.rawContentBlocks && m.rawContentBlocks.length > 0
            ? [
                ...m.rawContentBlocks,
                ...(m.content ? [{ type: "text", text: m.content }] : []),
              ]
            : m.content,
      };
      if (m.tool_call_id) {
        msgObj.tool_call_id = m.tool_call_id;
      }
      if (m.tool_calls && m.tool_calls.length > 0) {
        msgObj.tool_calls = m.tool_calls;
      }
      if (m.reasoning_content) {
        msgObj.reasoning_content = m.reasoning_content;
      }
      return msgObj;
    });
    return next;
  }
  if (kind === "anthropic-messages") {
    const sys = messages.filter((m) => m.role === "system");
    const rest = messages.filter((m) => m.role !== "system");
    if (sys.length > 0) {
      next.system = sys.map((s) => s.content).join("\n\n");
    } else {
      delete next.system;
    }
    next.messages = rest.map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content:
        m.rawContentBlocks && m.rawContentBlocks.length > 0
          ? [
              ...m.rawContentBlocks,
              ...(m.content ? [{ type: "text", text: m.content }] : []),
            ]
          : m.content,
    }));
    return next;
  }
  if (kind === "openai-responses") {
    const sys = messages.filter((m) => m.role === "system");
    const rest = messages.filter((m) => m.role !== "system");
    if (sys.length > 0) {
      next.instructions = sys.map((s) => s.content).join("\n\n");
    } else {
      delete next.instructions;
    }
    next.input = rest.map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: m.content,
    }));
    return next;
  }
  return next;
}

/**
 * Appends the assistant response turn (preserving real `tool_calls` and
 * `reasoning_content` when present) and a fresh follow-up `user` or `tool`
 * turn to a conversational request body.
 */
export function appendAssistantTurnToBody(
  kind: EndpointKind,
  body: Record<string, unknown>,
  parsed: ParsedApiResult,
  nextUserContent = "",
): Record<string, unknown> {
  if (kind === "openai-responses" && body.store === true && parsed.responseId) {
    return {
      ...body,
      previous_response_id: parsed.responseId,
      input: [{ role: "user", content: nextUserContent }],
    };
  }
  const current = extractMessagesFromBody(kind, body);
  if (kind === "openai-chat" && parsed.toolCalls.length > 0) {
    const assistantTurn: TurnMessage = {
      role: "assistant",
      content: parsed.content,
      tool_calls: parsed.toolCalls.map((tc) => ({
        id: tc.id,
        type: "function",
        function: { name: tc.name, arguments: tc.arguments },
      })),
      ...(parsed.reasoning ? { reasoning_content: parsed.reasoning } : {}),
    };
    const toolTurns: TurnMessage[] = parsed.toolCalls.map((tc, idx) => ({
      role: "tool",
      tool_call_id: tc.id,
      content:
        idx === 0 && nextUserContent.trim().length > 0
          ? nextUserContent
          : '{"status":"ok"}',
    }));
    return applyMessagesToBody(kind, body, [
      ...current,
      assistantTurn,
      ...toolTurns,
    ]);
  }
  const assistantTurn: TurnMessage = {
    role: "assistant",
    content: parsed.content,
    ...(kind === "openai-chat" && parsed.reasoning
      ? { reasoning_content: parsed.reasoning }
      : {}),
  };
  const updated: TurnMessage[] = [
    ...current,
    assistantTurn,
    { role: "user", content: nextUserContent },
  ];
  return applyMessagesToBody(kind, body, updated);
}

/**
 * Parses a non-streaming JSON response from any supported endpoint into `ParsedApiResult`.
 */
export function parseJsonResponse(
  kind: EndpointKind,
  payload: unknown,
): ParsedApiResult {
  const res = emptyParsedResult();
  if (!payload || typeof payload !== "object") {
    return res;
  }
  const obj = payload as Record<string, unknown>;

  if (typeof obj.id === "string") {
    res.responseId = obj.id;
  }

  if (kind === "openai-chat") {
    const choices = Array.isArray(obj.choices) ? obj.choices : [];
    const first = choices[0] as Record<string, unknown> | undefined;
    if (first) {
      if (typeof first.finish_reason === "string") {
        res.finishReason = first.finish_reason;
      }
      const msg = first.message as Record<string, unknown> | undefined;
      if (msg) {
        if (typeof msg.reasoning_content === "string") {
          res.reasoning = msg.reasoning_content;
        } else if (typeof msg.reasoning === "string") {
          res.reasoning = msg.reasoning;
        }
        if (typeof msg.content === "string") {
          res.content = msg.content;
        }
        if (Array.isArray(msg.tool_calls)) {
          res.toolCalls = msg.tool_calls.map((tc, idx) => {
            const t = (tc ?? {}) as Record<string, unknown>;
            const fn = (t.function ?? {}) as Record<string, unknown>;
            return {
              id: typeof t.id === "string" ? t.id : `call_${idx}`,
              name: typeof fn.name === "string" ? fn.name : "function",
              arguments: typeof fn.arguments === "string" ? fn.arguments : "{}",
            };
          });
        }
      }
    }
    extractOpenAiUsage(obj.usage, res);
    return res;
  }

  if (kind === "openai-completions") {
    const choices = Array.isArray(obj.choices) ? obj.choices : [];
    const first = choices[0] as Record<string, unknown> | undefined;
    if (first) {
      if (typeof first.finish_reason === "string") {
        res.finishReason = first.finish_reason;
      }
      if (typeof first.text === "string") {
        res.content = first.text;
      }
    }
    extractOpenAiUsage(obj.usage, res);
    return res;
  }

  if (kind === "openai-responses") {
    if (typeof obj.status === "string") {
      res.finishReason = obj.status;
    }
    const errObj = obj.error as Record<string, unknown> | undefined;
    if (errObj && typeof errObj.message === "string") {
      res.streamError = errObj.message;
    }
    if (typeof obj.output_text === "string" && obj.output_text.length > 0) {
      res.content = obj.output_text;
    }
    const output = Array.isArray(obj.output) ? obj.output : [];
    const textParts: string[] = [];
    const reasoningParts: string[] = [];
    for (const item of output) {
      if (!item || typeof item !== "object") continue;
      const it = item as Record<string, unknown>;
      if (it.type === "reasoning") {
        if (Array.isArray(it.content)) {
          for (const c of it.content) {
            if (
              c &&
              typeof c === "object" &&
              typeof (c as { text?: unknown }).text === "string"
            ) {
              reasoningParts.push((c as { text: string }).text);
            }
          }
        }
        if (reasoningParts.length === 0 && Array.isArray(it.summary)) {
          for (const s of it.summary) {
            if (
              s &&
              typeof s === "object" &&
              typeof (s as { text?: unknown }).text === "string"
            ) {
              reasoningParts.push((s as { text: string }).text);
            }
          }
        }
      } else if (it.type === "message" && Array.isArray(it.content)) {
        for (const c of it.content) {
          if (
            c &&
            typeof c === "object" &&
            typeof (c as { text?: unknown }).text === "string"
          ) {
            textParts.push((c as { text: string }).text);
          }
        }
      } else if (it.type === "function_call") {
        res.toolCalls.push({
          id: typeof it.call_id === "string" ? it.call_id : "call_0",
          name: typeof it.name === "string" ? it.name : "function",
          arguments: typeof it.arguments === "string" ? it.arguments : "{}",
        });
      }
    }
    if (textParts.length > 0 && !res.content) {
      res.content = textParts.join("");
    }
    if (reasoningParts.length > 0) {
      res.reasoning = reasoningParts.join("\n");
    }
    extractResponsesUsage(obj.usage, res);
    return res;
  }

  if (kind === "anthropic-messages") {
    if (typeof obj.stop_reason === "string") {
      res.finishReason = obj.stop_reason;
    }
    const content = Array.isArray(obj.content) ? obj.content : [];
    for (const block of content) {
      if (!block || typeof block !== "object") continue;
      const b = block as Record<string, unknown>;
      if (b.type === "thinking" && typeof b.thinking === "string") {
        res.reasoning += b.thinking;
      } else if (b.type === "text" && typeof b.text === "string") {
        res.content += b.text;
      } else if (b.type === "tool_use") {
        res.toolCalls.push({
          id: typeof b.id === "string" ? b.id : "toolu_0",
          name: typeof b.name === "string" ? b.name : "tool",
          arguments: JSON.stringify(b.input ?? {}),
        });
      }
    }
    extractAnthropicUsage(obj.usage, res);
    return res;
  }

  if (kind === "decisions") {
    const usage = obj.usage as Record<string, unknown> | undefined;
    if (usage) {
      if (typeof usage.input_tokens === "number") {
        res.usage.inputTokens = usage.input_tokens;
      }
      if (typeof usage.cached_tokens === "number") {
        res.usage.cachedTokens = usage.cached_tokens;
      }
      if (typeof usage.output_tokens === "number") {
        res.usage.outputTokens = usage.output_tokens;
      }
    }
    res.content = JSON.stringify(obj.decisions ?? obj.answers ?? obj, null, 2);
    return res;
  }

  res.content = JSON.stringify(obj, null, 2);
  return res;
}

function extractOpenAiUsage(usageRaw: unknown, res: ParsedApiResult) {
  if (!usageRaw || typeof usageRaw !== "object") return;
  const u = usageRaw as Record<string, unknown>;
  if (typeof u.prompt_tokens === "number") {
    res.usage.inputTokens = u.prompt_tokens;
  }
  if (typeof u.completion_tokens === "number") {
    res.usage.outputTokens = u.completion_tokens;
  }
  const details = u.prompt_tokens_details as
    Record<string, unknown> | undefined;
  if (details && typeof details.cached_tokens === "number") {
    res.usage.cachedTokens = details.cached_tokens;
  } else if (typeof u.cached_tokens === "number") {
    res.usage.cachedTokens = u.cached_tokens;
  }
}

function extractResponsesUsage(usageRaw: unknown, res: ParsedApiResult) {
  if (!usageRaw || typeof usageRaw !== "object") return;
  const u = usageRaw as Record<string, unknown>;
  if (typeof u.input_tokens === "number") {
    res.usage.inputTokens = u.input_tokens;
  }
  if (typeof u.output_tokens === "number") {
    res.usage.outputTokens = u.output_tokens;
  }
  const inDetails = u.input_tokens_details as
    Record<string, unknown> | undefined;
  if (inDetails && typeof inDetails.cached_tokens === "number") {
    res.usage.cachedTokens = inDetails.cached_tokens;
  }
  const outDetails = u.output_tokens_details as
    Record<string, unknown> | undefined;
  if (outDetails && typeof outDetails.reasoning_tokens === "number") {
    res.usage.reasoningTokens = outDetails.reasoning_tokens;
  }
}

function extractAnthropicUsage(usageRaw: unknown, res: ParsedApiResult) {
  if (!usageRaw || typeof usageRaw !== "object") return;
  const u = usageRaw as Record<string, unknown>;
  const uncached = typeof u.input_tokens === "number" ? u.input_tokens : null;
  const cached =
    typeof u.cache_read_input_tokens === "number"
      ? u.cache_read_input_tokens
      : null;
  if (uncached !== null) {
    res.usage.inputTokens = uncached + (cached ?? 0);
  }
  if (cached !== null) {
    res.usage.cachedTokens = cached;
  }
  if (typeof u.output_tokens === "number") {
    res.usage.outputTokens = u.output_tokens;
  }
}

/**
 * Folds an incremental SSE frame (`event`, `data`) into `ParsedApiResult`.
 */
export function applySseFrameToResult(
  kind: EndpointKind,
  current: ParsedApiResult,
  frame: SseFrame,
): ParsedApiResult {
  if (frame.data === "[DONE]") {
    return current;
  }
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(frame.data) as Record<string, unknown>;
  } catch {
    return current;
  }

  const next: ParsedApiResult = {
    ...current,
    toolCalls: current.toolCalls.map((tc) => ({ ...tc })),
    usage: { ...current.usage },
  };

  if (typeof parsed.id === "string" && !next.responseId) {
    next.responseId = parsed.id;
  }

  // Surface mid-stream error frames across OpenAI, Anthropic, and Responses streams
  if (parsed.error && typeof parsed.error === "object") {
    const errMsg = (parsed.error as { message?: unknown }).message;
    if (typeof errMsg === "string" && errMsg.length > 0) {
      next.streamError = errMsg;
    }
  }

  if (kind === "openai-chat") {
    const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
    const first = choices[0] as Record<string, unknown> | undefined;
    if (first) {
      if (typeof first.finish_reason === "string") {
        next.finishReason = first.finish_reason;
      }
      const delta = first.delta as Record<string, unknown> | undefined;
      if (delta) {
        if (typeof delta.reasoning_content === "string") {
          next.reasoning += delta.reasoning_content;
        } else if (typeof delta.reasoning === "string") {
          next.reasoning += delta.reasoning;
        }
        if (typeof delta.content === "string") {
          next.content += delta.content;
        }
        if (Array.isArray(delta.tool_calls)) {
          for (const item of delta.tool_calls) {
            if (!item || typeof item !== "object") continue;
            const tc = item as Record<string, unknown>;
            const idx = typeof tc.index === "number" ? tc.index : 0;
            while (next.toolCalls.length <= idx) {
              next.toolCalls.push({
                id: `call_${next.toolCalls.length}`,
                name: "",
                arguments: "",
              });
            }
            if (typeof tc.id === "string" && tc.id) {
              next.toolCalls[idx].id = tc.id;
            }
            const fn = tc.function as Record<string, unknown> | undefined;
            if (fn) {
              if (typeof fn.name === "string") {
                next.toolCalls[idx].name += fn.name;
              }
              if (typeof fn.arguments === "string") {
                next.toolCalls[idx].arguments += fn.arguments;
              }
            }
          }
        }
      }
    }
    extractOpenAiUsage(parsed.usage, next);
    return next;
  }

  if (kind === "openai-completions") {
    const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
    const first = choices[0] as Record<string, unknown> | undefined;
    if (first) {
      if (typeof first.finish_reason === "string") {
        next.finishReason = first.finish_reason;
      }
      if (typeof first.text === "string") {
        next.content += first.text;
      }
    }
    extractOpenAiUsage(parsed.usage, next);
    return next;
  }

  if (kind === "openai-responses") {
    const evType =
      typeof parsed.type === "string" ? parsed.type : (frame.event ?? "");
    if (
      (evType === "response.reasoning_text.delta" ||
        evType === "response.reasoning_summary_text.delta") &&
      typeof parsed.delta === "string"
    ) {
      next.reasoning += parsed.delta;
    } else if (
      evType === "response.output_text.delta" &&
      typeof parsed.delta === "string"
    ) {
      next.content += parsed.delta;
    } else if (
      evType === "response.completed" ||
      evType === "response.incomplete" ||
      evType === "response.failed" ||
      evType === "response.cancelled"
    ) {
      const respObj = parsed.response as Record<string, unknown> | undefined;
      if (respObj) {
        const finalParsed = parseJsonResponse("openai-responses", respObj);
        return {
          reasoning: next.reasoning || finalParsed.reasoning,
          content: next.content || finalParsed.content,
          toolCalls:
            finalParsed.toolCalls.length > 0
              ? finalParsed.toolCalls
              : next.toolCalls,
          finishReason: finalParsed.finishReason ?? next.finishReason,
          responseId: finalParsed.responseId ?? next.responseId,
          streamError: finalParsed.streamError ?? next.streamError,
          usage: finalParsed.usage,
        };
      }
    }
    return next;
  }

  if (kind === "anthropic-messages") {
    const evType =
      typeof parsed.type === "string" ? parsed.type : (frame.event ?? "");
    if (evType === "message_start" && parsed.message) {
      const msg = parsed.message as Record<string, unknown>;
      if (typeof msg.id === "string") next.responseId = msg.id;
      extractAnthropicUsage(msg.usage, next);
    } else if (evType === "content_block_start" && parsed.content_block) {
      const block = parsed.content_block as Record<string, unknown>;
      if (block.type === "tool_use") {
        next.toolCalls.push({
          id: typeof block.id === "string" ? block.id : "toolu_0",
          name: typeof block.name === "string" ? block.name : "tool",
          arguments: "",
        });
      }
    } else if (evType === "content_block_delta" && parsed.delta) {
      const delta = parsed.delta as Record<string, unknown>;
      if (
        delta.type === "thinking_delta" &&
        typeof delta.thinking === "string"
      ) {
        next.reasoning += delta.thinking;
      } else if (
        delta.type === "text_delta" &&
        typeof delta.text === "string"
      ) {
        next.content += delta.text;
      } else if (
        delta.type === "input_json_delta" &&
        typeof delta.partial_json === "string"
      ) {
        const last = next.toolCalls.at(-1);
        if (last) {
          last.arguments += delta.partial_json;
        }
      }
    } else if (evType === "message_delta") {
      const delta = parsed.delta as Record<string, unknown> | undefined;
      if (delta && typeof delta.stop_reason === "string") {
        next.finishReason = delta.stop_reason;
      }
      const u = parsed.usage as Record<string, unknown> | undefined;
      if (u && typeof u.output_tokens === "number") {
        next.usage.outputTokens = u.output_tokens;
      }
      if (u && typeof u.cache_read_input_tokens === "number") {
        next.usage.cachedTokens = u.cache_read_input_tokens;
      }
    }
    return next;
  }

  return next;
}

/**
 * Splits a raw SSE stream chunk buffer into complete SSE events + remaining partial tail.
 */
export function drainSseBuffer(
  buffer: string,
  startIndex: number,
  elapsedMs: number,
): { frames: SseFrame[]; remainder: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const remainder = blocks.pop() ?? "";
  const frames: SseFrame[] = [];
  let idx = startIndex;

  for (const block of blocks) {
    const lines = block.split("\n");
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (dataLines.length > 0) {
      frames.push({
        index: idx++,
        elapsedMs,
        event,
        data: dataLines.join("\n"),
      });
    }
  }
  return { frames, remainder };
}

/**
 * Builds a reproducible `curl` command for the current endpoint and JSON body.
 */
export function formatCurlCommand(
  origin: string,
  endpoint: EndpointSpec,
  rawJson: string,
  apiKey: string,
): string {
  const url = `${origin}${endpoint.path}`;
  let isStreaming = false;
  let needsStatefulScope = false;
  try {
    const parsed = JSON.parse(rawJson) as Record<string, unknown>;
    isStreaming = Boolean(parsed.stream);
    needsStatefulScope =
      endpoint.kind === "openai-responses" &&
      (parsed.store === true ||
        typeof parsed.previous_response_id === "string");
  } catch {
    // Ignore invalid JSON when formatting cURL
  }
  const flag = isStreaming ? "-sSN" : "-sS";
  const lines: string[] = [
    endpoint.method === "GET"
      ? `curl ${flag} '${url}'`
      : `curl ${flag} -X ${endpoint.method} '${url}'`,
  ];
  if (endpoint.method !== "GET") {
    lines.push(`  -H 'Content-Type: application/json'`);
  }
  const effectiveKey =
    apiKey.trim().length > 0
      ? apiKey.trim()
      : needsStatefulScope
        ? "playground-local"
        : "";
  if (effectiveKey.length > 0) {
    if (endpoint.kind === "anthropic-messages") {
      lines.push(`  -H 'x-api-key: ${shellEscapeSingleQuotes(effectiveKey)}'`);
    } else {
      lines.push(
        `  -H 'Authorization: Bearer ${shellEscapeSingleQuotes(effectiveKey)}'`,
      );
    }
  }
  if (endpoint.method !== "GET" && rawJson.trim().length > 0) {
    lines.push(`  -d '${shellEscapeSingleQuotes(rawJson.trim())}'`);
  }
  return lines.join(" \\\n");
}

function shellEscapeSingleQuotes(value: string): string {
  return value.replace(/'/g, `'\\''`);
}
