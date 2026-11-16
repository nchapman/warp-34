import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Dashboard } from "./App";
import { snapshot } from "./fixtures";
import {
  ENDPOINTS,
  appendAssistantTurnToBody,
  applyMessagesToBody,
  applySseFrameToResult,
  drainSseBuffer,
  emptyParsedResult,
  extractMessagesFromBody,
  formatCurlCommand,
  parseJsonResponse,
} from "./playground";

beforeEach(() => {
  window.history.replaceState(null, "", "/");
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("playground wire helpers", () => {
  it("extracts and updates messages across OpenAI Chat, Responses, and Anthropic", () => {
    const chatBody = {
      model: "qwen",
      messages: [
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
      ],
    };
    const extractedChat = extractMessagesFromBody("openai-chat", chatBody);
    expect(extractedChat).toEqual([
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
    ]);

    const anthroBody = applyMessagesToBody(
      "anthropic-messages",
      { model: "qwen" },
      [
        { role: "system", content: "be concise" },
        { role: "user", content: "hello" },
        { role: "assistant", content: "world" },
      ],
    );
    expect(anthroBody.system).toBe("be concise");
    expect(anthroBody.messages).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "world" },
    ]);

    const respBody = applyMessagesToBody(
      "openai-responses",
      { model: "qwen" },
      [
        { role: "system", content: "instructions text" },
        { role: "user", content: "prompt text" },
      ],
    );
    expect(respBody.instructions).toBe("instructions text");
    expect(respBody.input).toEqual([{ role: "user", content: "prompt text" }]);
  });

  it("chains previous_response_id on stateful /v1/responses multi-turn append", () => {
    const parsed = {
      ...emptyParsedResult(),
      content: "GDN uses fixed-size recurrent state.",
      responseId: "resp_abc123",
    };
    const next = appendAssistantTurnToBody(
      "openai-responses",
      {
        model: "qwen",
        store: true,
        input: [{ role: "user", content: "turn 1" }],
      },
      parsed,
      "turn 2 follow-up",
    );
    expect(next.previous_response_id).toBe("resp_abc123");
    expect(next.input).toEqual([{ role: "user", content: "turn 2 follow-up" }]);
  });

  it("parses SSE frames for OpenAI Chat reasoning, content, tool calls, and cached usage", () => {
    const rawSse = [
      'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning_content":"step 1. "}}]}\n\n',
      'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Answer: 42"}}]}\n\n',
      'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"check_gpu","arguments":"{\\"dev\\":0}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":120,"completion_tokens":18,"prompt_tokens_details":{"cached_tokens":96}}}\n\n',
      "data: [DONE]\n\n",
    ].join("");

    const { frames, remainder } = drainSseBuffer(rawSse, 0, 25);
    expect(remainder).toBe("");
    expect(frames).toHaveLength(4);

    let state = emptyParsedResult();
    for (const f of frames) {
      state = applySseFrameToResult("openai-chat", state, f);
    }
    expect(state.reasoning).toBe("step 1. ");
    expect(state.content).toBe("Answer: 42");
    expect(state.finishReason).toBe("tool_calls");
    expect(state.toolCalls).toEqual([
      { id: "call_9", name: "check_gpu", arguments: '{"dev":0}' },
    ]);
    expect(state.usage.inputTokens).toBe(120);
    expect(state.usage.cachedTokens).toBe(96);
    expect(state.usage.outputTokens).toBe(18);
  });

  it("parses SSE frames for Anthropic Messages and OpenAI Responses", () => {
    const anthroFrames = drainSseBuffer(
      [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":20,"cache_read_input_tokens":64}}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"pondering"}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello anthropic"}}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9,"cache_read_input_tokens":64}}\n\n',
      ].join(""),
      0,
      10,
    ).frames;

    let anthroState = emptyParsedResult();
    for (const f of anthroFrames) {
      anthroState = applySseFrameToResult("anthropic-messages", anthroState, f);
    }
    expect(anthroState.reasoning).toBe("pondering");
    expect(anthroState.content).toBe("hello anthropic");
    expect(anthroState.finishReason).toBe("end_turn");
    expect(anthroState.usage.inputTokens).toBe(84);
    expect(anthroState.usage.cachedTokens).toBe(64);
    expect(anthroState.usage.outputTokens).toBe(9);

    const respFrames = drainSseBuffer(
      [
        'event: response.reasoning_text.delta\ndata: {"type":"response.reasoning_text.delta","delta":"thinking step "}\n\n',
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"streamed text"}\n\n',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_77","status":"completed","output":[{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"thinking step "}]},{"type":"message","content":[{"type":"output_text","text":"streamed text"}]}],"usage":{"input_tokens":50,"output_tokens":12,"input_tokens_details":{"cached_tokens":32},"output_tokens_details":{"reasoning_tokens":4}}}}\n\n',
      ].join(""),
      0,
      15,
    ).frames;

    let respState = emptyParsedResult();
    for (const f of respFrames) {
      respState = applySseFrameToResult("openai-responses", respState, f);
    }
    expect(respState.responseId).toBe("resp_77");
    expect(respState.reasoning).toBe("thinking step ");
    expect(respState.content).toBe("streamed text");
    expect(respState.finishReason).toBe("completed");
    expect(respState.usage.cachedTokens).toBe(32);
    expect(respState.usage.reasoningTokens).toBe(4);
  });

  it("preserves multimodal image_url blocks and structured tool_calls across multi-turn edits", () => {
    const chatEp = ENDPOINTS.find((e) => e.id === "chat-completions")!;
    const visionPreset = chatEp.presets.find((p) => p.id === "vision")!;
    const initialBody = visionPreset.buildBody("qwen-vl");

    const turns = extractMessagesFromBody("openai-chat", initialBody);
    expect(turns).toHaveLength(1);
    expect(turns[0].images).toHaveLength(1);
    expect(turns[0].content).toContain("Describe the colors");

    // Edit text of the multimodal turn and ensure image_url block survives round-trip
    turns[0].content = "Updated vision prompt";
    const roundTripped = applyMessagesToBody("openai-chat", initialBody, turns);
    const msg0 = (roundTripped.messages as Array<Record<string, unknown>>)[0];
    expect(Array.isArray(msg0.content)).toBe(true);
    expect(msg0.content).toEqual([
      {
        type: "image_url",
        image_url: { url: expect.stringContaining("data:image/png;base64,") },
      },
      { type: "text", text: "Updated vision prompt" },
    ]);

    // Verify Anthropic and Responses serialization of attached images
    const anthroVision = applyMessagesToBody(
      "anthropic-messages",
      { model: "qwen-vl" },
      turns,
    );
    const anthroMsg0 = (
      anthroVision.messages as Array<Record<string, unknown>>
    )[0];
    expect(anthroMsg0.content).toEqual([
      {
        type: "image",
        source: {
          type: "base64",
          media_type: "image/png",
          data: expect.any(String),
        },
      },
      { type: "text", text: "Updated vision prompt" },
    ]);

    const respVision = applyMessagesToBody(
      "openai-responses",
      { model: "qwen-vl" },
      turns,
    );
    const respMsg0 = (respVision.input as Array<Record<string, unknown>>)[0];
    expect(respMsg0.content).toEqual([
      {
        type: "input_image",
        image_url: expect.stringContaining("data:image/png;base64,"),
      },
      { type: "input_text", text: "Updated vision prompt" },
    ]);

    // Verify tool call continuation appends structured assistant.tool_calls + role:"tool"
    const afterToolCall = appendAssistantTurnToBody(
      "openai-chat",
      {
        model: "qwen",
        messages: [{ role: "user", content: "Check GPU 0" }],
      },
      {
        ...emptyParsedResult(),
        reasoning: "Need to query device 0.",
        toolCalls: [
          {
            id: "call_gpu0",
            name: "query_gpu_status",
            arguments: '{"device_index":0}',
          },
        ],
      },
      '{"temp_c":58,"free_mib":5120}',
    );
    const msgs = afterToolCall.messages as Array<Record<string, unknown>>;
    expect(msgs).toHaveLength(3);
    expect(msgs[1].role).toBe("assistant");
    expect(msgs[1].reasoning_content).toBe("Need to query device 0.");
    expect(msgs[1].tool_calls).toEqual([
      {
        id: "call_gpu0",
        type: "function",
        function: {
          name: "query_gpu_status",
          arguments: '{"device_index":0}',
        },
      },
    ]);
    expect(msgs[2]).toEqual({
      role: "tool",
      tool_call_id: "call_gpu0",
      content: '{"temp_c":58,"free_mib":5120}',
    });

    // Verify Anthropic tool_use + tool_result continuation
    const anthroAfterTool = appendAssistantTurnToBody(
      "anthropic-messages",
      {
        model: "qwen",
        messages: [{ role: "user", content: "Check GPU 0" }],
      },
      {
        ...emptyParsedResult(),
        toolCalls: [
          {
            id: "toolu_1",
            name: "query_gpu_status",
            arguments: '{"device_index":0}',
          },
        ],
      },
      '{"temp_c":58}',
    );
    const anthroMsgs = anthroAfterTool.messages as Array<
      Record<string, unknown>
    >;
    expect(anthroMsgs).toHaveLength(3);
    expect(anthroMsgs[1]).toEqual({
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "toolu_1",
          name: "query_gpu_status",
          input: { device_index: 0 },
        },
      ],
    });
    expect(anthroMsgs[2]).toEqual({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "toolu_1",
          content: '{"temp_c":58}',
        },
      ],
    });

    // Verify out-of-range SSE tool_call index does not throw or allocate huge arrays
    const badIdxFrame = {
      index: 0,
      elapsedMs: 1,
      event: null,
      data: '{"choices":[{"delta":{"tool_calls":[{"index":-1,"id":"c1","function":{"name":"fn","arguments":"{}"}},{"index":99999999,"function":{"arguments":"{\\"x\\":1}"}}]}}]}',
    };
    const safeFolded = applySseFrameToResult(
      "openai-chat",
      emptyParsedResult(),
      badIdxFrame,
    );
    expect(safeFolded.toolCalls).toHaveLength(1);
    expect(safeFolded.toolCalls[0].name).toBe("fn");
  });

  it("enforces Anthropic thinking budget_tokens < max_tokens and stateful Responses scope", () => {
    const anthroEp = ENDPOINTS.find((e) => e.id === "anthropic-messages")!;
    const thinkingPreset = anthroEp.presets.find(
      (p) => p.id === "messages-thinking",
    )!;
    const body = thinkingPreset.buildBody("qwen");
    const thinking = body.thinking as { type: string; budget_tokens: number };
    expect(thinking.type).toBe("enabled");
    expect(thinking.budget_tokens).toBeGreaterThanOrEqual(1024);
    expect(body.max_tokens as number).toBeGreaterThan(thinking.budget_tokens);

    const respEp = ENDPOINTS.find((e) => e.id === "responses")!;
    const statefulBody = JSON.stringify(respEp.presets[0].buildBody("qwen"));
    const curlNoKey = formatCurlCommand(
      "http://127.0.0.1:8001",
      respEp,
      statefulBody,
      "",
    );
    expect(curlNoKey).toContain("curl -sSN");
    expect(curlNoKey).toContain("Authorization: Bearer playground-local");
  });

  it("formats cURL commands with Bearer or x-api-key headers and escaped quotes", () => {
    const chatEp = ENDPOINTS.find((e) => e.id === "chat-completions")!;
    const anthroEp = ENDPOINTS.find((e) => e.id === "anthropic-messages")!;

    const chatCurl = formatCurlCommand(
      "http://127.0.0.1:8000",
      chatEp,
      '{"prompt":"it\'s ok"}',
      "secret-token",
    );
    expect(chatCurl).toContain("Authorization: Bearer secret-token");
    expect(chatCurl).toContain("'http://127.0.0.1:8000/v1/chat/completions'");

    const anthroCurl = formatCurlCommand(
      "http://127.0.0.1:8000",
      anthroEp,
      "{}",
      "anthro-key",
    );
    expect(anthroCurl).toContain("x-api-key: anthro-key");
  });

  it("parses non-streaming decision responses with cached token counts", () => {
    const parsed = parseJsonResponse("decisions", {
      model: "glide",
      decisions: [{ id: "q1", answer: true, confidence: 0.99 }],
      usage: { input_tokens: 140, cached_tokens: 128, output_tokens: 0 },
    });
    expect(parsed.usage.inputTokens).toBe(140);
    expect(parsed.usage.cachedTokens).toBe(128);
    expect(parsed.content).toContain('"answer": true');

    const compSync = parseJsonResponse("openai-completions", {
      id: "cmpl-1",
      choices: [{ text: "void* out,", finish_reason: "length" }],
      usage: { prompt_tokens: 12, completion_tokens: 4, cached_tokens: 8 },
    });
    expect(compSync.content).toBe("void* out,");
    expect(compSync.finishReason).toBe("length");
    expect(compSync.usage.cachedTokens).toBe(8);

    const compFrames = drainSseBuffer(
      [
        'data: {"id":"cmpl-2","choices":[{"text":"int n"}]}\n\n',
        'data: {"id":"cmpl-2","choices":[{"text":");","finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}\n\n',
        "data: [DONE]\n\n",
      ].join(""),
      0,
      5,
    ).frames;
    let compStream = emptyParsedResult();
    for (const f of compFrames) {
      compStream = applySseFrameToResult("openai-completions", compStream, f);
    }
    expect(compStream.content).toBe("int n);");
    expect(compStream.finishReason).toBe("stop");
    expect(compStream.usage.outputTokens).toBe(3);
  });
});

describe("Playground UI integration", () => {
  it("switches to Playground, syncs structured turns with raw JSON, and executes requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: new Headers({ "content-type": "application/json" }),
      text: async () =>
        JSON.stringify({
          id: "chatcmpl-test",
          choices: [
            {
              index: 0,
              finish_reason: "stop",
              message: {
                role: "assistant",
                reasoning_content: "Verified warp shuffle.",
                content: "Use __shfl_down_sync with mask 0xffffffff.",
              },
            },
          ],
          usage: {
            prompt_tokens: 64,
            completion_tokens: 12,
            prompt_tokens_details: { cached_tokens: 48 },
          },
        }),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <Dashboard
        state={{
          history: [snapshot({ model: "nchapman/Qwen3.8-27B-W4A16-HGPTQ" })],
          connection: "live",
          error: null,
        }}
      />,
    );

    fireEvent.click(screen.getByRole("link", { name: "Playground" }));
    expect(
      screen.getByRole("heading", { name: "API Playground" }),
    ).toBeVisible();

    // Edit turn 2 content and verify it reflects in Raw JSON mode
    const turn2 = screen.getByLabelText("Turn 2 content");
    fireEvent.change(turn2, { target: { value: "Custom kernel question" } });

    fireEvent.click(screen.getByRole("tab", { name: "JSON Body" }));
    const rawEditor = screen.getByLabelText(
      "Request JSON body",
    ) as HTMLTextAreaElement;
    expect(rawEditor.value).toContain("nchapman/Qwen3.8-27B-W4A16-HGPTQ");
    expect(rawEditor.value).toContain("Custom kernel question");

    // Send the request and verify parsed output + multi-turn continuation
    fireEvent.click(screen.getByRole("button", { name: "Send Request" }));

    await waitFor(() => {
      expect(
        screen.getByText("Use __shfl_down_sync with mask 0xffffffff."),
      ).toBeVisible();
    });
    expect(screen.getByText("Verified warp shuffle.")).toBeVisible();
    expect(screen.getByText("200 OK")).toBeVisible();

    // Append follow-up turn and send
    const followUp = screen.getByLabelText("Follow-up user message");
    fireEvent.change(followUp, {
      target: { value: "Now show the 32-lane butterfly reduction." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Append & Send" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
    const secondCallBody = JSON.parse(fetchMock.mock.calls[1][1].body);
    expect(secondCallBody.messages).toHaveLength(4);
    expect(secondCallBody.messages[2]).toEqual({
      role: "assistant",
      content: "Use __shfl_down_sync with mask 0xffffffff.",
      reasoning_content: "Verified warp shuffle.",
    });
    expect(secondCallBody.messages[3]).toEqual({
      role: "user",
      content: "Now show the 32-lane butterfly reduction.",
    });
  });

  it("streams text/event-stream SSE responses and records frames in the SSE tab", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"id":"chatcmpl-sse","choices":[{"index":0,"delta":{"content":"Hello "}}]}\n\n',
          ),
        );
        controller.enqueue(
          encoder.encode(
            'data: {"id":"chatcmpl-sse","choices":[{"index":0,"delta":{"content":"SSE!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":32,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":16}}}\n\ndata: [DONE]\n\n',
          ),
        );
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: "OK",
        headers: new Headers({ "content-type": "text/event-stream" }),
        body: stream,
      }),
    );

    render(
      <Dashboard
        state={{
          history: [snapshot({ model: "test/model" })],
          connection: "live",
          error: null,
        }}
      />,
    );

    fireEvent.click(screen.getByRole("link", { name: "Playground" }));
    fireEvent.click(screen.getByRole("button", { name: "Send Request" }));

    await waitFor(() => {
      expect(screen.getByText("Hello SSE!")).toBeVisible();
    });

    const sseTab = screen.getByRole("tab", { name: "SSE (3)" });
    fireEvent.click(sseTab);
    expect(screen.getByText("[DONE]")).toBeVisible();
  });

  it("validates invalid JSON, handles GET endpoints, and surfaces HTTP/network errors", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: "Unauthorized",
        headers: new Headers({ "content-type": "application/json" }),
        text: async () =>
          JSON.stringify({ error: { message: "invalid api key" } }),
      })
      .mockRejectedValueOnce(new Error("Connection refused"));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <Dashboard
        state={{
          history: [snapshot({ model: "test/model" })],
          connection: "live",
          error: null,
        }}
      />,
    );

    fireEvent.click(screen.getByRole("link", { name: "Playground" }));

    // Switch to JSON Body and enter invalid JSON
    fireEvent.click(screen.getByRole("tab", { name: "JSON Body" }));
    const rawEditor = screen.getByLabelText("Request JSON body");
    fireEvent.change(rawEditor, { target: { value: "{bad json" } });
    expect(screen.getByRole("alert")).toHaveTextContent(/JSON syntax error/);
    expect(screen.getByRole("button", { name: "Send Request" })).toBeDisabled();

    // Switch to GET /health and send
    fireEvent.change(screen.getByLabelText("ENDPOINT"), {
      target: { value: "health" },
    });
    expect(screen.getByText(/takes no request body/)).toBeVisible();

    const sendBtn = screen.getByRole("button", { name: "Send Request" });
    expect(sendBtn).not.toBeDisabled();
    fireEvent.click(sendBtn);

    await waitFor(() => {
      expect(screen.getByText("401 Unauthorized")).toBeVisible();
    });
    expect(screen.getByText(/invalid api key/)).toBeVisible();

    // Second click triggers network failure
    fireEvent.click(screen.getByRole("button", { name: "Send Request" }));
    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Connection refused");
    });
  });

  it("syncs navigation and endpoint selection with browser URL and popstate", () => {
    window.history.replaceState(null, "", "/playground?endpoint=decisions");
    render(
      <Dashboard
        state={{
          history: [snapshot({ model: "test/model" })],
          connection: "live",
          error: null,
        }}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "API Playground" }),
    ).toBeVisible();
    expect(document.title).toBe("warp34 · Playground");
    expect((screen.getByLabelText("ENDPOINT") as HTMLSelectElement).value).toBe(
      "decisions",
    );

    fireEvent.change(screen.getByLabelText("ENDPOINT"), {
      target: { value: "responses" },
    });
    expect(window.location.pathname + window.location.search).toBe(
      "/playground?endpoint=responses",
    );

    fireEvent.click(screen.getByRole("link", { name: "Connect" }));
    expect(window.location.pathname).toBe("/connect");
    expect(document.title).toBe("warp34 · Connect");

    window.history.replaceState(null, "", "/");
    fireEvent(window, new PopStateEvent("popstate"));
    expect(screen.getByRole("heading", { name: "Overview" })).toBeVisible();
    expect(document.title).toBe("warp34 · Overview");
  });

  it("uploads an image file onto a user turn, renders thumbnail preview, and removes it", async () => {
    render(
      <Dashboard
        state={{
          history: [snapshot({ model: "nchapman/Qwen3.5-VL" })],
          connection: "live",
          error: null,
        }}
      />,
    );

    fireEvent.click(screen.getByRole("link", { name: "Playground" }));

    const fileInput = screen.getByLabelText(
      "Attach image to turn 2",
    ) as HTMLInputElement;
    const pngBytes = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
    const file = new File([pngBytes], "sample.png", { type: "image/png" });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByAltText("Turn 2 attachment 1")).toBeVisible();
    });

    fireEvent.click(
      screen.getByRole("button", { name: "Remove image 1 from turn 2" }),
    );
    expect(screen.queryByAltText("Turn 2 attachment 1")).toBeNull();
  });

  it("supports /v1/images/generations structured controls, renders generated PNGs, and forwards to /v1/images/edits", async () => {
    const fakeB64 =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          created: 1700000000,
          model: "Qwen/Qwen-Image-2.1-Turbo",
          data: [
            {
              b64_json: fakeB64,
              width: 512,
              height: 512,
              channels: 4,
              steps: 8,
              seed: 42,
              timings: {
                condition_encode_ms: 46.0,
                vae_encode_ms: 0.0,
                denoise_step0_ms: 219.0,
                denoise_cached_ms: 1366.0,
                vae_decode_ms: 152.0,
                total_ms: 1783.0,
              },
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    window.history.replaceState(null, "", "/playground");
    render(
      <Dashboard
        state={{
          history: [snapshot({ model: "Qwen/Qwen-Image-2.1-Turbo" })],
          connection: "live",
          error: null,
        }}
      />,
    );

    expect((screen.getByLabelText("ENDPOINT") as HTMLSelectElement).value).toBe(
      "images-generations",
    );

    fireEvent.change(screen.getByLabelText("Image size"), {
      target: { value: "512x512" },
    });
    fireEvent.click(screen.getByLabelText("Transparent RGBA"));
    fireEvent.change(screen.getByLabelText("Image prompt"), {
      target: { value: "Translucent crystal lotus on dark water" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Send Request" }));

    await waitFor(() => {
      expect(screen.getByAltText("Generated output 1")).toBeVisible();
    });
    expect(screen.getByText(/512x512x4 \(RGBA\) · steps=8/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Download PNG" })).toHaveAttribute(
      "href",
      `data:image/png;base64,${fakeB64}`,
    );

    fireEvent.click(screen.getByRole("button", { name: "↪ Edit Image" }));
    expect((screen.getByLabelText("ENDPOINT") as HTMLSelectElement).value).toBe(
      "images-edits",
    );
    expect(screen.getByAltText("Condition image 1")).toHaveAttribute(
      "src",
      `data:image/png;base64,${fakeB64}`,
    );

    // Inject explicit width/height + cfg_scale alias in JSON tab, then verify structured controls normalize them
    fireEvent.click(screen.getByRole("tab", { name: "JSON Body" }));
    const jsonEditor = screen.getByLabelText(
      "Request JSON body",
    ) as HTMLTextAreaElement;
    const rawWithAliases = JSON.parse(jsonEditor.value) as Record<
      string,
      unknown
    >;
    delete rawWithAliases.size;
    delete rawWithAliases.guidance_scale;
    rawWithAliases.width = 768;
    rawWithAliases.height = 768;
    rawWithAliases.cfg_scale = 3.5;
    fireEvent.change(jsonEditor, {
      target: { value: JSON.stringify(rawWithAliases, null, 2) },
    });

    fireEvent.click(screen.getByRole("tab", { name: "Prompt & Params" }));
    expect(
      (screen.getByLabelText("Image size") as HTMLSelectElement).value,
    ).toBe("768x768");
    expect(
      Number(
        (screen.getByLabelText("Guidance scale") as HTMLInputElement).value,
      ),
    ).toBe(3.5);

    fireEvent.change(screen.getByLabelText("Image size"), {
      target: { value: "auto" },
    });
    fireEvent.change(screen.getByLabelText("Guidance scale"), {
      target: { value: "4" },
    });
    fireEvent.click(screen.getByRole("tab", { name: "JSON Body" }));
    const normalized = JSON.parse(
      (screen.getByLabelText("Request JSON body") as HTMLTextAreaElement).value,
    ) as Record<string, unknown>;
    expect(normalized.size).toBe("auto");
    expect(normalized.width).toBeUndefined();
    expect(normalized.height).toBeUndefined();
    expect(normalized.guidance_scale).toBe(4);
    expect(normalized.cfg_scale).toBeUndefined();

    // Verify only Image + System endpoints appear in the selector when serving an Image model
    const epSelect = screen.getByLabelText("ENDPOINT") as HTMLSelectElement;
    const optionValues = Array.from(epSelect.options).map((o) => o.value);
    expect(optionValues).toContain("images-generations");
    expect(optionValues).toContain("images-edits");
    expect(optionValues).toContain("models");
    expect(optionValues).not.toContain("chat-completions");
    expect(optionValues).not.toContain("decisions");
    expect(optionValues).not.toContain("tokenize");
  });

  it("hides /v1/images/* endpoints when serving an LLM model", () => {
    window.history.replaceState(
      null,
      "",
      "/playground?endpoint=images-generations",
    );
    const llmSnap = snapshot({
      model: "nchapman/Qwen3.8-27B-W4A16-HGPTQ",
    });
    render(
      <Dashboard
        state={{
          history: [llmSnap],
          connection: "live",
          error: null,
        }}
      />,
    );
    const epSelect = screen.getByLabelText("ENDPOINT") as HTMLSelectElement;
    // Falls back to chat-completions because images-generations is not enabled on an LLM server
    expect(epSelect.value).toBe("chat-completions");
    const optionValues = Array.from(epSelect.options).map((o) => o.value);
    expect(optionValues).toContain("chat-completions");
    expect(optionValues).not.toContain("images-generations");
    expect(optionValues).not.toContain("images-edits");
  });
});
