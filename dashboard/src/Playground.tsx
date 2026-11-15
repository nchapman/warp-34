import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
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
  type EndpointSpec,
  type ParsedApiResult,
  type SseFrame,
  type TurnMessage,
} from "./playground";

interface PlaygroundProps {
  model: string | null;
  decisionMode?: boolean;
}

type RequestTab = "structured" | "json";
type ResponseTab = "parsed" | "raw" | "sse" | "curl";

interface CallMetrics {
  status: number | null;
  statusText: string;
  ttftMs: number | null;
  e2eMs: number | null;
}

const DEFAULT_MODEL = "default";
const MAX_SSE_INSPECTOR_FRAMES = 512;
const MAX_SSE_BUFFER_CHARS = 1_048_576;
const REQUEST_IDLE_TIMEOUT_MS = 120_000;
const MAX_IMAGES_PER_REQUEST = 4;
const MAX_IMAGE_FILE_BYTES = 16 * 1024 * 1024;
const MAX_CANVAS_DIM = 2048;

function resolveRoute(
  pathEndpointId: string | undefined,
  searchParams: URLSearchParams,
  decisionMode: boolean,
): {
  endpoint: EndpointSpec;
  presetId: string;
} {
  const requestedEndpoint = searchParams.get("endpoint") ?? pathEndpointId;
  const fallbackId = decisionMode ? "decisions" : "chat-completions";
  const endpoint =
    ENDPOINTS.find((e) => e.id === requestedEndpoint) ??
    ENDPOINTS.find((e) => e.id === fallbackId) ??
    ENDPOINTS[0];
  const requestedPreset = searchParams.get("preset");
  const preset =
    endpoint.presets.find((p) => p.id === requestedPreset) ??
    endpoint.presets[0];
  return { endpoint, presetId: preset.id };
}

function readFileAsEngineDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    if (file.size > MAX_IMAGE_FILE_BYTES) {
      reject(new Error(`Image '${file.name}' exceeds 16 MiB limit.`));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      if (
        file.type === "image/png" ||
        file.type === "image/jpeg" ||
        result.startsWith("data:image/png;") ||
        result.startsWith("data:image/jpeg;")
      ) {
        resolve(result);
        return;
      }
      // Transcode WebP/GIF/BMP to PNG data URL so warp-34's native Rust PNG/JPEG decoder accepts it
      const img = new Image();
      img.onload = () => {
        const rawW = img.naturalWidth || img.width || 64;
        const rawH = img.naturalHeight || img.height || 64;
        const scale = Math.min(1, MAX_CANVAS_DIM / Math.max(rawW, rawH, 1));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(rawW * scale));
        canvas.height = Math.max(1, Math.round(rawH * scale));
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(result);
          return;
        }
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/png"));
      };
      img.onerror = () =>
        reject(new Error(`Failed to decode image '${file.name}'.`));
      img.src = result;
    };
    reader.readAsDataURL(file);
  });
}

export function Playground({ model, decisionMode = false }: PlaygroundProps) {
  const navigate = useNavigate();
  const { endpointId: pathEndpointId } = useParams<{ endpointId?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const defaultEndpointId = decisionMode ? "decisions" : "chat-completions";
  const initial = resolveRoute(pathEndpointId, searchParams, decisionMode);
  const [endpointId, setEndpointId] = useState<string>(initial.endpoint.id);
  const endpoint = ENDPOINTS.find((e) => e.id === endpointId) ?? ENDPOINTS[0];

  const [presetId, setPresetId] = useState<string>(initial.presetId);
  const activeModel = model ?? DEFAULT_MODEL;

  function updateRouteParams(nextEndpointId: string, nextPresetId: string) {
    const spec = ENDPOINTS.find((e) => e.id === nextEndpointId) ?? ENDPOINTS[0];
    const nextParams = new URLSearchParams();
    if (nextEndpointId !== defaultEndpointId) {
      nextParams.set("endpoint", nextEndpointId);
    }
    if (nextPresetId !== spec.presets[0].id) {
      nextParams.set("preset", nextPresetId);
    }
    if (pathEndpointId) {
      const qs = nextParams.toString();
      navigate(qs ? `/playground?${qs}` : "/playground", { replace: true });
    } else {
      setSearchParams(nextParams, { replace: true });
    }
  }

  const [rawJson, setRawJson] = useState<string>(() => {
    const preset =
      initial.endpoint.presets.find((p) => p.id === initial.presetId) ??
      initial.endpoint.presets[0];
    return initial.endpoint.method === "GET"
      ? ""
      : JSON.stringify(preset.buildBody(activeModel), null, 2);
  });
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string>("");
  const [requestTab, setRequestTab] = useState<RequestTab>(() =>
    initial.endpoint.conversational ? "structured" : "json",
  );
  const [responseTab, setResponseTab] = useState<ResponseTab>("parsed");

  const [inFlight, setInFlight] = useState<boolean>(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [callMetrics, setCallMetrics] = useState<CallMetrics>({
    status: null,
    statusText: "",
    ttftMs: null,
    e2eMs: null,
  });
  const [parsedResult, setParsedResult] =
    useState<ParsedApiResult>(emptyParsedResult);
  const [rawResponse, setRawResponse] = useState<string>("");
  const [sseFrames, setSseFrames] = useState<SseFrame[]>([]);
  const [followUpText, setFollowUpText] = useState<string>("");
  const [copiedCurl, setCopiedCurl] = useState<boolean>(false);

  const abortRef = useRef<AbortController | null>(null);
  const prevModelRef = useRef<string>(activeModel);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  // Keep the request JSON's `model` field synced when the first live metrics snapshot arrives
  useEffect(() => {
    if (prevModelRef.current === activeModel) return;
    const prev = prevModelRef.current;
    prevModelRef.current = activeModel;
    try {
      const parsed = JSON.parse(rawJson) as Record<string, unknown>;
      if (parsed.model === prev || parsed.model === DEFAULT_MODEL) {
        parsed.model = activeModel;
        setRawJson(JSON.stringify(parsed, null, 2));
      }
    } catch {
      // Leave user's unparseable JSON untouched
    }
  }, [activeModel, rawJson]);

  // Synchronize endpoint & preset state when URL path or search params change externally
  useEffect(() => {
    const resolved = resolveRoute(pathEndpointId, searchParams, decisionMode);
    if (resolved.endpoint.id !== endpointId || resolved.presetId !== presetId) {
      setEndpointId(resolved.endpoint.id);
      setPresetId(resolved.presetId);
      const nextPreset =
        resolved.endpoint.presets.find((p) => p.id === resolved.presetId) ??
        resolved.endpoint.presets[0];
      setRawJson(
        resolved.endpoint.method === "GET"
          ? ""
          : JSON.stringify(nextPreset.buildBody(activeModel), null, 2),
      );
      setJsonError(null);
      setRequestTab(resolved.endpoint.conversational ? "structured" : "json");
    }
  }, [pathEndpointId, searchParams, decisionMode]);

  function selectEndpoint(nextId: string) {
    const nextSpec = ENDPOINTS.find((e) => e.id === nextId) ?? ENDPOINTS[0];
    const firstPreset = nextSpec.presets[0];
    setEndpointId(nextSpec.id);
    setPresetId(firstPreset.id);
    updateRouteParams(nextSpec.id, firstPreset.id);
    const body = firstPreset.buildBody(activeModel);
    setRawJson(nextSpec.method === "GET" ? "" : JSON.stringify(body, null, 2));
    setJsonError(null);
    setRequestTab(nextSpec.conversational ? "structured" : "json");
    clearResponseState();
  }

  function selectPreset(nextPresetId: string) {
    const preset =
      endpoint.presets.find((p) => p.id === nextPresetId) ??
      endpoint.presets[0];
    setPresetId(preset.id);
    updateRouteParams(endpoint.id, preset.id);
    const body = preset.buildBody(activeModel);
    setRawJson(endpoint.method === "GET" ? "" : JSON.stringify(body, null, 2));
    setJsonError(null);
    clearResponseState();
  }

  function clearResponseState() {
    setRequestError(null);
    setCallMetrics({ status: null, statusText: "", ttftMs: null, e2eMs: null });
    setParsedResult(emptyParsedResult());
    setRawResponse("");
    setSseFrames([]);
  }

  function parseCurrentBody(): Record<string, unknown> | null {
    if (endpoint.method === "GET") return {};
    try {
      const val = JSON.parse(rawJson);
      if (!val || typeof val !== "object" || Array.isArray(val)) {
        return null;
      }
      return val as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  function updateBodyObject(
    updater: (body: Record<string, unknown>) => Record<string, unknown>,
  ) {
    const current = parseCurrentBody();
    if (!current) return;
    const updated = updater(current);
    setRawJson(JSON.stringify(updated, null, 2));
    setJsonError(null);
  }

  const bodyObj = parseCurrentBody();
  const messages: TurnMessage[] =
    endpoint.conversational && bodyObj
      ? extractMessagesFromBody(endpoint.kind, bodyObj)
      : [];

  const totalImages = messages.reduce(
    (sum, m) => sum + (m.images?.length ?? 0),
    0,
  );

  function handleMessageChange(index: number, patch: Partial<TurnMessage>) {
    updateBodyObject((curr) => {
      const list = extractMessagesFromBody(endpoint.kind, curr).map((m, i) =>
        i === index ? { ...m, ...patch } : m,
      );
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  async function handleAttachImages(index: number, files: FileList | File[]) {
    const fileArr = Array.from(files).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (fileArr.length === 0) return;
    const remainingBudget = Math.max(0, MAX_IMAGES_PER_REQUEST - totalImages);
    const toRead = fileArr.slice(0, remainingBudget);
    if (toRead.length === 0) return;
    try {
      const dataUrls = await Promise.all(toRead.map(readFileAsEngineDataUrl));
      setRequestError(null);
      updateBodyObject((curr) => {
        const list = extractMessagesFromBody(endpoint.kind, curr).map((m, i) =>
          i === index
            ? { ...m, images: [...(m.images ?? []), ...dataUrls] }
            : m,
        );
        return applyMessagesToBody(endpoint.kind, curr, list);
      });
    } catch (err) {
      setRequestError(
        err instanceof Error ? err.message : "Failed to attach image.",
      );
    }
  }

  function handleRemoveImage(turnIdx: number, imgIdx: number) {
    updateBodyObject((curr) => {
      const list = extractMessagesFromBody(endpoint.kind, curr).map((m, i) => {
        if (i !== turnIdx || !m.images) return m;
        const nextImgs = m.images.filter((_, j) => j !== imgIdx);
        return { ...m, images: nextImgs.length > 0 ? nextImgs : undefined };
      });
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  function handleAddMessage(role: TurnMessage["role"] = "user") {
    updateBodyObject((curr) => {
      const list = [
        ...extractMessagesFromBody(endpoint.kind, curr),
        { role, content: "" },
      ];
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  function handleRemoveMessage(index: number) {
    updateBodyObject((curr) => {
      const list = extractMessagesFromBody(endpoint.kind, curr).filter(
        (_, i) => i !== index,
      );
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  function handleRawJsonEdit(value: string) {
    setRawJson(value);
    if (endpoint.method === "GET") {
      setJsonError(null);
      return;
    }
    try {
      const parsed = JSON.parse(value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setJsonError("Request body must be a JSON object");
      } else {
        setJsonError(null);
      }
    } catch (err) {
      setJsonError((err as Error).message);
    }
  }

  async function executeRequest(overrideJson?: string) {
    const payloadText = overrideJson ?? rawJson;
    let reqObj: Record<string, unknown> | null = null;
    if (endpoint.method !== "GET") {
      try {
        reqObj = JSON.parse(payloadText) as Record<string, unknown>;
      } catch (err) {
        setJsonError((err as Error).message);
        return;
      }
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    let timedOut = false;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    const resetIdleTimer = () => {
      if (idleTimer !== null) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, REQUEST_IDLE_TIMEOUT_MS);
    };
    const clearIdleTimer = () => {
      if (idleTimer !== null) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
    };

    setInFlight(true);
    setRequestError(null);
    setCallMetrics({ status: null, statusText: "", ttftMs: null, e2eMs: null });
    setParsedResult(emptyParsedResult());
    setRawResponse("");
    setSseFrames([]);

    const headers: Record<string, string> = {};
    if (endpoint.method !== "GET") {
      headers["Content-Type"] = "application/json";
    }
    const needsStatefulScope =
      endpoint.kind === "openai-responses" &&
      reqObj !== null &&
      (reqObj.store === true ||
        typeof reqObj.previous_response_id === "string");
    const effectiveKey =
      apiKey.trim().length > 0
        ? apiKey.trim()
        : needsStatefulScope
          ? "playground-local"
          : "";
    if (effectiveKey.length > 0) {
      if (endpoint.kind === "anthropic-messages") {
        headers["x-api-key"] = effectiveKey;
      } else {
        headers["Authorization"] = `Bearer ${effectiveKey}`;
      }
    }

    const startTime = performance.now();
    let firstByteMs: number | null = null;

    try {
      resetIdleTimer();
      const response = await fetch(endpoint.path, {
        method: endpoint.method,
        headers,
        body: endpoint.method === "GET" ? undefined : payloadText,
        signal: controller.signal,
      });
      resetIdleTimer();

      const contentType = response.headers.get("content-type") ?? "";
      const isEventStream = contentType.includes("text/event-stream");

      if (isEventStream && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let frameIndex = 0;
        let runningParsed = emptyParsedResult();
        const retainedFrames: SseFrame[] = [];

        while (true) {
          resetIdleTimer();
          const { value, done } = await reader.read();
          if (done) break;
          const nowMs = Math.round(performance.now() - startTime);
          if (firstByteMs === null) {
            firstByteMs = nowMs;
            setCallMetrics({
              status: response.status,
              statusText: response.statusText || "OK",
              ttftMs: firstByteMs,
              e2eMs: null,
            });
          }
          buffer += decoder.decode(value, { stream: true });
          if (buffer.length > MAX_SSE_BUFFER_CHARS) {
            throw new Error("SSE frame buffer exceeded 1 MiB limit.");
          }
          const drained = drainSseBuffer(buffer, frameIndex, nowMs);
          buffer = drained.remainder;
          if (drained.frames.length > 0) {
            frameIndex += drained.frames.length;
            for (const f of drained.frames) {
              runningParsed = applySseFrameToResult(
                endpoint.kind,
                runningParsed,
                f,
              );
              if (retainedFrames.length < MAX_SSE_INSPECTOR_FRAMES) {
                retainedFrames.push(f);
              }
            }
            setSseFrames(retainedFrames.slice());
            setParsedResult(runningParsed);
            if (runningParsed.streamError) {
              setRequestError(runningParsed.streamError);
            }
            const lastData = drained.frames.at(-1);
            if (lastData && lastData.data !== "[DONE]") {
              setRawResponse(lastData.data);
            }
          }
        }

        const totalMs = Math.round(performance.now() - startTime);
        setCallMetrics({
          status: response.status,
          statusText: response.statusText || "OK",
          ttftMs: firstByteMs ?? totalMs,
          e2eMs: totalMs,
        });
      } else {
        const rawText = await response.text();
        const text =
          rawText.length > MAX_SSE_BUFFER_CHARS
            ? `${rawText.slice(0, MAX_SSE_BUFFER_CHARS)}\n… [response truncated at 1 MiB]`
            : rawText;
        const totalMs = Math.round(performance.now() - startTime);
        setCallMetrics({
          status: response.status,
          statusText: response.statusText || (response.ok ? "OK" : "Error"),
          ttftMs: totalMs,
          e2eMs: totalMs,
        });
        try {
          const jsonVal = JSON.parse(text);
          setRawResponse(JSON.stringify(jsonVal, null, 2));
          if (response.ok) {
            setParsedResult(parseJsonResponse(endpoint.kind, jsonVal));
          } else {
            setParsedResult({
              ...emptyParsedResult(),
              content: JSON.stringify(jsonVal, null, 2),
            });
          }
        } catch {
          setRawResponse(text);
          setParsedResult({
            ...emptyParsedResult(),
            content: text,
          });
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setRequestError(
          timedOut
            ? "Request timed out waiting for server activity."
            : "Request aborted by user.",
        );
      } else {
        setRequestError((err as Error).message || "Network request failed.");
      }
    } finally {
      clearIdleTimer();
      setInFlight(false);
    }
  }

  function handleAppendAndContinue(sendImmediately: boolean) {
    if (!bodyObj) return;
    const nextBody = appendAssistantTurnToBody(
      endpoint.kind,
      bodyObj,
      parsedResult,
      followUpText,
    );
    const nextJson = JSON.stringify(nextBody, null, 2);
    setRawJson(nextJson);
    setJsonError(null);
    setFollowUpText("");
    if (
      sendImmediately &&
      (followUpText.trim().length > 0 || parsedResult.toolCalls.length > 0)
    ) {
      void executeRequest(nextJson);
    }
  }

  const curlCommand = formatCurlCommand(
    window.location.origin,
    endpoint,
    rawJson,
    apiKey,
  );

  async function handleCopyCurl() {
    try {
      await navigator.clipboard?.writeText(curlCommand);
      setCopiedCurl(true);
      setTimeout(() => setCopiedCurl(false), 1500);
    } catch {
      setResponseTab("curl");
    }
  }

  const decodeTokPerSec =
    parsedResult.usage.outputTokens &&
    parsedResult.usage.outputTokens > 1 &&
    callMetrics.e2eMs !== null &&
    callMetrics.ttftMs !== null &&
    callMetrics.e2eMs > callMetrics.ttftMs
      ? (
          (parsedResult.usage.outputTokens * 1000) /
          (callMetrics.e2eMs - callMetrics.ttftMs)
        ).toFixed(1)
      : null;

  const hasAssistantReply =
    endpoint.conversational &&
    (parsedResult.content.length > 0 || parsedResult.toolCalls.length > 0);

  return (
    <div className="playground">
      <div className="section-heading">
        <div>
          <h1>API Playground</h1>
          <p>
            Inspect wire payloads, SSE frames, prefix cache hits, and multi-turn
            state against this engine.
          </p>
        </div>
        <div className="playground-auth">
          <label htmlFor="pg-api-key" className="eyebrow">
            API KEY
          </label>
          <input
            id="pg-api-key"
            type="password"
            placeholder="Optional Bearer / x-api-key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </div>
      </div>

      <div className="playground-workspace">
        <aside className="panel pg-sidebar" aria-label="API endpoints">
          {(
            [
              "Generative",
              "Decision & Scoring",
              "Tokenizer & Counts",
              "System",
            ] as const
          ).map((group) => (
            <div key={group} className="pg-sidebar-group">
              <span className="eyebrow">{group.toUpperCase()}</span>
              <div className="pg-sidebar-items">
                {ENDPOINTS.filter((ep) => ep.group === group).map((ep) => (
                  <button
                    key={ep.id}
                    type="button"
                    className={`pg-sidebar-item ${
                      endpoint.id === ep.id ? "active" : ""
                    }`}
                    onClick={() => selectEndpoint(ep.id)}
                  >
                    <span
                      className={`method-badge method-${ep.method.toLowerCase()}`}
                    >
                      {ep.method}
                    </span>
                    <span className="pg-sidebar-path mono">{ep.path}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </aside>

        <div className="pg-main">
          <section
            className="panel playground-toolbar"
            aria-label="Endpoint controls"
          >
            <div className="pg-field pg-endpoint-mobile">
              <label htmlFor="pg-endpoint" className="eyebrow">
                ENDPOINT
              </label>
              <select
                id="pg-endpoint"
                value={endpoint.id}
                onChange={(e) => selectEndpoint(e.target.value)}
              >
                {(
                  [
                    "Generative",
                    "Decision & Scoring",
                    "Tokenizer & Counts",
                    "System",
                  ] as const
                ).map((group) => (
                  <optgroup key={group} label={group}>
                    {ENDPOINTS.filter((ep) => ep.group === group).map((ep) => (
                      <option key={ep.id} value={ep.id}>
                        {ep.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="pg-field">
              <label htmlFor="pg-preset" className="eyebrow">
                PRESET ·{" "}
                {endpoint.presets.find((p) => p.id === presetId)?.description}
              </label>
              <select
                id="pg-preset"
                value={presetId}
                onChange={(e) => selectPreset(e.target.value)}
              >
                {endpoint.presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="pg-actions">
              <button
                type="button"
                className="pg-btn pg-btn-secondary"
                onClick={() => void handleCopyCurl()}
              >
                {copiedCurl ? "Copied cURL" : "Copy cURL"}
              </button>
              {inFlight ? (
                <button
                  type="button"
                  className="pg-btn pg-btn-danger"
                  onClick={() => abortRef.current?.abort()}
                >
                  Abort
                </button>
              ) : (
                <button
                  type="button"
                  className="pg-btn pg-btn-primary"
                  disabled={Boolean(jsonError)}
                  onClick={() => void executeRequest()}
                >
                  Send Request
                </button>
              )}
            </div>
          </section>

          <div className="playground-grid">
            <section
              className="panel playground-pane"
              aria-labelledby="req-pane-title"
            >
              <div className="panel-heading pg-pane-heading">
                <div className="pg-title-group">
                  <span
                    className={`method-badge method-${endpoint.method.toLowerCase()}`}
                  >
                    {endpoint.method}
                  </span>
                  <h2 id="req-pane-title" className="mono">
                    {endpoint.path}
                  </h2>
                </div>
                {endpoint.conversational && (
                  <div
                    className="pg-tab-group"
                    role="tablist"
                    aria-label="Request editor mode"
                  >
                    <button
                      type="button"
                      role="tab"
                      aria-selected={requestTab === "structured"}
                      className={requestTab === "structured" ? "active" : ""}
                      onClick={() => setRequestTab("structured")}
                    >
                      Messages & Params
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={requestTab === "json"}
                      className={requestTab === "json" ? "active" : ""}
                      onClick={() => setRequestTab("json")}
                    >
                      JSON Body
                    </button>
                  </div>
                )}
              </div>

              {endpoint.conversational &&
              requestTab === "structured" &&
              bodyObj ? (
                <div className="pg-structured">
                  <ParameterBar
                    endpoint={endpoint}
                    body={bodyObj}
                    onUpdate={updateBodyObject}
                  />

                  <div className="pg-messages-header">
                    <span className="eyebrow">
                      {endpoint.kind === "openai-responses"
                        ? "INSTRUCTIONS & INPUT TURNS"
                        : "MESSAGES ARRAY"}
                    </span>
                    <div className="pg-message-add-btns">
                      <button
                        type="button"
                        className="pg-mini-btn"
                        onClick={() => handleAddMessage("user")}
                      >
                        + User turn
                      </button>
                      <button
                        type="button"
                        className="pg-mini-btn"
                        onClick={() => handleAddMessage("assistant")}
                      >
                        + Assistant turn
                      </button>
                    </div>
                  </div>

                  <div className="pg-messages-list">
                    {messages.map((msg, idx) => (
                      <div
                        key={idx}
                        className="pg-message-row"
                        onDragOver={(e) => {
                          if (msg.role === "user") {
                            e.preventDefault();
                          }
                        }}
                        onDrop={(e) => {
                          if (
                            msg.role === "user" &&
                            e.dataTransfer?.files?.length
                          ) {
                            e.preventDefault();
                            void handleAttachImages(idx, e.dataTransfer.files);
                          }
                        }}
                      >
                        <div className="pg-message-meta">
                          <select
                            aria-label={`Turn ${idx + 1} role`}
                            value={msg.role}
                            onChange={(e) =>
                              handleMessageChange(idx, {
                                role: e.target.value as TurnMessage["role"],
                              })
                            }
                          >
                            <option value="system">system</option>
                            <option value="user">user</option>
                            <option value="assistant">assistant</option>
                            {endpoint.kind === "openai-chat" && (
                              <option value="tool">tool</option>
                            )}
                          </select>
                          <span className="pg-turn-index mono">#{idx}</span>
                          {msg.tool_calls && msg.tool_calls.length > 0 && (
                            <span className="pg-badge amber">
                              {msg.tool_calls.length} tool_call
                              {msg.tool_calls.length > 1 ? "s" : ""}
                            </span>
                          )}
                          <div className="pg-turn-actions">
                            {msg.role === "user" &&
                              totalImages < MAX_IMAGES_PER_REQUEST && (
                                <label className="pg-mini-btn pg-upload-label">
                                  + Image
                                  <input
                                    type="file"
                                    accept="image/png,image/jpeg,image/webp,image/gif"
                                    multiple
                                    aria-label={`Attach image to turn ${idx + 1}`}
                                    className="pg-upload-input"
                                    onChange={(e) => {
                                      if (
                                        e.target.files &&
                                        e.target.files.length > 0
                                      ) {
                                        void handleAttachImages(
                                          idx,
                                          e.target.files,
                                        );
                                        e.target.value = "";
                                      }
                                    }}
                                  />
                                </label>
                              )}
                            <button
                              type="button"
                              className="pg-remove-btn"
                              aria-label={`Remove turn ${idx + 1}`}
                              onClick={() => handleRemoveMessage(idx)}
                            >
                              ×
                            </button>
                          </div>
                        </div>

                        {msg.images && msg.images.length > 0 && (
                          <div className="pg-turn-images">
                            {msg.images.map((imgUrl, imgIdx) => (
                              <div key={imgIdx} className="pg-turn-thumb">
                                <img
                                  src={imgUrl}
                                  alt={`Turn ${idx + 1} attachment ${imgIdx + 1}`}
                                  className="pg-turn-thumb-img"
                                />
                                <span className="pg-turn-thumb-meta mono">
                                  {imgUrl.startsWith("data:image/jpeg")
                                    ? "JPEG"
                                    : imgUrl.startsWith("data:image/png")
                                      ? "PNG"
                                      : "IMG"}{" "}
                                  #{imgIdx + 1}
                                </span>
                                <button
                                  type="button"
                                  className="pg-remove-btn pg-thumb-remove"
                                  aria-label={`Remove image ${imgIdx + 1} from turn ${idx + 1}`}
                                  onClick={() => handleRemoveImage(idx, imgIdx)}
                                >
                                  ×
                                </button>
                              </div>
                            ))}
                          </div>
                        )}

                        <textarea
                          aria-label={`Turn ${idx + 1} content`}
                          rows={Math.min(
                            8,
                            Math.max(2, msg.content.split("\n").length),
                          )}
                          value={msg.content}
                          placeholder={
                            msg.role === "user"
                              ? "user content (or drop / paste an image)…"
                              : `${msg.role} content…`
                          }
                          onChange={(e) =>
                            handleMessageChange(idx, {
                              content: e.target.value,
                            })
                          }
                          onPaste={(e) => {
                            if (
                              msg.role === "user" &&
                              e.clipboardData?.files?.length
                            ) {
                              const imgFiles = Array.from(
                                e.clipboardData.files,
                              ).filter((f) => f.type.startsWith("image/"));
                              if (imgFiles.length > 0) {
                                e.preventDefault();
                                void handleAttachImages(idx, imgFiles);
                              }
                            }
                          }}
                          onKeyDown={(e) => {
                            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                              e.preventDefault();
                              void executeRequest();
                            }
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ) : endpoint.method === "GET" ? (
                <div className="pg-empty-body">
                  <code>GET {endpoint.path}</code> takes no request body. Click{" "}
                  <b>Send Request</b> to inspect the response.
                </div>
              ) : (
                <div className="pg-json-editor">
                  <textarea
                    aria-label="Request JSON body"
                    className="mono"
                    rows={18}
                    value={rawJson}
                    onChange={(e) => handleRawJsonEdit(e.target.value)}
                    onKeyDown={(e) => {
                      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                        e.preventDefault();
                        void executeRequest();
                      }
                    }}
                  />
                  {jsonError && (
                    <div className="pg-json-error" role="alert">
                      JSON syntax error: {jsonError}
                    </div>
                  )}
                </div>
              )}
            </section>

            <section
              className="panel playground-pane"
              aria-labelledby="resp-pane-title"
            >
              <div className="panel-heading pg-pane-heading">
                <div className="pg-title-group">
                  <h2 id="resp-pane-title">Wire Response</h2>
                  {inFlight && (
                    <span className="pg-badge amber">Streaming…</span>
                  )}
                  {callMetrics.status !== null && (
                    <span
                      className={`pg-badge ${
                        callMetrics.status >= 200 && callMetrics.status < 300
                          ? "sage"
                          : "danger"
                      }`}
                    >
                      {callMetrics.status} {callMetrics.statusText}
                    </span>
                  )}
                </div>
                <div
                  className="pg-tab-group"
                  role="tablist"
                  aria-label="Response inspector view"
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={responseTab === "parsed"}
                    className={responseTab === "parsed" ? "active" : ""}
                    onClick={() => setResponseTab("parsed")}
                  >
                    Parsed
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={responseTab === "raw"}
                    className={responseTab === "raw" ? "active" : ""}
                    onClick={() => setResponseTab("raw")}
                  >
                    Response JSON
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={responseTab === "sse"}
                    className={responseTab === "sse" ? "active" : ""}
                    onClick={() => setResponseTab("sse")}
                  >
                    SSE ({sseFrames.length})
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={responseTab === "curl"}
                    className={responseTab === "curl" ? "active" : ""}
                    onClick={() => setResponseTab("curl")}
                  >
                    cURL
                  </button>
                </div>
              </div>

              <div className="pg-metrics-strip" aria-label="Response telemetry">
                <div>
                  <span className="eyebrow">CLIENT TTFT</span>
                  <strong className="numeric">
                    {callMetrics.ttftMs !== null
                      ? `${callMetrics.ttftMs} ms`
                      : "—"}
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">TOTAL E2E</span>
                  <strong className="numeric">
                    {callMetrics.e2eMs !== null
                      ? `${callMetrics.e2eMs} ms`
                      : "—"}
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">STREAM PACE</span>
                  <strong className="numeric">
                    {decodeTokPerSec ? `${decodeTokPerSec} tok/s` : "—"}
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">PROMPT / CACHED</span>
                  <strong className="numeric">
                    {parsedResult.usage.inputTokens ?? "—"} /{" "}
                    <span
                      className={
                        (parsedResult.usage.cachedTokens ?? 0) > 0
                          ? "healthy"
                          : ""
                      }
                    >
                      {parsedResult.usage.cachedTokens ?? "—"}
                    </span>
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">OUTPUT TOKENS</span>
                  <strong className="numeric">
                    {parsedResult.usage.outputTokens ?? "—"}
                  </strong>
                </div>
              </div>

              {requestError && (
                <div className="notice" role="alert">
                  <strong>Request failed:</strong> {requestError}
                </div>
              )}

              {responseTab === "parsed" && (
                <div className="pg-parsed-view">
                  {parsedResult.reasoning && (
                    <details className="pg-block pg-reasoning-block" open>
                      <summary className="eyebrow">
                        REASONING / THINKING STREAM
                        {parsedResult.usage.reasoningTokens !== null &&
                          ` (${parsedResult.usage.reasoningTokens} tok)`}
                      </summary>
                      <pre className="mono">{parsedResult.reasoning}</pre>
                    </details>
                  )}

                  {parsedResult.toolCalls.length > 0 && (
                    <div className="pg-block">
                      <span className="eyebrow">TOOL CALLS</span>
                      {parsedResult.toolCalls.map((tc) => (
                        <div key={tc.id} className="pg-tool-call mono">
                          <div className="pg-tool-header">
                            <strong>{tc.name}</strong>
                            <span className="muted">id: {tc.id}</span>
                          </div>
                          <pre>{tc.arguments}</pre>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="pg-block">
                    <div className="pg-block-header">
                      <span className="eyebrow">
                        {endpoint.conversational
                          ? "ASSISTANT OUTPUT"
                          : "RESPONSE PAYLOAD"}
                      </span>
                      {parsedResult.finishReason && (
                        <span className="mono muted">
                          stop: {parsedResult.finishReason}
                        </span>
                      )}
                    </div>
                    <pre className="pg-output-pre mono">
                      {parsedResult.content ||
                        (inFlight
                          ? "Waiting for tokens…"
                          : "Send a request to inspect the response.")}
                    </pre>
                  </div>

                  {hasAssistantReply && !inFlight && (
                    <div className="pg-continue-box">
                      <div className="pg-continue-header">
                        <span className="eyebrow">
                          CONTINUE MULTI-TURN CONVERSATION
                        </span>
                        <button
                          type="button"
                          className="pg-mini-btn"
                          onClick={() => handleAppendAndContinue(false)}
                        >
                          ↪ Append assistant turn to messages[]
                        </button>
                      </div>
                      <div className="pg-continue-row">
                        <input
                          type="text"
                          aria-label="Follow-up user message"
                          placeholder={
                            parsedResult.toolCalls.length > 0
                              ? 'Tool output JSON (default: {"status":"ok"})…'
                              : "Type next user turn to append & send (tests prefix cache reuse)…"
                          }
                          value={followUpText}
                          onChange={(e) => setFollowUpText(e.target.value)}
                          onKeyDown={(e) => {
                            if (
                              e.key === "Enter" &&
                              (followUpText.trim().length > 0 ||
                                parsedResult.toolCalls.length > 0)
                            ) {
                              e.preventDefault();
                              handleAppendAndContinue(true);
                            }
                          }}
                        />
                        <button
                          type="button"
                          className="pg-btn pg-btn-primary"
                          disabled={
                            followUpText.trim().length === 0 &&
                            parsedResult.toolCalls.length === 0
                          }
                          onClick={() => handleAppendAndContinue(true)}
                        >
                          Append & Send
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {responseTab === "raw" && (
                <pre
                  className="pg-code-box mono"
                  aria-label="Raw response payload"
                >
                  {rawResponse || "No response payload captured yet."}
                </pre>
              )}

              {responseTab === "sse" && (
                <div className="pg-sse-list" aria-label="SSE frames">
                  {sseFrames.length === 0 ? (
                    <p className="muted">
                      No SSE frames recorded. Enable <code>stream: true</code>{" "}
                      and send a request.
                    </p>
                  ) : (
                    sseFrames.map((f) => (
                      <div key={f.index} className="pg-sse-frame mono">
                        <div className="pg-sse-meta">
                          <span>#{f.index}</span>
                          <span>+{f.elapsedMs}ms</span>
                          {f.event && (
                            <span className="amber">event: {f.event}</span>
                          )}
                        </div>
                        <pre>{f.data}</pre>
                      </div>
                    ))
                  )}
                </div>
              )}

              {responseTab === "curl" && (
                <pre className="pg-code-box mono" aria-label="cURL command">
                  {curlCommand}
                </pre>
              )}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

function ParameterBar({
  endpoint,
  body,
  onUpdate,
}: {
  endpoint: EndpointSpec;
  body: Record<string, unknown>;
  onUpdate: (
    updater: (b: Record<string, unknown>) => Record<string, unknown>,
  ) => void;
}) {
  const maxTokensKey =
    endpoint.kind === "openai-responses" ? "max_output_tokens" : "max_tokens";
  const maxTokens =
    typeof body[maxTokensKey] === "number"
      ? (body[maxTokensKey] as number)
      : 256;
  const temperature =
    typeof body.temperature === "number" ? body.temperature : 0;
  const stream = Boolean(body.stream);

  const thinkingEnabled =
    endpoint.kind === "openai-chat"
      ? Boolean(
          (body.chat_template_kwargs as Record<string, unknown> | undefined)
            ?.enable_thinking,
        )
      : endpoint.kind === "anthropic-messages"
        ? (body.thinking as Record<string, unknown> | undefined)?.type ===
          "enabled"
        : false;

  const minMaxTokens =
    endpoint.kind === "anthropic-messages" && thinkingEnabled ? 1025 : 1;

  return (
    <div className="pg-params-bar">
      <label className="pg-param">
        <span className="eyebrow">{maxTokensKey.toUpperCase()}</span>
        <input
          type="number"
          min={minMaxTokens}
          max={4096}
          value={maxTokens}
          onChange={(e) => {
            const val = Math.min(
              4096,
              Math.max(minMaxTokens, Number(e.target.value) || minMaxTokens),
            );
            onUpdate((curr) => ({ ...curr, [maxTokensKey]: val }));
          }}
        />
      </label>

      <label className="pg-param">
        <span className="eyebrow">TEMPERATURE</span>
        <input
          type="number"
          min={0}
          max={2}
          step={0.1}
          value={temperature}
          onChange={(e) => {
            const val = Math.min(2, Math.max(0, Number(e.target.value) || 0));
            onUpdate((curr) => ({ ...curr, temperature: val }));
          }}
        />
      </label>

      {endpoint.supportsStream && (
        <label className="pg-param pg-checkbox">
          <input
            type="checkbox"
            checked={stream}
            onChange={(e) => {
              const checked = e.target.checked;
              onUpdate((curr) => {
                const next: Record<string, unknown> = {
                  ...curr,
                  stream: checked,
                };
                if (endpoint.kind === "openai-chat") {
                  if (checked) {
                    next.stream_options = { include_usage: true };
                  } else {
                    delete next.stream_options;
                  }
                }
                return next;
              });
            }}
          />
          <span>Stream SSE</span>
        </label>
      )}

      {(endpoint.kind === "openai-chat" ||
        endpoint.kind === "anthropic-messages") && (
        <label className="pg-param pg-checkbox">
          <input
            type="checkbox"
            checked={thinkingEnabled}
            onChange={(e) => {
              const checked = e.target.checked;
              onUpdate((curr) => {
                const next = { ...curr };
                if (endpoint.kind === "openai-chat") {
                  next.chat_template_kwargs = {
                    ...((curr.chat_template_kwargs as
                      Record<string, unknown> | undefined) ?? {}),
                    enable_thinking: checked,
                  };
                } else if (endpoint.kind === "anthropic-messages") {
                  if (checked) {
                    const currMax =
                      typeof curr.max_tokens === "number"
                        ? curr.max_tokens
                        : 256;
                    next.thinking = { type: "enabled", budget_tokens: 1024 };
                    next.max_tokens = Math.min(4096, Math.max(currMax, 2048));
                  } else {
                    next.thinking = { type: "disabled" };
                  }
                }
                return next;
              });
            }}
          />
          <span>Thinking</span>
        </label>
      )}
    </div>
  );
}
