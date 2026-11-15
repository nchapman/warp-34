import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router";
import {
  ENDPOINTS,
  type EndpointSpec,
  type ParsedApiResult,
  type SseFrame,
  type TurnMessage,
  appendAssistantTurnToBody,
  applyMessagesToBody,
  applySseFrameToResult,
  drainSseBuffer,
  emptyParsedResult,
  extractMessagesFromBody,
  formatCurlCommand,
  parseJsonResponse,
} from "./playground";

interface PlaygroundProps {
  model: string | null;
  decisionMode?: boolean;
}

type RequestTab = "structured" | "json";
type ResponseTab = "parsed" | "raw" | "sse" | "curl";

function resolveRouteEndpointAndPreset(
  pathEndpointId: string | undefined,
  searchParams: URLSearchParams,
  fallbackId: string,
): { endpoint: EndpointSpec; presetId: string } {
  const epParam = pathEndpointId ?? searchParams.get("endpoint") ?? fallbackId;
  const endpoint =
    ENDPOINTS.find((e) => e.id === epParam) ??
    ENDPOINTS.find((e) => e.id === fallbackId) ??
    ENDPOINTS[0];
  const presetParam = searchParams.get("preset");
  const validPreset = endpoint.presets.find((p) => p.id === presetParam);
  return {
    endpoint,
    presetId: validPreset?.id ?? endpoint.presets[0]?.id ?? "",
  };
}

export function Playground({ model, decisionMode = false }: PlaygroundProps) {
  const modelName = model ?? "default";
  const defaultEndpointId = decisionMode ? "decisions" : "chat-completions";
  const { endpointId: pathEndpointId } = useParams<{ endpointId?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialRoute = resolveRouteEndpointAndPreset(
    pathEndpointId,
    searchParams,
    defaultEndpointId,
  );

  const [endpointId, setEndpointId] = useState<string>(
    initialRoute.endpoint.id,
  );
  const [presetId, setPresetId] = useState<string>(initialRoute.presetId);
  const [requestTab, setRequestTab] = useState<RequestTab>(
    initialRoute.endpoint.conversational ? "structured" : "json",
  );
  const [responseTab, setResponseTab] = useState<ResponseTab>("parsed");
  const [apiKey, setApiKey] = useState<string>("");

  const updateRouteParams = (nextEpId: string, nextPresetId: string) => {
    setSearchParams({ endpoint: nextEpId, preset: nextPresetId });
  };

  const endpoint: EndpointSpec = useMemo(
    () => ENDPOINTS.find((e) => e.id === endpointId) ?? ENDPOINTS[0],
    [endpointId],
  );

  const [rawJson, setRawJson] = useState<string>(() => {
    const preset =
      initialRoute.endpoint.presets.find(
        (p) => p.id === initialRoute.presetId,
      ) ?? initialRoute.endpoint.presets[0];
    if (!preset || initialRoute.endpoint.method === "GET") return "";
    return JSON.stringify(preset.buildBody(modelName), null, 2);
  });

  const [jsonError, setJsonError] = useState<string | null>(null);
  const [followUpText, setFollowUpText] = useState<string>("");

  // Execution / response state
  const [running, setRunning] = useState<boolean>(false);
  const [statusCode, setStatusCode] = useState<number | null>(null);
  const [statusText, setStatusText] = useState<string>("");
  const [ttftMs, setTtftMs] = useState<number | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const [rawResponse, setRawResponse] = useState<string>("");
  const [sseFrames, setSseFrames] = useState<SseFrame[]>([]);
  const [parsedResult, setParsedResult] =
    useState<ParsedApiResult>(emptyParsedResult);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [copiedCurl, setCopiedCurl] = useState<boolean>(false);

  const abortRef = useRef<AbortController | null>(null);
  const prevModelRef = useRef<string>(modelName);

  // Keep `model` field synced if the server swaps models or resolves from "connecting..."
  useEffect(() => {
    if (prevModelRef.current === modelName) return;
    const oldModel = prevModelRef.current;
    prevModelRef.current = modelName;
    if (!rawJson.trim()) return;
    try {
      const obj = JSON.parse(rawJson) as Record<string, unknown>;
      if (obj.model === oldModel || obj.model === "default") {
        obj.model = modelName;
        setRawJson(JSON.stringify(obj, null, 2));
      }
    } catch {
      // Ignore if user has mid-edit invalid JSON
    }
  }, [modelName, rawJson]);

  // Synchronize endpoint & preset state when URL path or search params change externally
  // (e.g. browser Back/Forward or deep links like `/playground/decisions`).
  useEffect(() => {
    const resolved = resolveRouteEndpointAndPreset(
      pathEndpointId,
      searchParams,
      defaultEndpointId,
    );
    if (resolved.endpoint.id !== endpointId || resolved.presetId !== presetId) {
      setEndpointId(resolved.endpoint.id);
      setPresetId(resolved.presetId);
      const nextPreset =
        resolved.endpoint.presets.find((p) => p.id === resolved.presetId) ??
        resolved.endpoint.presets[0];
      if (!nextPreset || resolved.endpoint.method === "GET") {
        setRawJson("");
      } else {
        setRawJson(JSON.stringify(nextPreset.buildBody(modelName), null, 2));
      }
      setJsonError(null);
      setRequestTab(resolved.endpoint.conversational ? "structured" : "json");
    }
  }, [pathEndpointId, searchParams, defaultEndpointId]);

  const parsedBody = useMemo<Record<string, unknown> | null>(() => {
    if (endpoint.method === "GET" || !rawJson.trim()) return {};
    try {
      const val = JSON.parse(rawJson);
      if (val && typeof val === "object" && !Array.isArray(val)) {
        return val as Record<string, unknown>;
      }
      return null;
    } catch {
      return null;
    }
  }, [rawJson, endpoint.method]);

  const structuredTurns = useMemo<TurnMessage[]>(() => {
    if (!endpoint.conversational || !parsedBody) return [];
    return extractMessagesFromBody(endpoint.kind, parsedBody);
  }, [endpoint.conversational, endpoint.kind, parsedBody]);

  const selectEndpoint = (nextId: string) => {
    const nextEp = ENDPOINTS.find((e) => e.id === nextId) ?? ENDPOINTS[0];
    const firstPreset = nextEp.presets[0];
    const nextPresetId = firstPreset?.id ?? "";
    setEndpointId(nextEp.id);
    setPresetId(nextPresetId);
    updateRouteParams(nextEp.id, nextPresetId);
    if (!firstPreset || nextEp.method === "GET") {
      setRawJson("");
    } else {
      setRawJson(JSON.stringify(firstPreset.buildBody(modelName), null, 2));
    }
    setJsonError(null);
    setRequestTab(nextEp.conversational ? "structured" : "json");
    clearResponseState();
  };

  const selectPreset = (nextPresetId: string) => {
    const preset = endpoint.presets.find((p) => p.id === nextPresetId);
    if (!preset) return;
    setPresetId(preset.id);
    updateRouteParams(endpoint.id, preset.id);
    if (endpoint.method === "GET") {
      setRawJson("");
    } else {
      setRawJson(JSON.stringify(preset.buildBody(modelName), null, 2));
    }
    setJsonError(null);
    clearResponseState();
  };

  const clearResponseState = () => {
    setStatusCode(null);
    setStatusText("");
    setTtftMs(null);
    setTotalMs(null);
    setRawResponse("");
    setSseFrames([]);
    setParsedResult(emptyParsedResult());
    setRequestError(null);
  };

  const updateBodyObject = (
    updater: (current: Record<string, unknown>) => Record<string, unknown>,
  ) => {
    if (!parsedBody) return;
    const updated = updater(parsedBody);
    setRawJson(JSON.stringify(updated, null, 2));
    setJsonError(null);
  };

  const updateTurns = (nextTurns: TurnMessage[]) => {
    updateBodyObject((curr) =>
      applyMessagesToBody(endpoint.kind, curr, nextTurns),
    );
  };

  const handleRawJsonChange = (val: string) => {
    setRawJson(val);
    if (!val.trim() && endpoint.method === "GET") {
      setJsonError(null);
      return;
    }
    try {
      JSON.parse(val);
      setJsonError(null);
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : "Invalid JSON");
    }
  };

  const cancelInFlight = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
  };

  const executeRequest = async (overrideBodyJson?: string) => {
    const bodyToSend = overrideBodyJson ?? rawJson;
    let reqObj: Record<string, unknown> | null = null;
    if (endpoint.method !== "GET") {
      try {
        reqObj = JSON.parse(bodyToSend) as Record<string, unknown>;
        setJsonError(null);
      } catch (e) {
        setJsonError(e instanceof Error ? e.message : "Invalid JSON payload");
        setRequestTab("json");
        return;
      }
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setStatusCode(null);
    setStatusText("");
    setTtftMs(null);
    setTotalMs(null);
    setRawResponse("");
    setSseFrames([]);
    setParsedResult(emptyParsedResult());
    setRequestError(null);

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
        headers.Authorization = `Bearer ${effectiveKey}`;
      }
    }

    const startTime = performance.now();
    let recordedFirstByte = false;

    try {
      const res = await fetch(endpoint.path, {
        method: endpoint.method,
        headers,
        body: endpoint.method === "GET" ? undefined : bodyToSend,
        signal: controller.signal,
      });

      setStatusCode(res.status);
      setStatusText(
        res.statusText ||
          (res.status >= 200 && res.status < 300 ? "OK" : "Error"),
      );
      const contentType = res.headers.get("content-type") ?? "";

      if (contentType.includes("text/event-stream") && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let frameIndex = 0;
        let folded = emptyParsedResult();
        const collectedFrames: SseFrame[] = [];

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const now = performance.now();
          if (!recordedFirstByte) {
            recordedFirstByte = true;
            setTtftMs(now - startTime);
          }
          buffer += decoder.decode(value, { stream: true });
          const drained = drainSseBuffer(buffer, frameIndex, now - startTime);
          buffer = drained.remainder;
          if (drained.frames.length > 0) {
            frameIndex += drained.frames.length;
            for (const f of drained.frames) {
              folded = applySseFrameToResult(endpoint.kind, folded, f);
              if (collectedFrames.length < 512) {
                collectedFrames.push(f);
              }
            }
            setParsedResult(folded);
            if (folded.streamError) {
              setRequestError(folded.streamError);
            }
            setSseFrames([...collectedFrames]);
            const lastNonDone = [...drained.frames]
              .reverse()
              .find((f) => f.data !== "[DONE]");
            if (lastNonDone) {
              setRawResponse(lastNonDone.data);
            }
          }
        }
        setTotalMs(performance.now() - startTime);
      } else {
        const text = await res.text();
        const elapsed = performance.now() - startTime;
        setTtftMs(elapsed);
        setTotalMs(elapsed);
        try {
          const json = JSON.parse(text);
          setRawResponse(JSON.stringify(json, null, 2));
          if (res.ok) {
            setParsedResult(parseJsonResponse(endpoint.kind, json));
          } else {
            const errMsg =
              json?.error?.message ??
              json?.message ??
              `HTTP ${res.status} error`;
            setRequestError(String(errMsg));
          }
        } catch {
          setRawResponse(text);
          if (!res.ok) {
            setRequestError(text || `HTTP ${res.status}`);
          }
        }
      }
    } catch (e) {
      if ((e as { name?: string })?.name === "AbortError") {
        setRequestError("Request cancelled by user");
      } else {
        setRequestError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setRunning(false);
      }
    }
  };

  const handleAppendAndContinue = (andSend: boolean) => {
    if (!parsedBody) return;
    const nextObj = appendAssistantTurnToBody(
      endpoint.kind,
      parsedBody,
      parsedResult,
      followUpText,
    );
    const nextJson = JSON.stringify(nextObj, null, 2);
    setRawJson(nextJson);
    setFollowUpText("");
    if (andSend) {
      void executeRequest(nextJson);
    }
  };

  const origin =
    typeof window !== "undefined"
      ? window.location.origin
      : "http://127.0.0.1:8000";

  const curlCommand = useMemo(
    () => formatCurlCommand(origin, endpoint, rawJson, apiKey),
    [origin, endpoint, rawJson, apiKey],
  );

  const handleCopyCurl = () => {
    void navigator.clipboard?.writeText(curlCommand);
    setCopiedCurl(true);
    setTimeout(() => setCopiedCurl(false), 1500);
  };

  const streamTokPerSec = useMemo(() => {
    const outTok = parsedResult.usage.outputTokens;
    if (!outTok || outTok <= 1 || ttftMs === null || totalMs === null) {
      return null;
    }
    const decodeWindowSec = (totalMs - ttftMs) / 1000;
    if (decodeWindowSec <= 0.005) return null;
    return (outTok - 1) / decodeWindowSec;
  }, [parsedResult.usage.outputTokens, ttftMs, totalMs]);

  const cachedRatioPct = useMemo(() => {
    const { inputTokens, cachedTokens } = parsedResult.usage;
    if (inputTokens && inputTokens > 0 && cachedTokens !== null) {
      return Math.round((cachedTokens / inputTokens) * 100);
    }
    return null;
  }, [parsedResult.usage]);

  const canAppendTurn =
    endpoint.conversational &&
    !running &&
    Boolean(parsedResult.content || parsedResult.toolCalls.length > 0);

  const endpointGroups = [
    "Generative",
    "Decision & Scoring",
    "Tokenizer & Counts",
    "System",
  ] as const;

  return (
    <section className="playground-section" aria-label="API Playground">
      <div className="playground-header">
        <div>
          <h1 className="page-title">API Playground</h1>
          <p className="page-sub">
            Inspect wire payloads, SSE frames, prefix cache hits, and multi-turn
            state against this engine.
          </p>
        </div>
        <div className="playground-auth">
          <label className="pg-field-label" htmlFor="pg-api-key">
            API KEY
          </label>
          <input
            id="pg-api-key"
            type="password"
            className="pg-input pg-auth-input"
            placeholder="Optional Bearer / x-api-key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </div>
      </div>

      <div className="playground-workspace">
        {/* Left Endpoint Sidebar (wide viewports) */}
        <aside className="panel pg-sidebar" aria-label="API Endpoints">
          {endpointGroups.map((group) => (
            <div key={group} className="pg-sidebar-group">
              <div className="pg-sidebar-group-title">{group}</div>
              <div className="pg-sidebar-list">
                {ENDPOINTS.filter((e) => e.group === group).map((ep) => {
                  const active = ep.id === endpoint.id;
                  return (
                    <button
                      key={ep.id}
                      type="button"
                      className={`pg-sidebar-item ${active ? "active" : ""}`}
                      onClick={() => selectEndpoint(ep.id)}
                    >
                      <span
                        className={`method-badge method-${ep.method.toLowerCase()}`}
                      >
                        {ep.method}
                      </span>
                      <span className="pg-sidebar-path">{ep.path}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </aside>

        {/* Main Playground Workbench */}
        <div className="pg-main">
          {/* Preset & Action Bar (includes Endpoint select on narrower screens) */}
          <div className="panel playground-toolbar">
            <div className="pg-field pg-endpoint-mobile">
              <label className="pg-field-label" htmlFor="pg-endpoint-select">
                ENDPOINT
              </label>
              <select
                id="pg-endpoint-select"
                className="pg-select"
                value={endpoint.id}
                onChange={(e) => selectEndpoint(e.target.value)}
              >
                {endpointGroups.map((group) => (
                  <optgroup key={group} label={group}>
                    {ENDPOINTS.filter((e) => e.group === group).map((ep) => (
                      <option key={ep.id} value={ep.id}>
                        {ep.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="pg-field">
              <label className="pg-field-label" htmlFor="pg-preset-select">
                PRESET
                {endpoint.presets.find((p) => p.id === presetId)
                  ?.description ? (
                  <span className="pg-preset-desc">
                    {" · "}
                    {
                      endpoint.presets.find((p) => p.id === presetId)
                        ?.description
                    }
                  </span>
                ) : null}
              </label>
              <select
                id="pg-preset-select"
                className="pg-select"
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
                onClick={handleCopyCurl}
              >
                {copiedCurl ? "Copied cURL" : "Copy cURL"}
              </button>
              {running ? (
                <button
                  type="button"
                  className="pg-btn pg-btn-danger"
                  onClick={cancelInFlight}
                >
                  Stop Stream
                </button>
              ) : (
                <button
                  type="button"
                  className="pg-btn pg-btn-primary"
                  disabled={Boolean(jsonError) && endpoint.method !== "GET"}
                  onClick={() => void executeRequest()}
                >
                  Send Request
                </button>
              )}
            </div>
          </div>

          {/* Split Workbench: Request Left, Response Right */}
          <div className="playground-grid">
            {/* Request Column */}
            <div className="panel playground-pane">
              <div className="pg-pane-head">
                <div className="pg-pane-title">
                  <span
                    className={`method-badge method-${endpoint.method.toLowerCase()}`}
                  >
                    {endpoint.method}
                  </span>
                  <code className="pg-path">{endpoint.path}</code>
                </div>
                <div
                  className="pg-subtabs"
                  role="tablist"
                  aria-label="Request View"
                >
                  {endpoint.conversational && (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={requestTab === "structured"}
                      className={`pg-subtab ${requestTab === "structured" ? "active" : ""}`}
                      onClick={() => setRequestTab("structured")}
                    >
                      Messages &amp; Params
                    </button>
                  )}
                  <button
                    type="button"
                    role="tab"
                    aria-selected={requestTab === "json"}
                    className={`pg-subtab ${requestTab === "json" ? "active" : ""}`}
                    onClick={() => setRequestTab("json")}
                  >
                    JSON Body
                  </button>
                </div>
              </div>

              {jsonError && (
                <div role="alert" className="notice notice-warn pg-notice">
                  <span className="notice-dot" />
                  <span>JSON syntax error: {jsonError}</span>
                </div>
              )}

              {endpoint.method === "GET" ? (
                <div className="pg-empty-body">
                  <code>GET {endpoint.path}</code> takes no request body. Click{" "}
                  <strong>Send Request</strong> to inspect the live response.
                </div>
              ) : requestTab === "structured" &&
                endpoint.conversational &&
                parsedBody ? (
                <div className="pg-structured">
                  <ParameterBar
                    endpoint={endpoint}
                    body={parsedBody}
                    onUpdate={updateBodyObject}
                  />
                  <StructuredTurnEditor
                    endpoint={endpoint}
                    turns={structuredTurns}
                    onChange={updateTurns}
                  />
                </div>
              ) : (
                <div className="pg-json-wrap">
                  <textarea
                    aria-label="Request JSON body"
                    className="pg-json-editor"
                    spellCheck={false}
                    value={rawJson}
                    onChange={(e) => handleRawJsonChange(e.target.value)}
                  />
                </div>
              )}
            </div>

            {/* Response Column */}
            <div className="panel playground-pane">
              <div className="pg-pane-head">
                <div className="pg-pane-title">
                  <span className="pg-section-title">Wire Response</span>
                  {statusCode !== null && (
                    <span
                      className={`pg-status-badge ${statusCode >= 200 && statusCode < 300 ? "ok" : "err"}`}
                    >
                      {statusCode} {statusText}
                    </span>
                  )}
                  {parsedResult.finishReason && (
                    <span className="pg-meta-pill">
                      stop: {parsedResult.finishReason}
                    </span>
                  )}
                </div>
                <div
                  className="pg-subtabs"
                  role="tablist"
                  aria-label="Response View"
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={responseTab === "parsed"}
                    className={`pg-subtab ${responseTab === "parsed" ? "active" : ""}`}
                    onClick={() => setResponseTab("parsed")}
                  >
                    Parsed
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={responseTab === "raw"}
                    className={`pg-subtab ${responseTab === "raw" ? "active" : ""}`}
                    onClick={() => setResponseTab("raw")}
                  >
                    Response JSON
                  </button>
                  {endpoint.supportsStream && (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={responseTab === "sse"}
                      className={`pg-subtab ${responseTab === "sse" ? "active" : ""}`}
                      onClick={() => setResponseTab("sse")}
                    >
                      SSE ({sseFrames.length})
                    </button>
                  )}
                  <button
                    type="button"
                    role="tab"
                    aria-selected={responseTab === "curl"}
                    className={`pg-subtab ${responseTab === "curl" ? "active" : ""}`}
                    onClick={() => setResponseTab("curl")}
                  >
                    cURL
                  </button>
                </div>
              </div>

              {/* Telemetry Ribbon */}
              <div className="pg-telemetry-bar">
                <div className="pg-metric">
                  <span className="pg-metric-k">CLIENT TTFT</span>
                  <span className="pg-metric-v">
                    {ttftMs !== null ? `${ttftMs.toFixed(1)} ms` : "—"}
                  </span>
                </div>
                <div className="pg-metric">
                  <span className="pg-metric-k">TOTAL E2E</span>
                  <span className="pg-metric-v">
                    {totalMs !== null ? `${totalMs.toFixed(1)} ms` : "—"}
                  </span>
                </div>
                <div className="pg-metric">
                  <span className="pg-metric-k">STREAM PACE</span>
                  <span className="pg-metric-v">
                    {streamTokPerSec !== null
                      ? `${streamTokPerSec.toFixed(1)} tok/s`
                      : "—"}
                  </span>
                </div>
                <div className="pg-metric">
                  <span className="pg-metric-k">PROMPT / CACHED</span>
                  <span className="pg-metric-v">
                    {parsedResult.usage.inputTokens ?? "—"}
                    {" / "}
                    <span
                      className={
                        parsedResult.usage.cachedTokens &&
                        parsedResult.usage.cachedTokens > 0
                          ? "pg-accent"
                          : ""
                      }
                    >
                      {parsedResult.usage.cachedTokens ?? "—"}
                      {cachedRatioPct !== null ? ` (${cachedRatioPct}%)` : ""}
                    </span>
                  </span>
                </div>
                <div className="pg-metric">
                  <span className="pg-metric-k">OUTPUT TOKENS</span>
                  <span className="pg-metric-v">
                    {parsedResult.usage.outputTokens ?? "—"}
                  </span>
                </div>
              </div>

              {requestError && (
                <div role="alert" className="notice notice-err pg-notice">
                  <span className="notice-dot" />
                  <span>{requestError}</span>
                </div>
              )}

              {responseTab === "parsed" && (
                <div className="pg-response-body">
                  {parsedResult.reasoning && (
                    <div className="pg-reasoning-box">
                      <div className="pg-box-label">
                        REASONING / THINKING STREAM
                      </div>
                      <pre className="pg-pre pg-reasoning-pre">
                        {parsedResult.reasoning}
                      </pre>
                    </div>
                  )}

                  {parsedResult.toolCalls.length > 0 && (
                    <div className="pg-tools-box">
                      <div className="pg-box-label">TOOL CALLS</div>
                      {parsedResult.toolCalls.map((tc, i) => (
                        <div key={`${tc.id}-${i}`} className="pg-tool-call">
                          <div className="pg-tool-head">
                            <code>{tc.name}</code>
                            <span className="pg-tool-id">{tc.id}</span>
                          </div>
                          <pre className="pg-pre">{tc.arguments}</pre>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="pg-output-box">
                    <div className="pg-box-label">
                      {endpoint.conversational
                        ? "ASSISTANT OUTPUT"
                        : "RESPONSE PAYLOAD"}
                      {parsedResult.responseId ? (
                        <span className="pg-resp-id">
                          {parsedResult.responseId}
                        </span>
                      ) : null}
                    </div>
                    <pre className="pg-pre pg-output-pre">
                      {parsedResult.content ||
                        (running
                          ? "Awaiting tokens..."
                          : "Send a request to inspect the response.")}
                    </pre>
                  </div>

                  {canAppendTurn && (
                    <div className="pg-continue-bar">
                      <div className="pg-box-label">
                        CONTINUE CONVERSATION (TESTS PREFIX CACHE &amp; STATEFUL
                        CHAINS)
                      </div>
                      <div className="pg-continue-controls">
                        <input
                          type="text"
                          aria-label="Follow-up user message"
                          className="pg-input pg-continue-input"
                          placeholder={
                            parsedResult.toolCalls.length > 0
                              ? 'Tool output JSON (default: {"status":"ok"})...'
                              : "Type follow-up user turn..."
                          }
                          value={followUpText}
                          onChange={(e) => setFollowUpText(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              handleAppendAndContinue(true);
                            }
                          }}
                        />
                        <button
                          type="button"
                          className="pg-btn pg-btn-secondary"
                          onClick={() => handleAppendAndContinue(false)}
                        >
                          Append Only
                        </button>
                        <button
                          type="button"
                          className="pg-btn pg-btn-primary"
                          onClick={() => handleAppendAndContinue(true)}
                        >
                          Append &amp; Send
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {responseTab === "raw" && (
                <div className="pg-response-body">
                  <pre className="pg-pre pg-raw-pre">
                    {rawResponse || "// No response body captured yet"}
                  </pre>
                </div>
              )}

              {responseTab === "sse" && (
                <div className="pg-response-body pg-sse-list">
                  {sseFrames.length === 0 ? (
                    <div className="pg-empty-body">
                      No SSE frames recorded. Enable{" "}
                      <code>&quot;stream&quot;: true</code> and send a request.
                    </div>
                  ) : (
                    sseFrames.map((f) => (
                      <div key={f.index} className="pg-sse-frame">
                        <div className="pg-sse-meta">
                          <span className="pg-sse-idx">#{f.index}</span>
                          <span className="pg-sse-time">
                            +{f.elapsedMs.toFixed(1)} ms
                          </span>
                          {f.event && (
                            <span className="pg-sse-event">
                              event: {f.event}
                            </span>
                          )}
                        </div>
                        <pre className="pg-pre pg-sse-data">{f.data}</pre>
                      </div>
                    ))
                  )}
                </div>
              )}

              {responseTab === "curl" && (
                <div className="pg-response-body">
                  <pre className="pg-pre pg-raw-pre">{curlCommand}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

interface ParameterBarProps {
  endpoint: EndpointSpec;
  body: Record<string, unknown>;
  onUpdate: (
    updater: (curr: Record<string, unknown>) => Record<string, unknown>,
  ) => void;
}

function ParameterBar({ endpoint, body, onUpdate }: ParameterBarProps) {
  const maxTokensKey =
    endpoint.kind === "openai-responses" ? "max_output_tokens" : "max_tokens";
  const maxTokens =
    typeof body[maxTokensKey] === "number"
      ? (body[maxTokensKey] as number)
      : 256;
  const temperature =
    typeof body.temperature === "number" ? (body.temperature as number) : 0;
  const stream = Boolean(body.stream);

  const thinkingEnabled = useMemo(() => {
    if (endpoint.kind === "openai-chat") {
      const kw = body.chat_template_kwargs as
        Record<string, unknown> | undefined;
      return kw?.enable_thinking !== false;
    }
    if (endpoint.kind === "anthropic-messages") {
      const th = body.thinking as Record<string, unknown> | undefined;
      return th?.type === "enabled";
    }
    return false;
  }, [endpoint.kind, body]);

  const minMaxTokens =
    endpoint.kind === "anthropic-messages" && thinkingEnabled ? 1025 : 1;

  return (
    <div className="pg-param-bar">
      <div className="pg-param-item">
        <label className="pg-field-label" htmlFor="pg-max-tokens">
          {maxTokensKey.toUpperCase()}
        </label>
        <input
          id="pg-max-tokens"
          type="number"
          min={minMaxTokens}
          max={4096}
          className="pg-input pg-num-input"
          value={maxTokens}
          onChange={(e) => {
            const val = Number.parseInt(e.target.value, 10);
            if (!Number.isNaN(val) && val > 0) {
              const clamped = Math.min(4096, Math.max(minMaxTokens, val));
              onUpdate((curr) => ({ ...curr, [maxTokensKey]: clamped }));
            }
          }}
        />
      </div>

      <div className="pg-param-item">
        <label className="pg-field-label" htmlFor="pg-temperature">
          TEMPERATURE
        </label>
        <input
          id="pg-temperature"
          type="number"
          step={0.1}
          min={0}
          max={2}
          className="pg-input pg-num-input"
          value={temperature}
          onChange={(e) => {
            const val = Number.parseFloat(e.target.value);
            if (!Number.isNaN(val) && val >= 0) {
              onUpdate((curr) => ({
                ...curr,
                temperature: Math.min(2, val),
              }));
            }
          }}
        />
      </div>

      {endpoint.supportsStream && (
        <label className="pg-toggle">
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
        <label className="pg-toggle">
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
                    next.thinking = {
                      type: "enabled",
                      budget_tokens: 1024,
                    };
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

interface StructuredTurnEditorProps {
  endpoint: EndpointSpec;
  turns: TurnMessage[];
  onChange: (next: TurnMessage[]) => void;
}

function StructuredTurnEditor({
  endpoint,
  turns,
  onChange,
}: StructuredTurnEditorProps) {
  const updateTurn = (idx: number, patch: Partial<TurnMessage>) => {
    const next = turns.map((t, i) => (i === idx ? { ...t, ...patch } : t));
    onChange(next);
  };

  const removeTurn = (idx: number) => {
    onChange(turns.filter((_, i) => i !== idx));
  };

  const addTurn = (role: TurnMessage["role"]) => {
    onChange([...turns, { role, content: "" }]);
  };

  return (
    <div className="pg-turns">
      <div className="pg-turns-head">
        <span className="pg-box-label">MESSAGES ARRAY</span>
        <div className="pg-turn-add-group">
          <button
            type="button"
            className="pg-mini-btn"
            onClick={() => addTurn("user")}
          >
            + User turn
          </button>
          <button
            type="button"
            className="pg-mini-btn"
            onClick={() => addTurn("assistant")}
          >
            + Assistant turn
          </button>
        </div>
      </div>

      {turns.map((turn, idx) => (
        <div key={idx} className={`pg-turn-card pg-role-${turn.role}`}>
          <div className="pg-turn-top">
            <select
              aria-label={`Turn ${idx + 1} Role`}
              className="pg-role-select"
              value={turn.role}
              onChange={(e) =>
                updateTurn(idx, {
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
            <span className="pg-turn-idx">#{idx}</span>
            {turn.rawContentBlocks && turn.rawContentBlocks.length > 0 && (
              <span className="pg-meta-pill">
                +{turn.rawContentBlocks.length} multimodal block
                {turn.rawContentBlocks.length > 1 ? "s" : ""} preserved
              </span>
            )}
            {turn.tool_calls && turn.tool_calls.length > 0 && (
              <span className="pg-meta-pill">
                +{turn.tool_calls.length} tool_call
                {turn.tool_calls.length > 1 ? "s" : ""}
              </span>
            )}
            <button
              type="button"
              className="pg-turn-remove"
              aria-label={`Remove turn ${idx + 1}`}
              onClick={() => removeTurn(idx)}
            >
              ×
            </button>
          </div>
          <textarea
            aria-label={`Turn ${idx + 1} content`}
            className="pg-turn-textarea"
            rows={Math.min(8, Math.max(2, turn.content.split("\n").length))}
            value={turn.content}
            onChange={(e) => updateTurn(idx, { content: e.target.value })}
            placeholder={`Enter ${turn.role} message...`}
          />
        </div>
      ))}
    </div>
  );
}
