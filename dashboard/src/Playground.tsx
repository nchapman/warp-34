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
  model?: string | null,
): {
  endpoint: EndpointSpec;
  presetId: string;
} {
  const requestedEndpoint = searchParams.get("endpoint") ?? pathEndpointId;
  const isImageModel = Boolean(model?.toLowerCase().includes("qwen-image"));
  const fallbackId = decisionMode
    ? "decisions"
    : isImageModel
      ? "images-generations"
      : "chat-completions";
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
      // Transcode WebP/GIF/BMP to PNG data URL so warp34's native Rust PNG/JPEG decoder accepts it
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
  const isImageModel = Boolean(model?.toLowerCase().includes("qwen-image"));
  const defaultEndpointId = decisionMode
    ? "decisions"
    : isImageModel
      ? "images-generations"
      : "chat-completions";
  const initial = resolveRoute(
    pathEndpointId,
    searchParams,
    decisionMode,
    model,
  );
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
    initial.endpoint.conversational || initial.endpoint.kind === "openai-images"
      ? "structured"
      : "json",
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
    const resolved = resolveRoute(
      pathEndpointId,
      searchParams,
      decisionMode,
      model,
    );
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
      setRequestTab(
        resolved.endpoint.conversational ||
          resolved.endpoint.kind === "openai-images"
          ? "structured"
          : "json",
      );
    }
  }, [pathEndpointId, searchParams, decisionMode, model]);

  function selectEndpoint(nextId: string) {
    const nextSpec = ENDPOINTS.find((e) => e.id === nextId) ?? ENDPOINTS[0];
    const firstPreset = nextSpec.presets[0];
    setEndpointId(nextSpec.id);
    setPresetId(firstPreset.id);
    updateRouteParams(nextSpec.id, firstPreset.id);
    const body = firstPreset.buildBody(activeModel);
    setRawJson(nextSpec.method === "GET" ? "" : JSON.stringify(body, null, 2));
    setJsonError(null);
    setRequestTab(
      nextSpec.conversational || nextSpec.kind === "openai-images"
        ? "structured"
        : "json",
    );
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

  function handleSendGeneratedToImageEdit(imageUrl: string) {
    const editSpec = ENDPOINTS.find((e) => e.id === "images-edits");
    if (!editSpec) return;
    const preset = editSpec.presets[0];
    setEndpointId(editSpec.id);
    setPresetId(preset.id);
    updateRouteParams(editSpec.id, preset.id);
    const baseBody = preset.buildBody(activeModel);
    const nextBody: Record<string, unknown> = {
      ...baseBody,
      image: [imageUrl],
    };
    setRawJson(JSON.stringify(nextBody, null, 2));
    setJsonError(null);
    setRequestTab("structured");
  }

  async function handleAttachConditionImages(files: FileList | File[]) {
    const fileArr = Array.from(files).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (fileArr.length === 0) return;
    const currentImgs = extractConditionImages(bodyObj);
    const remainingBudget = Math.max(
      0,
      MAX_IMAGES_PER_REQUEST - currentImgs.length,
    );
    const toRead = fileArr.slice(0, remainingBudget);
    if (toRead.length === 0) return;
    try {
      const dataUrls = await Promise.all(toRead.map(readFileAsEngineDataUrl));
      setRequestError(null);
      updateBodyObject((curr) => {
        const nextImgs = [...extractConditionImages(curr), ...dataUrls];
        const next: Record<string, unknown> = { ...curr, image: nextImgs };
        delete next.images;
        return next;
      });
    } catch (err) {
      setRequestError(
        err instanceof Error ? err.message : "Failed to attach image.",
      );
    }
  }

  function handleRemoveConditionImage(imgIdx: number) {
    updateBodyObject((curr) => {
      const nextImgs = extractConditionImages(curr).filter(
        (_, i) => i !== imgIdx,
      );
      const next = { ...curr };
      delete next.images;
      if (nextImgs.length > 0) {
        next.image = nextImgs;
      } else {
        delete next.image;
      }
      return next;
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
                {(endpoint.conversational ||
                  endpoint.kind === "openai-images") && (
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
                      {endpoint.kind === "openai-images"
                        ? "Prompt & Params"
                        : "Messages & Params"}
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

              {endpoint.kind === "openai-images" &&
              requestTab === "structured" &&
              bodyObj ? (
                <ImageRequestEditor
                  endpoint={endpoint}
                  body={bodyObj}
                  onUpdate={updateBodyObject}
                  onAttachConditionImages={handleAttachConditionImages}
                  onRemoveConditionImage={handleRemoveConditionImage}
                  onSubmit={() => void executeRequest()}
                />
              ) : endpoint.conversational &&
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
                            <span className="pg-badge violet">
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
                    <span className="pg-badge violet">Streaming…</span>
                  )}
                  {callMetrics.status !== null && (
                    <span
                      className={`pg-badge ${
                        callMetrics.status >= 200 && callMetrics.status < 300
                          ? "aqua"
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
                    Raw JSON
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
                  <span className="eyebrow">TTFT</span>
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
                  <span className="eyebrow">PACE</span>
                  <strong className="numeric">
                    {decodeTokPerSec ? `${decodeTokPerSec} tok/s` : "—"}
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">IN / CACHED</span>
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
                  <span className="eyebrow">OUTPUT</span>
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
                    {parsedResult.images && parsedResult.images.length > 0 && (
                      <div
                        className="pg-image-output-grid"
                        style={{
                          display: "grid",
                          gridTemplateColumns:
                            "repeat(auto-fit, minmax(260px, 1fr))",
                          gap: "12px",
                          marginTop: "12px",
                        }}
                      >
                        {parsedResult.images.map((src, idx) => (
                          <figure
                            key={idx}
                            style={{
                              margin: 0,
                              padding: "8px",
                              borderRadius: "6px",
                              border: "1px solid var(--border, #2a2f3a)",
                              background:
                                "repeating-conic-gradient(#1b1f27 0% 25%, #242936 0% 50%) 50% / 16px 16px",
                            }}
                          >
                            <img
                              src={src}
                              alt={`Generated output ${idx + 1}`}
                              style={{
                                width: "100%",
                                height: "auto",
                                display: "block",
                                borderRadius: "4px",
                              }}
                            />
                            <figcaption
                              className="mono muted"
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                gap: "8px",
                                marginTop: "6px",
                                fontSize: "12px",
                              }}
                            >
                              <span>Output #{idx + 1}</span>
                              <span
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  gap: "8px",
                                }}
                              >
                                <button
                                  type="button"
                                  className="pg-mini-btn"
                                  onClick={() =>
                                    handleSendGeneratedToImageEdit(src)
                                  }
                                >
                                  ↪ Edit Image
                                </button>
                                <a
                                  href={src}
                                  download={`qwen-image-${idx + 1}.png`}
                                  style={{ color: "var(--accent, #7aa2f7)" }}
                                >
                                  Download PNG
                                </a>
                              </span>
                            </figcaption>
                          </figure>
                        ))}
                      </div>
                    )}
                  </div>

                  {hasAssistantReply && !inFlight && (
                    <div className="pg-continue-box">
                      <div className="pg-continue-header">
                        <span className="eyebrow">MULTI-TURN CONTINUATION</span>
                        <button
                          type="button"
                          className="pg-mini-btn"
                          onClick={() => handleAppendAndContinue(false)}
                        >
                          ↪ Append assistant turn
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
                            <span className="violet">event: {f.event}</span>
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

function extractConditionImages(
  body: Record<string, unknown> | null,
): string[] {
  if (!body) return [];
  const out: string[] = [];
  for (const key of ["image", "images"] as const) {
    const val = body[key];
    if (typeof val === "string" && val.trim().length > 0) {
      out.push(val.trim());
    } else if (Array.isArray(val)) {
      for (const item of val) {
        if (typeof item === "string" && item.trim().length > 0) {
          out.push(item.trim());
        }
      }
    }
  }
  return out;
}

const IMAGE_SIZE_OPTIONS = [
  "1024x1024",
  "512x512",
  "768x768",
  "1024x768",
  "768x1024",
  "auto",
] as const;

function ImageRequestEditor({
  endpoint,
  body,
  onUpdate,
  onAttachConditionImages,
  onRemoveConditionImage,
  onSubmit,
}: {
  endpoint: EndpointSpec;
  body: Record<string, unknown>;
  onUpdate: (
    updater: (b: Record<string, unknown>) => Record<string, unknown>,
  ) => void;
  onAttachConditionImages: (files: FileList | File[]) => Promise<void>;
  onRemoveConditionImage: (index: number) => void;
  onSubmit: () => void;
}) {
  const isEditEndpoint = endpoint.id === "images-edits";
  const prompt = typeof body.prompt === "string" ? body.prompt : "";
  const negativePrompt =
    typeof body.negative_prompt === "string" ? body.negative_prompt : "";
  const size = typeof body.size === "string" ? body.size : "1024x1024";
  const steps =
    typeof body.steps === "number"
      ? body.steps
      : typeof body.num_inference_steps === "number"
        ? body.num_inference_steps
        : 8;
  const guidanceScale =
    typeof body.guidance_scale === "number"
      ? body.guidance_scale
      : typeof body.true_cfg_scale === "number"
        ? body.true_cfg_scale
        : 1.0;
  const seed = typeof body.seed === "number" ? body.seed : 42;
  const transparentBg = body.background === "transparent" || body.rgba === true;
  const conditionImages = extractConditionImages(body);

  return (
    <div className="pg-structured">
      <div className="pg-params-bar">
        <label className="pg-param">
          <span className="eyebrow">SIZE</span>
          <select
            aria-label="Image size"
            value={
              IMAGE_SIZE_OPTIONS.includes(
                size as (typeof IMAGE_SIZE_OPTIONS)[number],
              )
                ? size
                : "1024x1024"
            }
            onChange={(e) => {
              const nextSize = e.target.value;
              onUpdate((curr) => ({ ...curr, size: nextSize }));
            }}
          >
            {IMAGE_SIZE_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </label>

        <label className="pg-param">
          <span className="eyebrow">STEPS</span>
          <input
            aria-label="Inference steps"
            type="number"
            min={1}
            max={100}
            value={steps}
            onChange={(e) => {
              const val = Math.min(
                100,
                Math.max(1, Number(e.target.value) || 8),
              );
              onUpdate((curr) => {
                const next: Record<string, unknown> = { ...curr, steps: val };
                delete next.num_inference_steps;
                return next;
              });
            }}
          />
        </label>

        <label className="pg-param">
          <span className="eyebrow">CFG SCALE</span>
          <input
            aria-label="Guidance scale"
            type="number"
            min={0}
            max={30}
            step={0.5}
            value={guidanceScale}
            onChange={(e) => {
              const val = Math.min(
                30,
                Math.max(0, Number(e.target.value) || 0),
              );
              onUpdate((curr) => {
                const next: Record<string, unknown> = {
                  ...curr,
                  guidance_scale: val,
                };
                delete next.true_cfg_scale;
                return next;
              });
            }}
          />
        </label>

        <label className="pg-param">
          <span className="eyebrow">SEED</span>
          <input
            aria-label="Random seed"
            type="number"
            min={0}
            value={seed}
            onChange={(e) => {
              const val = Math.max(0, Number(e.target.value) || 0);
              onUpdate((curr) => ({ ...curr, seed: val }));
            }}
          />
        </label>

        <label className="pg-param pg-checkbox">
          <input
            aria-label="Transparent RGBA"
            type="checkbox"
            checked={transparentBg}
            onChange={(e) => {
              const checked = e.target.checked;
              onUpdate((curr) => {
                const next: Record<string, unknown> = {
                  ...curr,
                  background: checked ? "transparent" : "opaque",
                };
                delete next.rgba;
                return next;
              });
            }}
          />
          <span>Transparent RGBA</span>
        </label>
      </div>

      {(isEditEndpoint || conditionImages.length > 0) && (
        <div
          className="pg-message-row"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            if (e.dataTransfer?.files?.length) {
              e.preventDefault();
              void onAttachConditionImages(e.dataTransfer.files);
            }
          }}
        >
          <div className="pg-message-meta">
            <span className="eyebrow">
              CONDITION IMAGES ({conditionImages.length}/
              {MAX_IMAGES_PER_REQUEST})
            </span>
            <div className="pg-turn-actions">
              {conditionImages.length < MAX_IMAGES_PER_REQUEST && (
                <label className="pg-mini-btn pg-upload-label">
                  + Condition Image
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp,image/gif"
                    multiple
                    aria-label="Attach condition image"
                    className="pg-upload-input"
                    onChange={(e) => {
                      if (e.target.files && e.target.files.length > 0) {
                        void onAttachConditionImages(e.target.files);
                        e.target.value = "";
                      }
                    }}
                  />
                </label>
              )}
            </div>
          </div>

          {conditionImages.length > 0 && (
            <div className="pg-turn-images">
              {conditionImages.map((imgUrl, imgIdx) => (
                <div key={imgIdx} className="pg-turn-thumb">
                  <img
                    src={imgUrl}
                    alt={`Condition image ${imgIdx + 1}`}
                    className="pg-turn-thumb-img"
                  />
                  <span className="pg-turn-thumb-meta mono">
                    IMG #{imgIdx + 1}
                  </span>
                  <button
                    type="button"
                    className="pg-remove-btn pg-thumb-remove"
                    aria-label={`Remove condition image ${imgIdx + 1}`}
                    onClick={() => onRemoveConditionImage(imgIdx)}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="pg-messages-list">
        <div
          className="pg-message-row"
          onPaste={(e) => {
            if (isEditEndpoint && e.clipboardData?.files?.length) {
              const imgFiles = Array.from(e.clipboardData.files).filter((f) =>
                f.type.startsWith("image/"),
              );
              if (imgFiles.length > 0) {
                e.preventDefault();
                void onAttachConditionImages(imgFiles);
              }
            }
          }}
        >
          <div className="pg-message-meta">
            <span className="eyebrow">PROMPT</span>
          </div>
          <textarea
            aria-label="Image prompt"
            rows={4}
            value={prompt}
            placeholder="Describe the image to generate or edit…"
            onChange={(e) => {
              const val = e.target.value;
              onUpdate((curr) => ({ ...curr, prompt: val }));
            }}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                onSubmit();
              }
            }}
          />
        </div>

        <div className="pg-message-row">
          <div className="pg-message-meta">
            <span className="eyebrow">
              NEGATIVE PROMPT (OPTIONAL, CFG &gt; 1)
            </span>
          </div>
          <textarea
            aria-label="Negative prompt"
            rows={2}
            value={negativePrompt}
            placeholder="Optional negative prompt when guidance_scale > 1.0…"
            onChange={(e) => {
              const val = e.target.value;
              onUpdate((curr) => {
                const next = { ...curr };
                if (val.trim().length > 0) {
                  next.negative_prompt = val;
                } else {
                  delete next.negative_prompt;
                }
                return next;
              });
            }}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                onSubmit();
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}
