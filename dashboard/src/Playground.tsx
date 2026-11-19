import { Card, CardHeader, CardContent } from "./components/ui/card";
import { Alert, AlertDescription } from "./components/ui/alert";
import { Badge } from "./components/ui/badge";
import {
  Field,
  FieldLabel,
  FieldDescription,
  FieldError,
} from "./components/ui/field";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { NativeSelect } from "./components/ui/native-select";
import { Switch } from "./components/ui/switch";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { Send, Square, Copy, Plus, X } from "lucide-react";
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
  loraAdapters?: string[];
  maxContextTokens?: number | null;
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
const NON_STREAM_TIMEOUT_MS = 600_000;
const MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024;
const MAX_TOTAL_IMAGE_DATA_URL_CHARS = 28 * 1024 * 1024;
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

export function Playground({
  model,
  loraAdapters = [],
  maxContextTokens = null,
  decisionMode = false,
}: PlaygroundProps) {
  const navigate = useNavigate();
  const { endpointId: pathEndpointId } = useParams<{ endpointId?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const defaultEndpointId = decisionMode ? "decisions" : "chat-completions";
  const initial = resolveRoute(pathEndpointId, searchParams, decisionMode);
  const [endpointId, setEndpointId] = useState<string>(initial.endpoint.id);
  const endpoint = ENDPOINTS.find((e) => e.id === endpointId) ?? ENDPOINTS[0];

  const [presetId, setPresetId] = useState<string>(initial.presetId);
  const [selectedAdapter, setSelectedAdapter] = useState<string>("");
  const baseModel = model ?? DEFAULT_MODEL;
  const activeModel =
    selectedAdapter && loraAdapters.includes(selectedAdapter)
      ? selectedAdapter
      : baseModel;
  const maxTokensLimit =
    maxContextTokens && maxContextTokens > 0 ? maxContextTokens : 1_048_576;

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
  const [copyError, setCopyError] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const prevModelRef = useRef<string>(activeModel);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
    };
  }, []);

  // Keep the request JSON's `model` field synced when the first live metrics snapshot arrives
  useEffect(() => {
    if (inFlight || prevModelRef.current === activeModel) return;
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
  }, [activeModel, rawJson, inFlight]);

  // Synchronize endpoint & preset state when URL path or search params change externally
  useEffect(() => {
    if (inFlight) return;
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
  }, [pathEndpointId, searchParams, decisionMode, inFlight]);

  function selectEndpoint(nextId: string) {
    if (inFlight) return;
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
    if (inFlight) return;
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
    if (inFlight) return;
    const fileArr = Array.from(files).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (fileArr.length === 0) return;
    const remainingBudget = Math.max(0, MAX_IMAGES_PER_REQUEST - totalImages);
    const toRead = fileArr.slice(0, remainingBudget);
    if (toRead.length === 0) return;
    try {
      const requestAtStart = abortRef.current;
      const dataUrls = await Promise.all(toRead.map(readFileAsEngineDataUrl));
      // A request started while decoding: do not mutate its editor snapshot.
      if (abortRef.current !== requestAtStart) return;
      const existingChars = messages.reduce(
        (sum, m) =>
          sum + (m.images?.reduce((acc, u) => acc + u.length, 0) ?? 0),
        0,
      );
      const addedChars = dataUrls.reduce((acc, u) => acc + u.length, 0);
      if (existingChars + addedChars > MAX_TOTAL_IMAGE_DATA_URL_CHARS) {
        setRequestError(
          "Attached images exceed the 32 MiB request body limit.",
        );
        return;
      }
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
    if (inFlight) return;
    const payloadText = overrideJson ?? rawJson;
    let reqObj: Record<string, unknown> | null = null;
    if (endpoint.method !== "GET") {
      if (
        new TextEncoder().encode(payloadText).byteLength >
        MAX_REQUEST_BODY_BYTES
      ) {
        setRequestError(
          "Request body exceeds 32 MiB server limit (reduce attached image size or count).",
        );
        return;
      }
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
    const initialTimeoutMs =
      endpoint.method !== "GET" && reqObj?.stream !== true
        ? NON_STREAM_TIMEOUT_MS
        : REQUEST_IDLE_TIMEOUT_MS;
    const resetIdleTimer = (ms: number = REQUEST_IDLE_TIMEOUT_MS) => {
      if (idleTimer !== null) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, ms);
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
      resetIdleTimer(initialTimeoutMs);
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
    setCopyError(false);
    try {
      if (!navigator.clipboard?.writeText)
        throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(curlCommand);
      setCopiedCurl(true);
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
      copyTimerRef.current = setTimeout(() => setCopiedCurl(false), 1500);
    } catch {
      setCopiedCurl(false);
      setCopyError(true);
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
        <Field className="playground-auth w-auto gap-2">
          <FieldLabel htmlFor="pg-api-key" className="eyebrow text-[11px]">
            API KEY
          </FieldLabel>
          <Input
            disabled={inFlight}
            id="pg-api-key"
            type="password"
            placeholder="Optional Bearer / x-api-key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </Field>
      </div>

      <div className="playground-workspace">
        <Card
          className="pg-sidebar gap-5 p-4 max-[1051px]:hidden"
          role="complementary"
          aria-label="API endpoints"
        >
          {(
            [
              "Generative",
              "Decision & Scoring",
              "Tokenizer & Counts",
              "System",
            ] as const
          ).map((group) => (
            <div key={group} className="pg-sidebar-group">
              <span className="eyebrow text-[11px]">{group.toUpperCase()}</span>
              <div className="pg-sidebar-items">
                {ENDPOINTS.filter((ep) => ep.group === group).map((ep) => (
                  <Button
                    key={ep.id}
                    variant="ghost"
                    disabled={inFlight}
                    type="button"
                    aria-current={endpoint.id === ep.id ? "true" : undefined}
                    title={ep.path}
                    className="w-full justify-start border border-transparent px-2 text-left text-muted-foreground aria-current:border-input aria-current:bg-accent aria-current:text-foreground"
                    onClick={() => selectEndpoint(ep.id)}
                  >
                    <MethodBadge method={ep.method} />
                    <span className="pg-sidebar-path mono">{ep.path}</span>
                  </Button>
                ))}
              </div>
            </div>
          ))}
        </Card>

        <div className="pg-main">
          <Card
            role="region"
            className="playground-toolbar flex-row gap-3 p-4"
            aria-label="Endpoint controls"
          >
            <Field className="pg-field gap-2 min-[1051px]:hidden">
              <FieldLabel htmlFor="pg-endpoint" className="eyebrow text-[11px]">
                ENDPOINT
              </FieldLabel>
              <NativeSelect
                disabled={inFlight}
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
              </NativeSelect>
            </Field>

            <Field className="pg-field gap-2">
              <FieldLabel htmlFor="pg-preset" className="eyebrow text-[11px]">
                PRESET
              </FieldLabel>
              <NativeSelect
                disabled={inFlight}
                id="pg-preset"
                aria-describedby="pg-preset-hint"
                value={presetId}
                onChange={(e) => selectPreset(e.target.value)}
              >
                {endpoint.presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </NativeSelect>
              <FieldDescription id="pg-preset-hint">
                {endpoint.presets.find((p) => p.id === presetId)?.description}
              </FieldDescription>
            </Field>

            {loraAdapters.length > 0 && endpoint.method !== "GET" && (
              <Field className="pg-field gap-2">
                <FieldLabel htmlFor="pg-model" className="eyebrow text-[11px]">
                  MODEL / LORA
                </FieldLabel>
                <NativeSelect
                  disabled={inFlight}
                  id="pg-model"
                  value={
                    selectedAdapter && loraAdapters.includes(selectedAdapter)
                      ? selectedAdapter
                      : ""
                  }
                  onChange={(e) => setSelectedAdapter(e.target.value)}
                >
                  <option value="">{baseModel} (base)</option>
                  {loraAdapters.map((adapter) => (
                    <option key={adapter} value={adapter}>
                      {adapter} (LoRA)
                    </option>
                  ))}
                </NativeSelect>
              </Field>
            )}

            <div className="pg-actions">
              <Button
                type="button"
                variant="outline"
                onClick={() => void handleCopyCurl()}
              >
                <Copy aria-hidden="true" />
                <span aria-live="polite">
                  {copiedCurl ? "Copied cURL" : "Copy cURL"}
                </span>
              </Button>
              <Button
                type="button"
                variant={inFlight ? "destructive" : "default"}
                disabled={!inFlight && Boolean(jsonError)}
                onClick={() =>
                  inFlight ? abortRef.current?.abort() : void executeRequest()
                }
              >
                {inFlight ? (
                  <Square aria-hidden="true" />
                ) : (
                  <Send aria-hidden="true" />
                )}
                {inFlight ? "Abort" : "Send Request"}
              </Button>
            </div>
          </Card>

          {copyError && (
            <Alert variant="destructive">
              <AlertDescription>
                Could not copy. Select and copy the command from the cURL tab.
              </AlertDescription>
            </Alert>
          )}
          <div className="playground-grid">
            <Card
              role="region"
              className="playground-pane gap-0 p-4"
              aria-labelledby="req-pane-title"
            >
              <Tabs
                value={requestTab}
                onValueChange={(value) => {
                  if (value === "structured" || value === "json")
                    setRequestTab(value);
                }}
              >
                <CardHeader className="pg-pane-heading flex flex-row flex-wrap items-center justify-between px-0">
                  <div className="pg-title-group">
                    <MethodBadge method={endpoint.method} />
                    <h2 id="req-pane-title" className="mono">
                      {endpoint.path}
                    </h2>
                  </div>
                  {endpoint.conversational && (
                    <TabsList aria-label="Request editor mode">
                      <TabsTrigger value="structured">
                        Messages & Params
                      </TabsTrigger>
                      <TabsTrigger value="json">JSON Body</TabsTrigger>
                    </TabsList>
                  )}
                </CardHeader>

                <CardContent className="px-0">
                  <fieldset
                    disabled={inFlight}
                    className="min-w-0 border-0 p-0 m-0"
                  >
                    <TabsContent value="structured">
                      {endpoint.conversational && bodyObj ? (
                        <div className="pg-structured">
                          <ParameterBar
                            endpoint={endpoint}
                            body={bodyObj}
                            maxTokensLimit={maxTokensLimit}
                            onUpdate={updateBodyObject}
                            disabled={inFlight}
                          />

                          <div className="pg-messages-header">
                            <span className="eyebrow text-[11px]">
                              {endpoint.kind === "openai-responses"
                                ? "INSTRUCTIONS & INPUT TURNS"
                                : "MESSAGES ARRAY"}
                            </span>
                            <div className="pg-message-add-btns">
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => handleAddMessage("user")}
                              >
                                <Plus aria-hidden="true" /> User turn
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => handleAddMessage("assistant")}
                              >
                                <Plus aria-hidden="true" /> Assistant turn
                              </Button>
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
                                    void handleAttachImages(
                                      idx,
                                      e.dataTransfer.files,
                                    );
                                  }
                                }}
                              >
                                <div className="pg-message-meta">
                                  <NativeSelect
                                    aria-label={`Turn ${idx + 1} role`}
                                    value={msg.role}
                                    onChange={(e) => {
                                      const role = e.target.value;
                                      if (
                                        role === "system" ||
                                        role === "user" ||
                                        role === "assistant" ||
                                        role === "tool"
                                      ) {
                                        handleMessageChange(idx, { role });
                                      }
                                    }}
                                  >
                                    <option value="system">system</option>
                                    <option value="user">user</option>
                                    <option value="assistant">assistant</option>
                                    {endpoint.kind === "openai-chat" && (
                                      <option value="tool">tool</option>
                                    )}
                                  </NativeSelect>
                                  <span className="pg-turn-index mono">
                                    #{idx}
                                  </span>
                                  {msg.tool_calls &&
                                    msg.tool_calls.length > 0 && (
                                      <Badge
                                        variant="secondary"
                                        className="text-violet-400"
                                      >
                                        {msg.tool_calls.length} tool_call
                                        {msg.tool_calls.length > 1 ? "s" : ""}
                                      </Badge>
                                    )}
                                  <div className="pg-turn-actions">
                                    {msg.role === "user" &&
                                      totalImages < MAX_IMAGES_PER_REQUEST && (
                                        <Button
                                          asChild
                                          variant="outline"
                                          size="sm"
                                          className="focus-within:ring-2 focus-within:ring-ring"
                                        >
                                          <label>
                                            <Plus aria-hidden="true" /> Image
                                            <input
                                              type="file"
                                              accept="image/png,image/jpeg,image/webp,image/gif"
                                              multiple
                                              aria-label={`Attach image to turn ${idx + 1}`}
                                              className="sr-only"
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
                                        </Button>
                                      )}
                                    <Button
                                      type="button"
                                      variant="ghost"
                                      size="icon-xs"
                                      aria-label={`Remove turn ${idx + 1}`}
                                      onClick={() => handleRemoveMessage(idx)}
                                    >
                                      <X aria-hidden="true" />
                                    </Button>
                                  </div>
                                </div>

                                {msg.images && msg.images.length > 0 && (
                                  <div className="pg-turn-images">
                                    {msg.images.map((imgUrl, imgIdx) => (
                                      <div
                                        key={imgIdx}
                                        className="pg-turn-thumb"
                                      >
                                        <img
                                          src={imgUrl}
                                          alt={`Turn ${idx + 1} attachment ${imgIdx + 1}`}
                                          className="pg-turn-thumb-img"
                                        />
                                        <span className="pg-turn-thumb-meta mono">
                                          {imgUrl.startsWith("data:image/jpeg")
                                            ? "JPEG"
                                            : imgUrl.startsWith(
                                                  "data:image/png",
                                                )
                                              ? "PNG"
                                              : "IMG"}{" "}
                                          #{imgIdx + 1}
                                        </span>
                                        <Button
                                          type="button"
                                          variant="ghost"
                                          size="icon-xs"
                                          className="pg-thumb-remove"
                                          aria-label={`Remove image ${imgIdx + 1} from turn ${idx + 1}`}
                                          onClick={() =>
                                            handleRemoveImage(idx, imgIdx)
                                          }
                                        >
                                          <X aria-hidden="true" />
                                        </Button>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                <Textarea
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
                                      ).filter((f) =>
                                        f.type.startsWith("image/"),
                                      );
                                      if (imgFiles.length > 0) {
                                        e.preventDefault();
                                        void handleAttachImages(idx, imgFiles);
                                      }
                                    }
                                  }}
                                  onKeyDown={(e) => {
                                    if (
                                      (e.metaKey || e.ctrlKey) &&
                                      e.key === "Enter"
                                    ) {
                                      e.preventDefault();
                                      void executeRequest();
                                    }
                                  }}
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <p className="muted">
                          Correct the request in JSON Body to use the structured
                          editor.
                        </p>
                      )}
                    </TabsContent>
                    <TabsContent
                      value="json"
                      {...(!endpoint.conversational
                        ? {
                            role: "region",
                            "aria-labelledby": "req-pane-title",
                          }
                        : {})}
                    >
                      {endpoint.method === "GET" ? (
                        <div className="pg-empty-body">
                          <code>GET {endpoint.path}</code> takes no request
                          body. Click <b>Send Request</b> to inspect the
                          response.
                        </div>
                      ) : (
                        <Field
                          className="pg-json-editor gap-2"
                          data-invalid={Boolean(jsonError)}
                        >
                          <FieldLabel
                            htmlFor="pg-json-body"
                            className="sr-only"
                          >
                            Request JSON body
                          </FieldLabel>
                          <Textarea
                            id="pg-json-body"
                            aria-invalid={Boolean(jsonError)}
                            aria-describedby={
                              jsonError
                                ? "pg-json-hint pg-json-error"
                                : "pg-json-hint"
                            }
                            className="mono"
                            rows={18}
                            value={rawJson}
                            onChange={(e) => handleRawJsonEdit(e.target.value)}
                            onKeyDown={(e) => {
                              if (
                                (e.metaKey || e.ctrlKey) &&
                                e.key === "Enter"
                              ) {
                                e.preventDefault();
                                void executeRequest();
                              }
                            }}
                          />
                          <FieldDescription id="pg-json-hint">
                            Enter a JSON object. Press Ctrl or Command + Enter
                            to send.
                          </FieldDescription>
                          {jsonError && (
                            <FieldError id="pg-json-error">
                              JSON syntax error: {jsonError}
                            </FieldError>
                          )}
                        </Field>
                      )}
                    </TabsContent>
                  </fieldset>
                </CardContent>
              </Tabs>
            </Card>

            <Card
              role="region"
              className="playground-pane gap-0 p-4"
              aria-labelledby="resp-pane-title"
            >
              <Tabs
                value={responseTab}
                onValueChange={(value) => {
                  if (
                    value === "parsed" ||
                    value === "raw" ||
                    value === "sse" ||
                    value === "curl"
                  )
                    setResponseTab(value);
                }}
              >
                <CardHeader className="pg-pane-heading flex flex-row flex-wrap items-center justify-between px-0">
                  <div className="pg-title-group">
                    <h2 id="resp-pane-title">Wire Response</h2>
                    {inFlight && (
                      <Badge variant="secondary" className="text-violet-400">
                        Streaming…
                      </Badge>
                    )}
                    {callMetrics.status !== null && (
                      <Badge
                        variant={
                          callMetrics.status >= 200 && callMetrics.status < 300
                            ? "secondary"
                            : "destructive"
                        }
                        className={
                          callMetrics.status >= 200 && callMetrics.status < 300
                            ? "text-teal-400"
                            : undefined
                        }
                      >
                        {callMetrics.status} {callMetrics.statusText}
                      </Badge>
                    )}
                  </div>
                  <TabsList aria-label="Response inspector view">
                    <TabsTrigger value="parsed">Parsed</TabsTrigger>
                    <TabsTrigger value="raw">Raw JSON</TabsTrigger>
                    <TabsTrigger value="sse">
                      SSE ({sseFrames.length})
                    </TabsTrigger>
                    <TabsTrigger value="curl">cURL</TabsTrigger>
                  </TabsList>
                </CardHeader>

                <CardContent className="px-0">
                  <div
                    className="pg-metrics-strip"
                    aria-label="Response telemetry"
                  >
                    <div>
                      <span className="eyebrow text-[11px]">TTFT</span>
                      <strong className="numeric">
                        {callMetrics.ttftMs !== null
                          ? `${callMetrics.ttftMs} ms`
                          : "—"}
                      </strong>
                    </div>
                    <div>
                      <span className="eyebrow text-[11px]">TOTAL E2E</span>
                      <strong className="numeric">
                        {callMetrics.e2eMs !== null
                          ? `${callMetrics.e2eMs} ms`
                          : "—"}
                      </strong>
                    </div>
                    <div>
                      <span className="eyebrow text-[11px]">PACE</span>
                      <strong className="numeric">
                        {decodeTokPerSec ? `${decodeTokPerSec} tok/s` : "—"}
                      </strong>
                    </div>
                    <div>
                      <span className="eyebrow text-[11px]">IN / CACHED</span>
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
                      <span className="eyebrow text-[11px]">OUTPUT</span>
                      <strong className="numeric">
                        {parsedResult.usage.outputTokens ?? "—"}
                      </strong>
                    </div>
                  </div>

                  {requestError && (
                    <Alert variant="destructive">
                      <AlertDescription>
                        <span>
                          <strong>Request failed:</strong> {requestError}
                        </span>
                      </AlertDescription>
                    </Alert>
                  )}

                  <TabsContent value="parsed">
                    <div className="pg-parsed-view">
                      {parsedResult.reasoning && (
                        <details className="pg-block pg-reasoning-block" open>
                          <summary className="eyebrow text-[11px]">
                            REASONING / THINKING STREAM
                            {parsedResult.usage.reasoningTokens !== null &&
                              ` (${parsedResult.usage.reasoningTokens} tok)`}
                          </summary>
                          <pre className="mono">{parsedResult.reasoning}</pre>
                        </details>
                      )}

                      {parsedResult.toolCalls.length > 0 && (
                        <div className="pg-block">
                          <span className="eyebrow text-[11px]">
                            TOOL CALLS
                          </span>
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
                          <span className="eyebrow text-[11px]">
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
                            <span className="eyebrow text-[11px]">
                              MULTI-TURN CONTINUATION
                            </span>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => handleAppendAndContinue(false)}
                            >
                              <Plus aria-hidden="true" /> Append assistant turn
                            </Button>
                          </div>
                          <div className="pg-continue-row">
                            <Input
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
                            <Button
                              type="button"
                              variant="default"
                              disabled={
                                followUpText.trim().length === 0 &&
                                parsedResult.toolCalls.length === 0
                              }
                              onClick={() => handleAppendAndContinue(true)}
                            >
                              <Send aria-hidden="true" /> Append & Send
                            </Button>
                          </div>
                        </div>
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="raw">
                    <pre
                      className="pg-code-box mono"
                      aria-label="Raw response payload"
                    >
                      {rawResponse || "No response payload captured yet."}
                    </pre>
                  </TabsContent>

                  <TabsContent value="sse">
                    <div className="pg-sse-list" aria-label="SSE frames">
                      {sseFrames.length === 0 ? (
                        <p className="muted">
                          No SSE frames recorded. Enable{" "}
                          <code>stream: true</code> and send a request.
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
                  </TabsContent>

                  <TabsContent value="curl">
                    <pre className="pg-code-box mono" aria-label="cURL command">
                      {curlCommand}
                    </pre>
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

function MethodBadge({ method }: { method: string }) {
  return (
    <Badge
      variant="outline"
      className={`font-mono text-[10px] ${method === "GET" ? "text-teal-400" : "text-violet-400"}`}
    >
      {method}
    </Badge>
  );
}

function ParameterBar({
  endpoint,
  body,
  maxTokensLimit,
  onUpdate,
  disabled,
}: {
  disabled: boolean;
  endpoint: EndpointSpec;
  body: Record<string, unknown>;
  maxTokensLimit: number;
  onUpdate: (
    updater: (b: Record<string, unknown>) => Record<string, unknown>,
  ) => void;
}) {
  const maxTokensKey =
    endpoint.kind === "openai-responses" ? "max_output_tokens" : "max_tokens";
  const maxTokensValue =
    typeof body[maxTokensKey] === "number"
      ? String(body[maxTokensKey] as number)
      : "";
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
      <Field className="w-auto gap-2">
        <FieldLabel htmlFor="pg-max-tokens" className="eyebrow text-[11px]">
          {maxTokensKey.toUpperCase()}
        </FieldLabel>
        <Input
          id="pg-max-tokens"
          className="w-24"
          type="number"
          min={minMaxTokens}
          max={maxTokensLimit}
          placeholder="auto"
          value={maxTokensValue}
          onChange={(e) => {
            const raw = e.target.value.trim();
            if (raw === "" && endpoint.kind !== "anthropic-messages") {
              onUpdate((curr) => {
                const next = { ...curr };
                delete next[maxTokensKey];
                return next;
              });
              return;
            }
            const val = Math.min(
              maxTokensLimit,
              Math.max(minMaxTokens, Number(raw) || minMaxTokens),
            );
            onUpdate((curr) => ({ ...curr, [maxTokensKey]: val }));
          }}
        />
      </Field>

      <Field className="w-auto gap-2">
        <FieldLabel htmlFor="pg-temperature" className="eyebrow text-[11px]">
          TEMPERATURE
        </FieldLabel>
        <Input
          id="pg-temperature"
          className="w-24"
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
      </Field>

      {endpoint.supportsStream && (
        <Field orientation="horizontal" className="w-auto gap-2 self-end pb-2">
          <Switch
            disabled={disabled}
            id="pg-stream"
            checked={stream}
            onCheckedChange={(checked) => {
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
          <FieldLabel htmlFor="pg-stream">Stream SSE</FieldLabel>
        </Field>
      )}

      {(endpoint.kind === "openai-chat" ||
        endpoint.kind === "anthropic-messages") && (
        <Field orientation="horizontal" className="w-auto gap-2 self-end pb-2">
          <Switch
            disabled={disabled}
            id="pg-thinking"
            checked={thinkingEnabled}
            onCheckedChange={(checked) => {
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
                    next.max_tokens = Math.min(
                      maxTokensLimit,
                      Math.max(currMax, 4096),
                    );
                  } else {
                    next.thinking = { type: "disabled" };
                  }
                }
                return next;
              });
            }}
          />
          <FieldLabel htmlFor="pg-thinking">Thinking</FieldLabel>
        </Field>
      )}
    </div>
  );
}
