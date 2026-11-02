export interface Histogram {
  bounds: number[];
  buckets: number[];
  count: number;
  sum: number;
}
export interface Pool {
  capacity: number;
  active: number | null;
  cached: number;
  free: number | null;
}
export interface Snapshot {
  schema_version: 1;
  instance_id: string;
  uptime_seconds: number;
  model: string;
  status: "ready" | "unhealthy" | "stopping";
  config: { tp_size: number; kv_format: string; speculation: string };
  requests: {
    submitted: number;
    completed: number;
    cancelled: number;
    failed: number;
    running: number;
    waiting: number;
  };
  tokens: {
    prompt: number;
    computed: number;
    cached: number;
    generated: number;
  };
  latency: {
    ttft: Histogram;
    queue: Histogram;
    e2e: Histogram;
    time_per_output_token: Histogram;
  };
  pools: { kv: Pool; gdn: Pool };
  speculation: {
    rounds: number;
    draft_tokens: number | null;
    accepted_tokens: number;
  };
  http_responses: {
    success: number;
    client_error: number;
    server_error: number;
  };
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const isNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0;
function hasNumbers(value: unknown, keys: string[]): boolean {
  return isRecord(value) && keys.every((key) => isNumber(value[key]));
}
function isPool(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasNumbers(value, ["capacity", "cached"]) &&
    (value.active === null || isNumber(value.active)) &&
    (value.free === null || isNumber(value.free))
  );
}
function isHistogram(value: unknown): boolean {
  if (!isRecord(value) || !hasNumbers(value, ["count", "sum"])) return false;
  const { bounds, buckets, count } = value;
  return (
    Array.isArray(bounds) &&
    Array.isArray(buckets) &&
    bounds.length > 0 &&
    bounds.length === buckets.length &&
    bounds.every((n, i) => isNumber(n) && (i === 0 || n > bounds[i - 1])) &&
    buckets.every(
      (n, i) =>
        isNumber(n) &&
        n <= (count as number) &&
        (i === 0 || n >= buckets[i - 1]),
    )
  );
}

/** Reject incompatible/partial payloads instead of displaying invented zeroes. */
export function parseSnapshot(value: unknown): Snapshot {
  if (!isRecord(value)) throw new Error("Invalid metrics response");
  const {
    config,
    requests,
    tokens,
    latency,
    pools,
    speculation,
    http_responses,
  } = value;
  const valid =
    value.schema_version === 1 &&
    typeof value.instance_id === "string" &&
    typeof value.model === "string" &&
    isNumber(value.uptime_seconds) &&
    ["ready", "unhealthy", "stopping"].includes(value.status as string) &&
    isRecord(config) &&
    isNumber(config.tp_size) &&
    typeof config.kv_format === "string" &&
    typeof config.speculation === "string" &&
    hasNumbers(requests, [
      "submitted",
      "completed",
      "cancelled",
      "failed",
      "running",
      "waiting",
    ]) &&
    hasNumbers(tokens, ["prompt", "computed", "cached", "generated"]) &&
    isRecord(latency) &&
    ["ttft", "queue", "e2e", "time_per_output_token"].every((key) =>
      isHistogram(latency[key]),
    ) &&
    isRecord(pools) &&
    ["kv", "gdn"].every((key) => isPool(pools[key])) &&
    hasNumbers(speculation, ["rounds", "accepted_tokens"]) &&
    isRecord(speculation) &&
    (speculation.draft_tokens === null || isNumber(speculation.draft_tokens)) &&
    hasNumbers(http_responses, ["success", "client_error", "server_error"]);
  if (!valid)
    throw new Error(
      "Unsupported metrics response; refresh after upgrading the server",
    );
  return value as unknown as Snapshot;
}

export const WINDOW_SECONDS = 300;
export const MAX_SAMPLES = 151;
export function appendSnapshot(
  history: Snapshot[],
  next: Snapshot,
): Snapshot[] {
  const previous = history.at(-1);
  // A new process (or a backward counter) must not create negative rates or bridge a restart.
  if (
    previous &&
    (previous.instance_id !== next.instance_id ||
      previous.uptime_seconds >= next.uptime_seconds ||
      previous.tokens.generated > next.tokens.generated)
  ) {
    return [next];
  }
  return [
    ...history.filter(
      (s) => next.uptime_seconds - s.uptime_seconds <= WINDOW_SECONDS,
    ),
    next,
  ].slice(-MAX_SAMPLES);
}

export function histogramDelta(
  before: Histogram,
  after: Histogram,
): Histogram | null {
  if (
    before.count > after.count ||
    before.bounds.length !== after.bounds.length ||
    before.bounds.some((bound, i) => bound !== after.bounds[i])
  )
    return null;
  const buckets = after.buckets.map((n, i) => n - before.buckets[i]);
  if (buckets.some((n, i) => n < 0 || (i > 0 && n < buckets[i - 1])))
    return null;
  return {
    bounds: after.bounds,
    buckets,
    count: after.count - before.count,
    sum: Math.max(0, after.sum - before.sum),
  };
}

export interface Quantile {
  value: number;
  lowerBound: boolean;
}
/** Linear bucket interpolation, like histogram_quantile; overflow is explicitly a lower bound. */
export function quantile(
  histogram: Histogram | null,
  q: number,
): Quantile | null {
  if (!histogram || histogram.count === 0) return null;
  const target = histogram.count * q;
  let lower = 0;
  let previousCount = 0;
  for (let i = 0; i < histogram.bounds.length; i++) {
    const count = histogram.buckets[i];
    const upper = histogram.bounds[i];
    if (count >= target && count > previousCount) {
      return {
        value:
          lower +
          ((target - previousCount) / (count - previousCount)) *
            (upper - lower),
        lowerBound: false,
      };
    }
    lower = upper;
    previousCount = count;
  }
  return { value: lower, lowerBound: true };
}

export function counterRate(
  history: Snapshot[],
  counter: (s: Snapshot) => number,
  seconds = 10,
): number | null {
  const last = history.at(-1);
  if (!last || history.length < 2) return null;
  const first =
    history.find((s) => last.uptime_seconds - s.uptime_seconds <= seconds) ??
    history[0];
  const elapsed = last.uptime_seconds - first.uptime_seconds;
  return elapsed > 0
    ? Math.max(0, counter(last) - counter(first)) / elapsed
    : null;
}

export function recentHistogram(
  history: Snapshot[],
  key: keyof Snapshot["latency"],
): Histogram | null {
  return history.length >= 2
    ? histogramDelta(
        history[0].latency[key],
        history[history.length - 1].latency[key],
      )
    : null;
}

export function ratio(numerator: number, denominator: number): number | null {
  return denominator > 0
    ? Math.min(1, Math.max(0, numerator / denominator))
    : null;
}

export function duration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  return seconds < 1
    ? `${(seconds * 1000).toFixed(seconds < 0.01 ? 1 : 0)} ms`
    : `${seconds.toFixed(2)} s`;
}
export function percentile(value: Quantile | null): string {
  return value
    ? `${value.lowerBound ? "≥ " : "≈ "}${duration(value.value)}`
    : "—";
}
export function percent(value: number | null): string {
  return value === null ? "—" : `${(100 * value).toFixed(1)}%`;
}
export function uptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400)
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}
