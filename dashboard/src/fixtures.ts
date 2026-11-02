import type { Histogram, Snapshot } from "./metrics";

// Test-only fixtures; never imported by the dashboard entrypoint.
export const histogram = (count = 0): Histogram => ({
  bounds: [0.01, 0.1, 1, 10],
  buckets: [0, count, count, count],
  count,
  sum: count * 0.05,
});
export function snapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    schema_version: 1,
    instance_id: "test-instance",
    uptime_seconds: 100,
    model: "test/model",
    status: "ready",
    config: { tp_size: 1, kv_format: "tq4", speculation: "none" },
    requests: {
      submitted: 0,
      completed: 0,
      cancelled: 0,
      failed: 0,
      running: 0,
      waiting: 0,
    },
    tokens: { prompt: 0, computed: 0, cached: 0, generated: 0 },
    latency: {
      ttft: histogram(),
      queue: histogram(),
      e2e: histogram(),
      time_per_output_token: histogram(),
    },
    pools: {
      kv: { capacity: 100, active: 10, cached: 20, free: 70 },
      gdn: { capacity: 10, active: 1, cached: 2, free: 7 },
    },
    speculation: { rounds: 0, draft_tokens: 0, accepted_tokens: 0 },
    http_responses: { success: 0, client_error: 0, server_error: 0 },
    ...overrides,
  };
}
