import { describe, expect, it } from "vitest";
import {
  appendSnapshot,
  counterRate,
  histogramDelta,
  parseSnapshot,
  percentile,
  quantile,
  ratio,
  recentHistogram,
  MAX_SAMPLES,
} from "./metrics";
import { histogram, snapshot } from "./fixtures";
import { chartPath } from "./Chart";

describe("shared metric derivations", () => {
  it("has no invented rates or percentiles before observations", () => {
    const samples = [snapshot()];
    expect(counterRate(samples, (s) => s.tokens.generated)).toBeNull();
    expect(quantile(histogram(), 0.95)).toBeNull();
    expect(recentHistogram(samples, "ttft")).toBeNull();
    expect(ratio(0, 0)).toBeNull();
  });
  it("uses engine elapsed time and accepted output counters for throughput", () => {
    const first = snapshot();
    const last = snapshot({
      uptime_seconds: 102,
      tokens: { ...first.tokens, generated: 120 },
    });
    expect(counterRate([first, last], (s) => s.tokens.generated)).toBe(60);
  });
  it("resets history on process restart or backward counters", () => {
    const first = snapshot({
      tokens: { prompt: 0, cached: 0, computed: 0, generated: 10 },
    });
    const restarted = snapshot({ instance_id: "new" });
    expect(appendSnapshot([first], restarted)).toEqual([restarted]);
    expect(
      appendSnapshot([first], snapshot({ uptime_seconds: 102 })),
    ).toHaveLength(1);
  });
  it("bounds both time and sample count", () => {
    let history = [snapshot()];
    for (let i = 1; i <= 1000; i++)
      history = appendSnapshot(history, snapshot({ uptime_seconds: 100 + i }));
    expect(history).toHaveLength(MAX_SAMPLES);
    expect(
      appendSnapshot(history, snapshot({ uptime_seconds: 2000 })),
    ).toHaveLength(1);
  });
  it("derives histogram intervals from cumulative buckets", () => {
    const delta = histogramDelta(histogram(4), histogram(10));
    expect(delta?.count).toBe(6);
    expect(delta?.buckets).toEqual([0, 6, 6, 6]);
    expect(quantile(delta, 0.5)?.value).toBeCloseTo(0.055);
    expect(histogramDelta(histogram(10), histogram(4))).toBeNull();
  });
  it("labels overflow percentiles as lower bounds, not fabricated exact values", () => {
    const overflow = { bounds: [0.1, 1], buckets: [0, 0], count: 1, sum: 5 };
    expect(percentile(quantile(overflow, 0.95))).toBe("≥ 1.00 s");
  });
  it("rejects partial/incompatible metrics and corrupt histogram data", () => {
    expect(parseSnapshot(snapshot())).toEqual(snapshot());
    expect(() => parseSnapshot({ schema_version: 2 })).toThrow();
    const invalid = snapshot();
    invalid.latency.ttft.buckets = [2, 1, 0, 0];
    expect(() => parseSnapshot(invalid)).toThrow();
    expect(() =>
      parseSnapshot({ ...snapshot(), uptime_seconds: NaN }),
    ).toThrow();
  });
  it("accepts explicitly unavailable pool and draft measurements without inventing values", () => {
    const unknown = snapshot();
    unknown.pools.gdn.active = null;
    unknown.pools.gdn.free = null;
    unknown.speculation.draft_tokens = null;
    expect(parseSnapshot(unknown).speculation.draft_tokens).toBeNull();
  });
  it("does not draw across unavailable samples or disconnected intervals", () => {
    const path = chartPath(
      [
        { time: 100, value: 1 },
        { time: 102, value: 2 },
        { time: 104, value: null },
        { time: 106, value: 1 },
        { time: 140, value: 1 },
      ],
      140,
      2,
      800,
      200,
    );
    expect(path.match(/M/g)).toHaveLength(3);
    expect(path.match(/L/g)).toHaveLength(1);
    expect(path).not.toContain("NaN");
  });
});
