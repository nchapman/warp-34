import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useMetrics } from "./useMetrics";
import { snapshot } from "./fixtures";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("metrics polling", () => {
  it("loads the actual endpoint and cancels pending work on unmount", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => snapshot() });
    vi.stubGlobal("fetch", fetchMock);
    const { result, unmount } = renderHook(useMetrics);
    await waitFor(() => expect(result.current.connection).toBe("live"));
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/metrics",
      expect.objectContaining({ cache: "no-store" }),
    );
    expect(result.current.history).toHaveLength(1);
    unmount();
  });
  it("marks malformed or failed responses offline rather than zeroing metrics", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ schema_version: 99 }),
      }),
    );
    const { result } = renderHook(useMetrics);
    await waitFor(() => expect(result.current.connection).toBe("offline"));
    expect(result.current.history).toHaveLength(0);
    expect(result.current.error).toMatch(/Unsupported metrics/);
  });
  it("retries failures and recovers without overlapping polls", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValue({ ok: true, json: async () => snapshot() });
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(useMetrics);
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.connection).toBe("offline");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(result.current.connection).toBe("live");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
