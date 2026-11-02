import { useEffect, useState } from "react";
import { appendSnapshot, parseSnapshot, type Snapshot } from "./metrics";

export interface MetricsState {
  history: Snapshot[];
  connection: "connecting" | "live" | "offline" | "paused";
  error: string | null;
}

export function useMetrics(): MetricsState {
  const [state, setState] = useState<MetricsState>({
    history: [],
    connection: "connecting",
    error: null,
  });
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let request: AbortController | null = null;

    async function poll() {
      if (disposed || document.hidden) return;
      const controller = new AbortController();
      request = controller;
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch("/api/metrics", {
          signal: controller.signal,
          cache: "no-store",
          credentials: "same-origin",
          headers: { Accept: "application/json" },
        });
        if (!response.ok)
          throw new Error(`Metrics endpoint returned HTTP ${response.status}`);
        const snapshot = parseSnapshot(await response.json());
        if (!disposed && request === controller)
          setState((previous) => ({
            history: appendSnapshot(previous.history, snapshot),
            connection: "live",
            error: null,
          }));
      } catch (error) {
        if (!disposed && request === controller)
          setState((previous) => ({
            ...previous,
            connection: "offline",
            error:
              error instanceof Error && error.name !== "AbortError"
                ? error.message
                : "Metrics endpoint did not respond in time",
          }));
      } finally {
        clearTimeout(timeout);
        if (!disposed && request === controller) {
          request = null;
          timer = setTimeout(() => void poll(), 2000);
        }
      }
    }

    function visibilityChanged() {
      clearTimeout(timer);
      const previous = request;
      request = null;
      previous?.abort();
      setState((old) => ({
        ...old,
        connection: document.hidden ? "paused" : "connecting",
      }));
      if (!document.hidden) void poll();
    }
    document.addEventListener("visibilitychange", visibilityChanged);
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
      request?.abort();
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, []);
  return state;
}
