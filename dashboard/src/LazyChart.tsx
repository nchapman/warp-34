import { Component, lazy, Suspense, type ReactNode } from "react";
import type { ChartProps } from "./Chart";

export type { Point } from "./Chart";

// Connect and Playground do not pay the Recharts download/parse cost.
const TelemetryChart = lazy(() =>
  import("./Chart").then((module) => ({ default: module.Chart })),
);

class ChartBoundary extends Component<
  { label: string; compact?: boolean; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <div
          className="flex flex-col items-center justify-center gap-1 text-sm text-muted-foreground"
          style={{ height: this.props.compact ? 130 : 200 }}
          aria-label={`${this.props.label}: chart unavailable`}
          role="status"
        >
          <span>Chart unavailable</span>
          <span className="text-xs">Reload the page to try again.</span>
        </div>
      );
    }
    return this.props.children;
  }
}

export function Chart(props: ChartProps) {
  return (
    <ChartBoundary label={props.label} compact={props.compact}>
      <Suspense
        fallback={
          <div
            className="flex items-center justify-center text-sm text-muted-foreground"
            style={{ height: props.compact ? 130 : 200 }}
            aria-label={`${props.label}: loading chart`}
            aria-busy="true"
          >
            Loading chart…
          </div>
        }
      >
        <TelemetryChart {...props} />
      </Suspense>
    </ChartBoundary>
  );
}
