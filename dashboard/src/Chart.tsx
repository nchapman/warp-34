import { useId } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { WINDOW_SECONDS } from "./metrics";

export interface Point {
  time: number;
  value: number | null;
}
export interface ChartProps {
  points: Point[];
  label: string;
  unit: string;
  color?: "violet" | "aqua";
  compact?: boolean;
}

/** Input is chronological engine uptime, not wall time. Separators are missing
 * observations, never interpolated measurements. Invalid values also break traces. */
export function normalizeChartPoints(points: Point[]): Point[] {
  const validTimes = points.filter((point) => Number.isFinite(point.time));
  const end = validTimes.at(-1)?.time ?? 0;
  const result: Point[] = [];
  for (const point of validTimes) {
    if (point.time < end - WINDOW_SECONDS) continue;
    const previous = result.at(-1);
    if (previous && point.time - previous.time > 8) {
      result.push({
        time: previous.time + (point.time - previous.time) / 2,
        value: null,
      });
    }
    result.push({
      time: point.time,
      value:
        point.value !== null && Number.isFinite(point.value)
          ? point.value
          : null,
    });
  }
  return result;
}

const tick = (n: number) =>
  n >= 1000
    ? `${(n / 1000).toFixed(1)}k`
    : n.toLocaleString(undefined, { maximumFractionDigits: 1 });

export function Chart({
  points,
  label,
  unit,
  color = "violet",
  compact = false,
}: ChartProps) {
  const summaryId = useId();
  const data = normalizeChartPoints(points);
  const values = data.flatMap((p) => (p.value === null ? [] : [p.value]));
  const peak = Math.max(0, ...values);
  const magnitude = 10 ** Math.floor(Math.log10(peak || 1));
  const max = Math.max(1, Math.ceil(peak / magnitude) * magnitude);
  const end = data.at(-1)?.time ?? 0;
  const latest = data.at(-1)?.value;
  const collecting = values.length < 2;
  const summary = `${label}. ${collecting ? "Collecting observations. " : ""}${latest == null ? "Latest observation unavailable." : `Latest ${tick(latest)} ${unit}.`}${values.length ? ` Peak ${tick(peak)} ${unit} in the observed window.` : ""}`;
  const height = compact ? 130 : 200;
  return (
    <div className="relative w-full min-w-0">
      {/* A sibling summary preserves existing image semantics without hiding the
          Recharts keyboard-accessible application inside an image role. */}
      <div role="img" aria-label={summary} id={summaryId} className="sr-only" />
      <ChartContainer
        config={{
          value: {
            label,
            color: color === "aqua" ? "var(--chart-2)" : "var(--chart-1)",
          },
        }}
        className="w-full aspect-auto"
        style={{ height, width: "100%", minWidth: 0 }}
        initialDimension={{ width: 800, height }}
      >
        <LineChart
          data={data}
          accessibilityLayer
          aria-label={`${label}: use arrow keys to explore observations`}
          aria-describedby={summaryId}
          margin={{ top: 8, right: 16, bottom: 0, left: 0 }}
        >
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="time"
            type="number"
            domain={[end - WINDOW_SECONDS, end]}
            allowDataOverflow
            ticks={[end - WINDOW_SECONDS, end]}
            tickFormatter={(time: number) =>
              time === end ? "Now" : "5 minutes ago"
            }
            tickLine={false}
            axisLine={false}
            height={28}
            interval={0}
          />
          <YAxis
            domain={[0, max]}
            ticks={[0, max / 2, max]}
            tickFormatter={tick}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                labelFormatter={(_label, payload) => {
                  const time = payload[0]?.payload?.time as number | undefined;
                  return time === undefined
                    ? ""
                    : `Engine uptime ${tick(time)} s (${tick(end - time)} s ago)`;
                }}
                formatter={(value) => (
                  <span>
                    {label}:{" "}
                    {typeof value === "number" ? tick(value) : String(value)}{" "}
                    {unit}
                  </span>
                )}
              />
            }
          />
          <Line
            dataKey="value"
            type="linear"
            stroke="var(--color-value)"
            strokeWidth={2}
            connectNulls={false}
            dot={{ r: 2, strokeWidth: 0, fill: "var(--color-value)" }}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ChartContainer>
      {collecting && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-sm text-muted-foreground">
          Collecting observations
          <span className="text-xs">Live history appears here</span>
        </div>
      )}
    </div>
  );
}
