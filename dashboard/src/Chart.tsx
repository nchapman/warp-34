import { useId } from "react";

export interface Point {
  time: number;
  value: number | null;
}
interface ChartProps {
  points: Point[];
  label: string;
  unit: string;
  color?: "violet" | "aqua";
  compact?: boolean;
}

/** Break the trace across missing samples rather than implying continuous observation. */
export function chartPath(
  points: Point[],
  end: number,
  max: number,
  width: number,
  height: number,
): string {
  let previous: Point | null = null;
  return points
    .map((point) => {
      if (point.value === null) {
        previous = null;
        return "";
      }
      const x = width * (1 - (end - point.time) / 300);
      const y = height * (1 - point.value / max);
      const command = previous && point.time - previous.time <= 8 ? "L" : "M";
      previous = point;
      return `${command}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

/** Closed area polygons per contiguous observation segment; never bridges missing samples. */
export function chartAreaPath(
  points: Point[],
  end: number,
  max: number,
  width: number,
  height: number,
): string {
  const segments: Array<Array<{ x: number; y: number }>> = [];
  let previous: Point | null = null;
  for (const point of points) {
    if (point.value === null) {
      previous = null;
      continue;
    }
    const x = width * (1 - (end - point.time) / 300);
    const y = height * (1 - point.value / max);
    if (previous && point.time - previous.time <= 8) {
      segments.at(-1)!.push({ x, y });
    } else {
      segments.push([{ x, y }]);
    }
    previous = point;
  }
  return segments
    .filter((seg) => seg.length >= 2)
    .map((seg) => {
      const first = seg[0];
      const last = seg.at(-1)!;
      const top = seg
        .map((pt) => `L${pt.x.toFixed(2)},${pt.y.toFixed(2)}`)
        .join(" ");
      return `M${first.x.toFixed(2)},${height.toFixed(2)} ${top} L${last.x.toFixed(2)},${height.toFixed(2)} Z`;
    })
    .join(" ");
}

export function Chart({
  points,
  label,
  unit,
  color = "violet",
  compact = false,
}: ChartProps) {
  const titleId = useId();
  const gradId = `${titleId}-area`;
  const width = 800;
  const height = compact ? 140 : 230;
  const values = points.filter((p) => p.value !== null);
  const peak = Math.max(0, ...values.map((p) => p.value ?? 0));
  const magnitude = 10 ** Math.floor(Math.log10(peak || 1));
  const max = Math.max(1, Math.ceil(peak / magnitude) * magnitude);
  const end = points.at(-1)?.time ?? 0;
  const path = chartPath(points, end, max, width, height);
  const area = chartAreaPath(points, end, max, width, height);
  const tick = (n: number) =>
    n >= 1000
      ? `${(n / 1000).toFixed(1)}k`
      : n.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return (
    <div className={`chart ${compact ? "compact" : ""}`}>
      <div className="chart-y" aria-hidden="true">
        <span>{tick(max)}</span>
        <span>{tick(max / 2)}</span>
        <span>0</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-labelledby={titleId}
      >
        <title id={titleId}>
          {label}.{" "}
          {values.length < 2
            ? "Collecting observations."
            : `Latest ${tick(values.at(-1)?.value ?? 0)} ${unit}; peak ${tick(peak)} ${unit} in the observed window.`}
        </title>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              className={`chart-area-stop ${color}`}
              stopOpacity="0.22"
            />
            <stop
              offset="100%"
              className={`chart-area-stop ${color}`}
              stopOpacity="0"
            />
          </linearGradient>
        </defs>
        {[0, 0.5, 1].map((fraction) => (
          <line
            key={fraction}
            x1="0"
            x2={width}
            y1={fraction * height}
            y2={fraction * height}
            className="chart-grid"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {area && <path d={area} fill={`url(#${gradId})`} />}
        <path
          d={path}
          fill="none"
          className={`chart-line ${color}`}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      {values.length < 2 && (
        <div className="chart-empty">
          Collecting observations<span>Live history appears here</span>
        </div>
      )}
      <div className="chart-x" aria-hidden="true">
        <span>5 minutes ago</span>
        <span>Now</span>
      </div>
    </div>
  );
}
