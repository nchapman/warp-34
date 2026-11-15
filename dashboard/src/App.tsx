import { useEffect } from "react";
import {
  BrowserRouter,
  Link,
  MemoryRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router";
import { Chart, type Point } from "./Chart";
import { Playground } from "./Playground";
import {
  counterRate,
  duration,
  histogramDelta,
  percent,
  percentile,
  quantile,
  ratio,
  recentHistogram,
  uptime,
  type Pool,
  type Snapshot,
} from "./metrics";
import { useMetrics, type MetricsState } from "./useMetrics";

const number = (value: number | null | undefined) =>
  value === undefined || value === null
    ? "—"
    : value.toLocaleString(undefined, { maximumFractionDigits: 0 });

function PoolBar({
  name,
  pool,
  unit,
}: {
  name: string;
  pool: Pool | undefined;
  unit: string;
}) {
  const available =
    pool && pool.capacity > 0 && pool.active !== null && pool.free !== null;
  const active = available ? (100 * (pool.active ?? 0)) / pool.capacity : 0;
  const cached = available ? (100 * pool.cached) / pool.capacity : 0;
  return (
    <div className="pool">
      <div className="pool-heading">
        <span>{name}</span>
        <span className="numeric">
          {available ? `${active.toFixed(1)}% active` : "Unavailable"}
        </span>
      </div>
      <div
        className="pool-track"
        role="img"
        aria-label={
          available
            ? `${name}: ${pool.active} active, ${pool.cached} evictable, ${pool.free} free of ${pool.capacity} ${unit}`
            : `${name} capacity unavailable`
        }
      >
        <span className="pool-active" style={{ width: `${active}%` }} />
        <span className="pool-cached" style={{ width: `${cached}%` }} />
      </div>
      <div className="pool-detail">
        <span>{number(pool?.free)} free</span>
        <span>
          {number(pool?.capacity)} {unit}
        </span>
      </div>
    </div>
  );
}

function throughputPoints(history: Snapshot[], decisionMode = false): Point[] {
  return history.map((s, i) => ({
    time: s.uptime_seconds,
    value: counterRate(history.slice(Math.max(0, i - 5), i + 1), (sample) =>
      decisionMode ? sample.tokens.computed : sample.tokens.generated,
    ),
  }));
}
function latencyPoints(history: Snapshot[]): Point[] {
  return history
    .map((s, i) => {
      const first =
        history.find(
          (sample) => sample.uptime_seconds >= s.uptime_seconds - 30,
        ) ?? history[0];
      const ttftHist =
        i === 0 ? null : histogramDelta(first.latency.ttft, s.latency.ttft);
      const histogram =
        ttftHist && ttftHist.count > 0
          ? ttftHist
          : i === 0
            ? null
            : histogramDelta(first.latency.e2e, s.latency.e2e);
      return {
        time: s.uptime_seconds,
        value: (quantile(histogram, 0.95)?.value ?? NaN) * 1000,
      };
    })
    .map((point) => ({
      ...point,
      value: Number.isFinite(point.value) ? point.value : null,
    }));
}

function Status({ state }: { state: MetricsState }) {
  const snapshot = state.history.at(-1);
  let label = "Connecting";
  let style = "muted";
  if (state.connection === "offline") {
    label = "Disconnected";
    style = "danger";
  } else if (state.connection === "paused") label = "Paused";
  else if (state.connection === "live") {
    label =
      snapshot?.status === "ready"
        ? "Engine ready"
        : snapshot?.status === "stopping"
          ? "Stopping"
          : "Engine unhealthy";
    style = snapshot?.status === "ready" ? "healthy" : "danger";
  }
  return (
    <span className={`status ${style}`} role="status">
      <span className="status-dot" />
      {label}
    </span>
  );
}

function Overview({ state }: { state: MetricsState }) {
  const { history, connection } = state;
  const latest = history.at(-1);
  const first = history[0];
  const fresh = connection === "live";
  const ttftRaw = recentHistogram(history, "ttft");
  const e2eRaw = recentHistogram(history, "e2e");
  const isDecisionWorkload = Boolean(
    latest &&
    first &&
    history.length > 1 &&
    latest.tokens.generated === first.tokens.generated &&
    (latest.tokens.computed > first.tokens.computed ||
      (e2eRaw && e2eRaw.count > 0 && (!ttftRaw || ttftRaw.count === 0))),
  );
  const genRate = fresh
    ? counterRate(history, (s) => s.tokens.generated)
    : null;
  const promptRate = fresh
    ? counterRate(history, (s) => s.tokens.computed)
    : null;
  const rate = isDecisionWorkload ? promptRate : genRate;
  const ttft = ttftRaw && ttftRaw.count > 0 ? ttftRaw : e2eRaw;
  const queue = recentHistogram(history, "queue");
  const pacing = recentHistogram(history, "time_per_output_token");
  const cached =
    latest && first && history.length > 1
      ? ratio(
          latest.tokens.cached - first.tokens.cached,
          latest.tokens.cached -
            first.tokens.cached +
            latest.tokens.computed -
            first.tokens.computed,
        )
      : null;
  const speculation = latest?.speculation;
  const specEnabled =
    latest &&
    !["none", "off", "disabled"].includes(
      latest.config.speculation.toLowerCase(),
    );
  return (
    <>
      <div className="section-heading">
        <div>
          <h1>Overview</h1>
          <p>Your inference engine, at a glance.</p>
        </div>
        <span className="window-label">
          <span className="tiny-clock" aria-hidden="true">
            ◷
          </span>{" "}
          Last 5 minutes · local history
        </span>
      </div>
      {connection === "offline" && (
        <div className="notice" role="alert">
          <strong>Connection lost.</strong> {state.error}.{" "}
          {latest
            ? "Showing the last received snapshot; "
            : "No metrics received yet; "}
          retrying automatically.
        </div>
      )}
      {latest && latest.status !== "ready" && (
        <div className="notice" role="alert">
          {latest.status === "stopping"
            ? "The engine is shutting down."
            : "The inference worker is unhealthy."}{" "}
          Metrics remain available for diagnosis.
        </div>
      )}
      <section className="engine-strip" aria-label="Engine configuration">
        <div className="model-icon" aria-hidden="true">
          ◇
        </div>
        <div className="model-info">
          <span className="eyebrow">LOADED MODEL</span>
          <span className="model-name" title={latest?.model}>
            {latest?.model ?? "Waiting for engine…"}
          </span>
        </div>
        <div className="model-tags">
          <span>TP {latest?.config.tp_size ?? "—"}</span>
          <span>{latest?.config.kv_format.toUpperCase() ?? "—"} KV</span>
          {specEnabled && <span>{latest.config.speculation}</span>}
        </div>
      </section>
      <div className="primary-grid">
        <section
          className="panel throughput-panel"
          aria-labelledby="throughput-title"
        >
          <div className="panel-heading">
            <h2 id="throughput-title">Generation throughput</h2>
            <span className="legend">
              <i className="violet-dot" />
              {isDecisionWorkload ? "Prompt tokens" : "Output tokens"}
            </span>
          </div>
          <div className="hero-number numeric">
            {rate === null ? "—" : number(rate)}
            <span>tok/s</span>
          </div>
          <p className="metric-caption">
            {isDecisionWorkload
              ? "Aggregate prompt evaluation · trailing 10 seconds"
              : "Aggregate engine output · trailing 10 seconds"}
          </p>
          <Chart
            points={throughputPoints(history, isDecisionWorkload)}
            label="Generation throughput"
            unit="tokens per second"
          />
        </section>
        <section
          className="panel requests-panel"
          aria-labelledby="requests-title"
        >
          <div className="panel-heading">
            <h2 id="requests-title">Requests</h2>
            <span className="muted">Current</span>
          </div>
          <div className="request-numbers">
            <div>
              <strong className="numeric">
                {number(latest?.requests.running)}
              </strong>
              <span>
                <i className="aqua-dot" />
                Running
              </span>
            </div>
            <div>
              <strong className="numeric">
                {number(latest?.requests.waiting)}
              </strong>
              <span>
                <i className="indigo-dot" />
                Queued
              </span>
            </div>
          </div>
          <Chart
            points={history.map((s) => ({
              time: s.uptime_seconds,
              value: s.requests.running + s.requests.waiting,
            }))}
            label="Running and queued requests"
            unit="requests"
            color="aqua"
            compact
          />
          <div className="request-totals">
            <div>
              <span>Completed</span>
              <strong>{number(latest?.requests.completed)}</strong>
            </div>
            <div>
              <span>Cancelled</span>
              <strong>{number(latest?.requests.cancelled)}</strong>
            </div>
            <div>
              <span>Failed</span>
              <strong className={latest?.requests.failed ? "error-text" : ""}>
                {number(latest?.requests.failed)}
              </strong>
            </div>
          </div>
          <p className="footnote">Engine requests · totals since startup</p>
        </section>
      </div>
      <div className="secondary-grid">
        <section className="panel" aria-labelledby="latency-title">
          <div className="panel-heading">
            <h2 id="latency-title">Time to first token</h2>
            <span className="muted">Engine</span>
          </div>
          <div className="latency-values">
            <div>
              <strong className="numeric">
                {percentile(quantile(ttft, 0.5))}
              </strong>
              <span>p50</span>
            </div>
            <div>
              <strong className="numeric">
                {percentile(quantile(ttft, 0.95))}
              </strong>
              <span>p95</span>
            </div>
          </div>
          <Chart
            points={latencyPoints(history)}
            label="Estimated p95 engine time to first token, trailing 30 seconds"
            unit="milliseconds"
            compact
          />
          <div className="detail-row">
            <span>Queue p95</span>
            <span>{percentile(quantile(queue, 0.95))}</span>
          </div>
          <div className="detail-row">
            <span>
              {pacing && pacing.count > 0
                ? "Time / output token"
                : isDecisionWorkload
                  ? "Mean decision E2E"
                  : "Time / output token"}
            </span>
            <span>
              {duration(
                pacing && pacing.count > 0
                  ? pacing.sum / pacing.count
                  : isDecisionWorkload && e2eRaw && e2eRaw.count > 0
                    ? e2eRaw.sum / e2eRaw.count
                    : null,
              )}
            </span>
          </div>
          <p className="footnote">
            Bucket estimates · excludes HTTP preprocessing.
            <br />
            Token pacing is a completed-request average, not SSE timing.
          </p>
        </section>
        <section className="panel" aria-labelledby="cache-title">
          <div className="panel-heading">
            <h2 id="cache-title">Prefix reuse</h2>
            <span className="muted">Observed window</span>
          </div>
          <div className="secondary-number numeric">{percent(cached)}</div>
          <p className="metric-caption">Prompt tokens restored from cache</p>
          <div
            className="cache-track"
            role="img"
            aria-label={
              cached === null
                ? "No prompt observations in this window"
                : `${percent(cached)} of prompt tokens reused`
            }
          >
            <span style={{ width: `${(cached ?? 0) * 100}%` }} />
          </div>
          <div className="detail-row">
            <span>Cached tokens</span>
            <span>{number(latest?.tokens.cached)}</span>
          </div>
          <div className="detail-row">
            <span>Computed prompt tokens</span>
            <span>{number(latest?.tokens.computed)}</span>
          </div>
          {specEnabled && speculation && (
            <div className="speculation">
              <span className="eyebrow">SPECULATIVE DECODING</span>
              <div className="detail-row">
                <span>
                  {speculation.draft_tokens === null
                    ? "Verify rounds"
                    : "Draft acceptance"}
                </span>
                <span>
                  {speculation.draft_tokens === null
                    ? number(speculation.rounds)
                    : percent(
                        ratio(
                          speculation.accepted_tokens,
                          speculation.draft_tokens,
                        ),
                      )}
                </span>
              </div>
              <div className="detail-row">
                <span>Tokens / verify round</span>
                <span>
                  {speculation.rounds
                    ? (
                        1 +
                        speculation.accepted_tokens / speculation.rounds
                      ).toFixed(2)
                    : "—"}
                </span>
              </div>
            </div>
          )}
          <p className="footnote">
            Token totals and speculation since startup.
            <br />
            Draft proposals are not counted as output.
          </p>
        </section>
        <section className="panel" aria-labelledby="capacity-title">
          <div className="panel-heading">
            <h2 id="capacity-title">Cache capacity</h2>
            <span className="muted">Target pools</span>
          </div>
          <PoolBar name="KV cache" pool={latest?.pools.kv} unit="pages" />
          <PoolBar name="GDN state" pool={latest?.pools.gdn} unit="slots" />
          <div className="capacity-legend">
            <span>
              <i className="violet-dot" />
              Active / reserved
            </span>
            <span>
              <i className="indigo-dot" />
              Evictable
            </span>
            <span>
              <i className="free-dot" />
              Free
            </span>
          </div>
          <p className="footnote">
            Cached resources can be reclaimed for new requests. High cache
            occupancy alone is not pressure.
          </p>
        </section>
      </div>
      <div className="bottom-line">
        <span>
          HTTP responses <b>{number(latest?.http_responses.client_error)}</b>{" "}
          client errors <span className="separator">/</span>{" "}
          <b>{number(latest?.http_responses.server_error)}</b> server errors{" "}
          <span className="muted">· since startup</span>
        </span>
        <a href="/metrics" target="_blank" rel="noreferrer">
          Prometheus metrics <span aria-hidden="true">↗</span>
        </a>
      </div>
    </>
  );
}

function ConnectionGuide() {
  const base = window.location.origin;
  return (
    <>
      <div className="section-heading">
        <div>
          <h1>Connect</h1>
          <p>One engine. Your tools.</p>
        </div>
      </div>
      <section className="panel guide">
        <span className="eyebrow">OPENAI-COMPATIBLE API</span>
        <h2>Bring your coding agent.</h2>
        <p>
          Point an OpenAI-compatible client at this server’s base URL and use
          the loaded model identifier shown in Overview.
        </p>
        <label>Base URL</label>
        <code>{base}/v1</code>
        <p>
          Native Responses and Anthropic Messages endpoints are also available.
          Model capabilities and client configuration vary; this page does not
          change engine settings.
        </p>
      </section>
      <section className="panel guide">
        <span className="eyebrow">OBSERVABILITY</span>
        <h2>Use your own monitoring.</h2>
        <p>
          Scrape <a href="/metrics">/metrics</a> with Prometheus, or consume the
          same counters and histogram buckets as JSON at{" "}
          <a href="/api/metrics">/api/metrics</a>. This dashboard keeps up to
          five minutes of history in this tab; use an external collector for
          persistence.
        </p>
        <pre>{`scrape_configs:\n  - job_name: warp34\n    scrape_interval: 10s\n    static_configs:\n      - targets: ['${window.location.host}']`}</pre>
      </section>
      <section className="guide-security">
        <h2>Keep your endpoint private.</h2>
        <p>
          API-key authentication is optional. When enabled, configure your
          client with the deployment’s key (Bearer authorization or x-api-key).
          The dashboard, metrics, and health endpoints remain public. Bind
          locally or use an authenticated TLS reverse proxy to protect the whole
          service. The dashboard is read-only and contains no prompt or response
          content.
        </p>
      </section>
    </>
  );
}

function titleFromPathname(pathname: string): string {
  if (pathname.startsWith("/playground")) return "warp34 · Playground";
  if (pathname.startsWith("/connect")) return "warp34 · Connect";
  return "warp34 · Overview";
}

function DashboardShell({ state }: { state: MetricsState }) {
  const location = useLocation();
  const latest = state.history.at(-1);
  const isPlayground = location.pathname.startsWith("/playground");

  useEffect(() => {
    document.title = titleFromPathname(location.pathname);
  }, [location.pathname]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <Link className="brand" to="/" aria-label="warp34 home">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <path className="warp-bar-1" d="M4.5 23 9.5 9h4l-5 14z" />
            <path className="warp-bar-2" d="M11.5 23 16.5 9h4l-5 14z" />
            <path className="warp-bar-3" d="M18.5 23 23.5 9h4l-5 14z" />
          </svg>
          <span>
            warp<span className="brand-suffix">34</span>
          </span>
        </Link>
        <div className="topbar-right">
          <span className="uptime">
            Uptime <b>{latest ? uptime(latest.uptime_seconds) : "—"}</b>
          </span>
          <Status state={state} />
        </div>
      </header>
      <nav aria-label="Main navigation">
        <NavLink to="/" end>
          Overview
        </NavLink>
        <NavLink to="/playground">Playground</NavLink>
        <NavLink to="/connect">Connect</NavLink>
        <span className="nav-label">INFERENCE CONSOLE</span>
      </nav>
      <main id="main">
        <Routes>
          <Route path="/" element={<Overview state={state} />} />
          <Route
            path="/playground"
            element={<Playground model={latest?.model ?? null} />}
          />
          <Route
            path="/playground/:endpointId"
            element={<Playground model={latest?.model ?? null} />}
          />
          <Route path="/connect" element={<ConnectionGuide />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer>
        <span>
          warp34 <span className="separator">/</span> Built for real work.
        </span>
        <span>
          {state.connection === "live"
            ? "Updates every 2s"
            : state.connection === "offline"
              ? "Waiting to reconnect"
              : "Connecting to engine"}
          <span className="separator">·</span>
          {isPlayground ? "Interactive console" : "Read-only"}
        </span>
      </footer>
    </div>
  );
}

export function Dashboard({
  state,
  initialEntries,
}: {
  state: MetricsState;
  initialEntries?: string[];
}) {
  if (initialEntries) {
    return (
      <MemoryRouter initialEntries={initialEntries}>
        <DashboardShell state={state} />
      </MemoryRouter>
    );
  }
  return (
    <BrowserRouter>
      <DashboardShell state={state} />
    </BrowserRouter>
  );
}

export default function App() {
  return <Dashboard state={useMetrics()} />;
}
