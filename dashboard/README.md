# warp34 dashboard

React + Vite, embedded into the Rust server at build time. Overview and Connect
are read-only; Playground sends real inference requests to the same origin.

## Access policy

The server defaults to `127.0.0.1`. Enabled UI pages/assets, `/health`, `/metrics`,
and `/api/metrics` are unauthenticated; binding and network/proxy access rules
control who can reach them. Configuring an API key protects API operations,
including Playground requests, **not** dashboard visibility or telemetry.

`warp34 serve MODEL --no-webui` returns 404 for dashboard pages and assets while
leaving API and metrics behavior unchanged. The flag does not remove assets from
the binary. Use an authenticated HTTPS reverse proxy if the entire UI needs a
login, and prevent direct access to the backend. See `docs/SERVING.md` for the
complete exposure policy.

## Develop and verify

```sh
pnpm --dir dashboard install --frozen-lockfile
pnpm --dir dashboard dev    # localhost:5173; API requests proxy to localhost:8000
pnpm --dir dashboard check  # TypeScript, Prettier, and all Vitest tests
pnpm --dir dashboard build # production assets in dist/, embedded by Cargo
```

Run `mise run check` from the repository root for the full integration gate,
including embedded asset serving. Rebuild the dashboard **before** rebuilding
Rust: `dashboard/dist` is compiled into the binary, not read at runtime.

## UI conventions

- Shared controls are locally owned [shadcn/ui](https://ui.shadcn.com) components
  in `src/components/ui`, generated with the `new-york` style and Radix primitives.
  `components.json` configures the registry and `@/` import alias.
- Use `Button` variants for action hierarchy, `Tabs` for in-page views, `Switch`
  for booleans, and labeled `Input`, `Textarea`, and `NativeSelect` controls.
  Native selects deliberately preserve browser keyboard and mobile picker behavior.
  Route navigation remains links, not tabs.
- Use `Card` for shared panel surfaces, `Alert` for operational notices, `Badge`
  for compact statuses, and `Field`/`FieldLabel`/`FieldDescription`/`FieldError`
  for form composition. Keep explicit label and description associations.
  Cards retain named regions and real headings; avoid duplicating their surfaces
  in legacy CSS. Native scrolling and simple reasoning disclosures remain native.
- `src/tokens.css` follows shadcn's **neutral dark** semantic tokens. `--accent`
  and `--muted` are surfaces; `--brand` (violet) and `--aqua` (cyan) are bright
  brand/data signals. Do not use `--muted` as a text color.
- Tailwind v4 utilities style shared components. `src/style.css` retains the
  dashboard's responsive layout and telemetry visualization rules in the base
  layer, so they do not override component utilities.
- Preserve keyboard focus, associated labels/panels, visible focus rings, and
  accessible status/error feedback. Don't rely on color alone for state.
  Request editors and endpoint selection stay locked while a request is running.
- Assets are same-origin and must work under the server CSP. No CDN styles,
  external fonts, inline scripts, or inline `<style>` elements. Style attributes
  are allowed. Adding an asset type also requires support
  in `crates/warp34-host/src/dashboard.rs`.

## Telemetry charts

`Chart.tsx` uses shadcn's Recharts integration, loaded through `LazyChart.tsx`
only when Overview renders. The production chart chunk is approximately 104 KB
compressed; Connect and Playground don't download or parse it. Check the Vite
bundle report after changes rather than suppressing chunk warnings.

- Keep the numeric five-minute engine-uptime axis, linear traces, real zero
  observations, explicit nulls, and breaks across sampling gaps over eight seconds.
  Do not interpolate missing observations or display stale samples as current.
- Preserve empty/collecting states, accessible summaries, keyboard exploration,
  and units/time in tooltips. Chart tests exercise real Recharts, not a mock.
- The locally owned `components/ui/chart.tsx` intentionally replaces generated
  `ChartStyle`/inline stylesheets with style-attribute CSS variables referencing
  external `--chart-1`/`--chart-2` tokens. Preserve this CSP adaptation when updating
  generated components; do not weaken the server policy to accommodate charts.

## Typography budget

`src/fonts.css` explicitly imports four self-hosted **Latin-only, normal WOFF2**
faces from Fontsource (Geist's SIL Open Font License):

| Face       | Weight |                  Bytes |
| ---------- | -----: | ---------------------: |
| Geist Sans |    400 |                 12,956 |
| Geist Sans |    500 |                 13,296 |
| Geist Sans |    600 |                 13,372 |
| Geist Mono |    400 |                  9,864 |
| **Total**  |        | **49,488 (~48.3 KiB)** |

Sans covers UI text, controls, and headings; Mono covers code, payloads, and
technical identifiers. Bold UI text uses 600 rather than loading 700. No italics,
unused language subsets, WOFF fallbacks, or full variable weight ranges ship.
System fallbacks cover glyphs outside the Latin faces. `font-display: swap`
keeps text visible while loading, and hashed assets receive immutable caching.
The Rust embedded-font test checks all four faces are served as `font/woff2`
and enforces a 52,000-byte budget.

Avoid importing `@fontsource/geist/index.css` or the entire font package's CSS:
those include unused subsets and formats. Keep explicit file imports when
adjusting typography, and inspect `dist/assets` after building.
