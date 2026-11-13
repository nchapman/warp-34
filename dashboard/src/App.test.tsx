import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Dashboard } from "./App";
import { histogram, snapshot } from "./fixtures";

describe("dashboard states", () => {
  it("shows an intentional connecting state without invented data", () => {
    render(
      <Dashboard
        state={{ history: [], connection: "connecting", error: null }}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Connecting");
    expect(screen.getByText("Waiting for engine…")).toBeVisible();
    expect(screen.queryByText("0 ms")).not.toBeInTheDocument();
    expect(screen.getAllByText("Collecting observations")).toHaveLength(3);
  });
  it("shows real model, capacity and load with accessible chart descriptions", () => {
    render(
      <Dashboard
        state={{ history: [snapshot()], connection: "live", error: null }}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Engine ready");
    expect(screen.getByText("test/model")).toBeVisible();
    expect(
      screen.getByRole("img", {
        name: /KV cache: 10 active, 20 evictable, 70 free/,
      }),
    ).toBeVisible();
    expect(
      screen.getByRole("img", { name: /Generation throughput/ }),
    ).toBeVisible();
    expect(screen.queryByText("Draft acceptance")).not.toBeInTheDocument();
  });
  it("keeps the last snapshot but explicitly labels disconnected data", () => {
    render(
      <Dashboard
        state={{
          history: [snapshot()],
          connection: "offline",
          error: "Network unavailable",
        }}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Disconnected");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "last received snapshot",
    );
    expect(screen.getByText("test/model")).toBeVisible();
  });
  it("separates unhealthy worker state from a healthy metrics connection", () => {
    render(
      <Dashboard
        state={{
          history: [snapshot({ status: "unhealthy" })],
          connection: "live",
          error: null,
        }}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Engine unhealthy");
    expect(screen.getByRole("alert")).toHaveTextContent("worker is unhealthy");
  });
  it("renders untrusted model names as text, never markup", () => {
    render(
      <Dashboard
        state={{
          history: [snapshot({ model: '<img src=x onerror="alert(1)">' })],
          connection: "live",
          error: null,
        }}
      />,
    );
    expect(screen.getByText('<img src=x onerror="alert(1)">')).toBeVisible();
    expect(document.querySelector("img")).toBeNull();
  });
  it("provides a read-only connection guide and security warning", () => {
    render(
      <Dashboard
        state={{ history: [], connection: "connecting", error: null }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    expect(
      screen.getByRole("heading", { name: "Bring your coding agent." }),
    ).toBeVisible();
    expect(
      screen.getByText(/API-key authentication is optional/),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "/api/metrics" })).toHaveAttribute(
      "href",
      "/api/metrics",
    );
  });
  it("populates throughput and latency panels for zero-generated-token decision workloads", () => {
    const s1 = snapshot({
      uptime_seconds: 100,
      tokens: { prompt: 1000, computed: 1000, cached: 0, generated: 0 },
      latency: {
        ttft: histogram(0),
        queue: histogram(10),
        e2e: histogram(10),
        time_per_output_token: histogram(0),
      },
    });
    const s2 = snapshot({
      uptime_seconds: 102,
      tokens: { prompt: 3000, computed: 3000, cached: 0, generated: 0 },
      latency: {
        ttft: histogram(0),
        queue: histogram(20),
        e2e: histogram(20),
        time_per_output_token: histogram(0),
      },
    });
    render(
      <Dashboard
        state={{ history: [s1, s2], connection: "live", error: null }}
      />,
    );
    expect(screen.getByText("Prompt tokens")).toBeVisible();
    expect(screen.getByText("1,000")).toBeVisible();
    expect(screen.getByText("Mean decision E2E")).toBeVisible();
  });
});
