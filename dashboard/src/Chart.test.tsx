import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Chart, normalizeChartPoints } from "./Chart";

describe("chart observations", () => {
  it("preserves zeros, explicit gaps and the eight-second boundary without inventing values", () => {
    expect(
      normalizeChartPoints([
        { time: 0, value: 0 },
        { time: 8, value: 2 },
        { time: 17, value: 3 },
        { time: 19, value: NaN },
        { time: 21, value: Infinity },
        { time: 23, value: null },
      ]),
    ).toEqual([
      { time: 0, value: 0 },
      { time: 8, value: 2 },
      { time: 12.5, value: null },
      { time: 17, value: 3 },
      { time: 19, value: null },
      { time: 21, value: null },
      { time: 23, value: null },
    ]);
  });
  it("bounds the window and discards nonfinite times", () => {
    expect(
      normalizeChartPoints([
        { time: 0, value: 7 },
        { time: NaN, value: 8 },
        { time: 301, value: 0 },
      ]),
    ).toEqual([{ time: 301, value: 0 }]);
  });
  it("shows collecting state without fake observations", () => {
    const { container } = render(
      <Chart points={[]} label="Throughput" unit="tok/s" />,
    );
    expect(screen.getByRole("img")).toHaveAccessibleName(
      /Collecting observations.*Latest observation unavailable/,
    );
    expect(screen.getByText("Collecting observations")).toBeVisible();
    expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(0);
    expect(container.querySelector("style")).toBeNull();
  });
  it("does not present a stale sample as latest and draws linear disconnected traces", () => {
    const { container } = render(
      <Chart
        points={[
          { time: 100, value: 0 },
          { time: 102, value: 2 },
          { time: 120, value: 3 },
          { time: 122, value: 4 },
          { time: 124, value: null },
        ]}
        label="Throughput"
        unit="tok/s"
        compact
      />,
    );
    expect(screen.getByRole("img")).toHaveAccessibleName(
      /Latest observation unavailable.*Peak 4 tok\/s/,
    );
    const path = container
      .querySelector(".recharts-line-curve")
      ?.getAttribute("d");
    expect(path?.match(/M/g)).toHaveLength(2);
    expect(path).not.toMatch(/[CQ]|NaN/);
    expect(container.querySelector('[data-slot="chart"]')).toHaveStyle({
      height: "130px",
      "--color-value": "var(--chart-1)",
    });
    expect(container.querySelector("style")).toBeNull();
    expect(screen.getByText("5 minutes ago")).toBeInTheDocument();
    expect(screen.getByText("Now")).toBeInTheDocument();
  });
  it("exposes real Recharts keyboard interaction and a units/time tooltip", async () => {
    render(
      <Chart
        points={[
          { time: 100, value: 0 },
          { time: 102, value: 2 },
        ]}
        label="Throughput"
        unit="tok/s"
      />,
    );
    const application = screen.getByRole("application");
    expect(application.closest('[role="img"]')).toBeNull();
    expect(application).toHaveAttribute("tabindex", "0");
    fireEvent.focus(application);
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() =>
      expect(screen.getByText(/Throughput: .*tok\/s/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/Engine uptime .* s ago/)).toBeInTheDocument();
  });
});
