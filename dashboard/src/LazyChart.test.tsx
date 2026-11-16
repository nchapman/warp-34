import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Dashboard } from "./App";

// Simulate a network/deployment failure at the dynamic-import boundary.
vi.mock("./Chart", () => {
  throw new Error("Chart chunk could not be downloaded");
});

describe("chart chunk failure", () => {
  it("keeps the dashboard and navigation usable when charts cannot load", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <Dashboard
        state={{ history: [], connection: "connecting", error: null }}
      />,
    );
    expect(await screen.findAllByText("Chart unavailable")).toHaveLength(3);
    expect(screen.getByRole("heading", { name: "Overview" })).toBeVisible();
    expect(screen.getByRole("link", { name: "Connect" })).toBeVisible();
    expect(screen.getByRole("link", { name: "Playground" })).toBeVisible();
  });
});
