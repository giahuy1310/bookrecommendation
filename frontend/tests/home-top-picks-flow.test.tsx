import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import TopPicksSection from "../components/TopPicksSection";
import { fetchTopPicks } from "../lib/api";

jest.mock("../lib/api", () => ({
  fetchTopPicks: jest.fn(),
}));

test("TopPicksSection requests top picks for the current user", async () => {
  (fetchTopPicks as jest.Mock).mockResolvedValue({
    userId: 123,
    contextIsbn: null,
    picks: [],
  });

  render(<TopPicksSection userId={123} />);
  expect(screen.getByText(/Top picks for you/i)).toBeInTheDocument();
  await waitFor(() => {
    expect(fetchTopPicks).toHaveBeenCalledWith(123);
  });
});
