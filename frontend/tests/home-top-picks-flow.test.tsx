import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import TopPicksSection from "../components/TopPicksSection";
import { fetchTopPicks } from "../lib/api";

jest.mock("../lib/api", () => ({
  fetchTopPicks: jest.fn(),
}));

function makePicks(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    isbn: `isbn-${i}`,
    title: `Title ${i}`,
    author: `Author ${i}`,
    finalScore: i,
  }));
}

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

test("home section shows one row and no expanded grid", async () => {
  (fetchTopPicks as jest.Mock).mockResolvedValue({
    userId: 123,
    contextIsbn: null,
    picks: makePicks(20),
  });

  render(<TopPicksSection userId={123} />);

  await waitFor(() => {
    expect(screen.getAllByTestId("book-card")).toHaveLength(15);
  });
  expect(screen.getAllByRole("heading", { name: /Top picks for you/i })).toHaveLength(
    1
  );
});

test("See all links to the top picks page when there are picks", async () => {
  (fetchTopPicks as jest.Mock).mockResolvedValue({
    userId: 123,
    contextIsbn: null,
    picks: makePicks(3),
  });

  render(<TopPicksSection userId={123} />);

  const link = await screen.findByRole("link", { name: /see all/i });
  expect(link).toHaveAttribute("href", "/top-picks");
});
