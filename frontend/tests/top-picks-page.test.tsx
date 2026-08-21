import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import TopPicksPage from "../app/top-picks/page";
import { fetchTopPicks, postInteraction } from "../lib/api";
import { getUserId } from "../lib/auth";

jest.mock("../lib/api", () => ({
  fetchTopPicks: jest.fn(),
  postInteraction: jest.fn(),
}));

jest.mock("../lib/auth", () => ({
  getUserId: jest.fn(),
  isLoggedIn: jest.fn(() => true),
}));

jest.mock("../components/AuthGuard", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

function makePicks(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    isbn: `isbn-${i}`,
    title: `Title ${i}`,
    author: `Author ${i}`,
    finalScore: i,
  }));
}

test("top picks page renders the full pick list", async () => {
  (getUserId as jest.Mock).mockReturnValue(123);
  (fetchTopPicks as jest.Mock).mockResolvedValue({
    userId: 123,
    contextIsbn: null,
    picks: makePicks(20),
  });

  render(<TopPicksPage />);

  expect(screen.getByRole("heading", { name: /Top picks for you/i })).toBeInTheDocument();
  await waitFor(() => {
    expect(screen.getAllByTestId("book-card")).toHaveLength(20);
  });
  expect(fetchTopPicks).toHaveBeenCalledWith(123);
});

test("top picks page posts add-to-cart interactions", async () => {
  (getUserId as jest.Mock).mockReturnValue(123);
  (fetchTopPicks as jest.Mock).mockResolvedValue({
    userId: 123,
    contextIsbn: null,
    picks: makePicks(1),
  });
  (postInteraction as jest.Mock).mockResolvedValue(undefined);

  render(<TopPicksPage />);
  const addToCart = await screen.findByRole("button", { name: /add to cart/i });
  fireEvent.click(addToCart);

  await waitFor(() => {
    expect(postInteraction).toHaveBeenCalledWith(
      expect.objectContaining({
        userId: 123,
        isbn: "isbn-0",
        eventType: "ADD_TO_CART",
      })
    );
  });
});
