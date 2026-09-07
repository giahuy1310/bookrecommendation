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
    expect(screen.getAllByTestId("book-card")).toHaveLength(3);
  });
  expect(screen.getAllByRole("heading", { name: /Top picks for you/i })).toHaveLength(
    1
  );
});

test("home slider uses coverUrl when present and placeholders otherwise", async () => {
  (fetchTopPicks as jest.Mock).mockResolvedValue({
    userId: 123,
    contextIsbn: null,
    picks: [
      {
        isbn: "isbn-0",
        title: "Covered",
        author: "Author 0",
        finalScore: 3,
        coverUrl: "https://covers.openlibrary.org/b/isbn/isbn-0-L.jpg",
      },
      {
        isbn: "isbn-1",
        title: "Plain",
        author: "Author 1",
        finalScore: 2,
      },
      {
        isbn: "isbn-2",
        title: "Null cover",
        author: "Author 2",
        finalScore: 1,
        coverUrl: null,
      },
      {
        isbn: "isbn-3",
        title: "Hidden",
        author: "Author 3",
        finalScore: 0,
      },
    ],
  });

  render(<TopPicksSection userId={123} />);

  const cards = await screen.findAllByTestId("book-card");
  expect(cards).toHaveLength(3);
  expect(cards[0].querySelector("img")).toHaveAttribute(
    "src",
    "https://covers.openlibrary.org/b/isbn/isbn-0-L.jpg"
  );
  expect(cards[1].querySelector("img")?.getAttribute("src")).toMatch(
    /^\/bookly\/product-item\d+\.png$/
  );
  expect(cards[2].querySelector("img")?.getAttribute("src")).toMatch(
    /^\/bookly\/product-item\d+\.png$/
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
