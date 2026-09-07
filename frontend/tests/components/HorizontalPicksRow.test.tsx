import { render, screen } from "@testing-library/react";
import HorizontalPicksRow from "../../components/HorizontalPicksRow";

test("renders at most 3 picks in the main row", () => {
  const picks = Array.from({ length: 30 }).map((_, i) => ({
    isbn: `isbn-${i}`,
    title: `Title ${i}`,
    author: `Author ${i}`,
    finalScore: i,
  }));

  render(<HorizontalPicksRow picks={picks} />);
  const cards = screen.getAllByTestId("book-card");
  expect(cards).toHaveLength(3);
});
