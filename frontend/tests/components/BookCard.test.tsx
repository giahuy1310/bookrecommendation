import { render, screen } from "@testing-library/react";
import BookCard from "../../components/BookCard";

const basePick = {
  isbn: "1234567890",
  title: "Test Title",
  author: "Test Author",
  finalScore: 1,
};

function cardImg() {
  return screen.getByTestId("book-card").querySelector("img");
}

test("uses coverUrl when present", () => {
  render(
    <BookCard
      pick={{
        ...basePick,
        coverUrl: "https://covers.openlibrary.org/b/isbn/1234567890-L.jpg",
      }}
    />
  );
  expect(cardImg()).toHaveAttribute(
    "src",
    "https://covers.openlibrary.org/b/isbn/1234567890-L.jpg"
  );
});

test("falls back to placeholder when coverUrl is missing", () => {
  render(<BookCard pick={basePick} />);
  expect(cardImg()?.getAttribute("src")).toMatch(
    /^\/bookly\/product-item\d+\.png$/
  );
});

test("falls back to placeholder when coverUrl is null", () => {
  render(<BookCard pick={{ ...basePick, coverUrl: null }} />);
  expect(cardImg()?.getAttribute("src")).toMatch(
    /^\/bookly\/product-item\d+\.png$/
  );
});
