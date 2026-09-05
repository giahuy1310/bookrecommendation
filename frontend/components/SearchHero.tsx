"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { searchBooks, type BookRow } from "../lib/api";
import { getUserId } from "../lib/auth";

type Props = {
  onSelectBook?: (book: BookRow) => void | Promise<void>;
};

export default function SearchHero({ onSelectBook }: Props) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<BookRow[]>([]);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults([]);
      setSearching(false);
      return;
    }

    let cancelled = false;
    setSearching(true);
    const handle = window.setTimeout(() => {
      searchBooks(q)
        .then((books) => {
          if (!cancelled) setResults(books);
        })
        .catch(() => {
          if (!cancelled) setResults([]);
        })
        .finally(() => {
          if (!cancelled) setSearching(false);
        });
    }, 300);

    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [query]);

  async function handleSelect(book: BookRow) {
    const userId = getUserId();
    if (userId == null) {
      router.push("/login");
      return;
    }
    await onSelectBook?.(book);
    setQuery("");
    setResults([]);
  }

  return (
    <section id="search-hero" className="padding-large pb-0">
      <div className="container">
        <div className="section-title mb-4 text-center text-md-start">
          <h3 className="mb-2">What do you like to read today?</h3>
          <p className="mb-0 text-black-50">
            Search by title or author to start a reading interaction.
          </p>
        </div>
        <form
          role="search"
          className="search-form position-relative"
          onSubmit={(e) => e.preventDefault()}
        >
          <label className="visually-hidden" htmlFor="search-form">
            Search books
          </label>
          <input
            type="search"
            id="search-form"
            className="search-field form-control form-control-lg rounded-3"
            placeholder="Search by title or author"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            name="s"
          />
          <button type="submit" className="search-submit" aria-label="Search">
            <svg className="search">
              <use xlinkHref="#search" />
            </svg>
          </button>
        </form>
        {searching ? (
          <p className="mt-3 text-black-50">Searching…</p>
        ) : null}
        {results.length > 0 ? (
          <ul className="search-hero-results mt-3">
            {results.map((book) => (
              <li key={book.isbn}>
                <button
                  type="button"
                  className="btn btn-light w-100 text-start border rounded-3 p-3"
                  onClick={() => handleSelect(book)}
                >
                  <div className="fw-bold">{book.title}</div>
                  <div className="fs-6 text-black-50">{book.author}</div>
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  );
}
