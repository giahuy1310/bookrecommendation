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
    <section
      style={{
        margin: "24px 0 32px",
        padding: "28px 24px",
        borderRadius: 20,
        background: "color-mix(in srgb, var(--palette-blue) 18%, white)",
        border: "1px solid var(--palette-tan)",
        boxShadow: "0 8px 24px color-mix(in srgb, var(--palette-blue) 20%, transparent)",
      }}
    >
      <h1 style={{ fontSize: "2rem", margin: "0 0 16px", fontWeight: 700 }}>
        what do you like to read today?
      </h1>
      <label style={{ display: "block" }}>
        <span style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>
          Search books
        </span>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by title or author"
          style={{
            width: "100%",
            padding: "14px 16px",
            fontSize: 16,
            borderRadius: 12,
            border: "1px solid var(--palette-tan)",
            fontFamily: "inherit",
          }}
        />
      </label>
      {searching ? (
        <p style={{ marginTop: 12, opacity: 0.7 }}>Searching…</p>
      ) : null}
      {results.length > 0 ? (
        <ul
          style={{
            listStyle: "none",
            margin: "16px 0 0",
            padding: 0,
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          {results.map((book) => (
            <li key={book.isbn}>
              <button
                type="button"
                onClick={() => handleSelect(book)}
                style={{
                  width: "100%",
                  textAlign: "left",
                  padding: "12px 14px",
                  borderRadius: 10,
                  border: "1px solid var(--palette-tan)",
                  background: "color-mix(in srgb, var(--palette-tan) 35%, white)",
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                <div style={{ fontWeight: 700 }}>{book.title}</div>
                <div style={{ fontSize: 13, opacity: 0.8 }}>{book.author}</div>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
