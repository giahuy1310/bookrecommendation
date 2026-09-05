"use client";

import { useEffect, useState } from "react";
import AuthGuard from "../../components/AuthGuard";
import BookCard from "../../components/BookCard";
import { getCollection, type BookRow } from "../../lib/api";
import { getUserId } from "../../lib/auth";

function CollectionContents() {
  const [items, setItems] = useState<BookRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const userId = getUserId();
    if (userId == null) return;
    getCollection(userId)
      .then((res) => setItems(res.items))
      .catch(() => setError("Could not load collection."));
  }, []);

  return (
    <section className="inner-page padding-large">
      <div className="container">
        <div className="section-title mb-4">
          <h1 className="mb-2">My collection</h1>
          <p className="text-black-50 mb-0">
            Books you have saved to your collection.
          </p>
        </div>
        {error ? <p className="text-danger">{error}</p> : null}
        {items.length === 0 && !error ? (
          <p>Your saved books will appear here.</p>
        ) : (
          <div className="book-grid">
            {items.map((book) => (
              <BookCard key={book.isbn} pick={book} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

export default function CollectionPage() {
  return (
    <AuthGuard requireAuth redirectTo="/login">
      <CollectionContents />
    </AuthGuard>
  );
}
