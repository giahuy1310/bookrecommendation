"use client";

import { useEffect, useState } from "react";
import AuthGuard from "../../components/AuthGuard";
import BookCard from "../../components/BookCard";
import { getCart, type BookRow } from "../../lib/api";
import { getUserId } from "../../lib/auth";

function CartContents() {
  const [items, setItems] = useState<BookRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const userId = getUserId();
    if (userId == null) return;
    getCart(userId)
      .then((res) => setItems(res.items))
      .catch(() => setError("Could not load cart."));
  }, []);

  return (
    <section className="inner-page padding-large">
      <div className="container">
        <div className="section-title mb-4">
          <h1 className="mb-2">My cart</h1>
          <p className="text-black-50 mb-0">
            Books you have added to your cart.
          </p>
        </div>
        {error ? <p className="text-danger">{error}</p> : null}
        {items.length === 0 && !error ? (
          <p>Books in your cart will appear here.</p>
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

export default function CartPage() {
  return (
    <AuthGuard requireAuth redirectTo="/login">
      <CartContents />
    </AuthGuard>
  );
}
