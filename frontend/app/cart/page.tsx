"use client";

import { useEffect, useState } from "react";
import AuthGuard from "../../components/AuthGuard";
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
    <section>
      <h1>My cart</h1>
      {error ? <p>{error}</p> : null}
      {items.length === 0 && !error ? (
        <p>Books in your cart will appear here.</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, marginTop: 16 }}>
          {items.map((book) => (
            <li
              key={book.isbn}
              style={{
                marginBottom: 12,
                padding: "12px 0",
                borderBottom: "1px solid var(--palette-tan)",
              }}
            >
              <div style={{ fontWeight: 700 }}>{book.title}</div>
              <div style={{ fontSize: 13, opacity: 0.8 }}>{book.author}</div>
            </li>
          ))}
        </ul>
      )}
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
