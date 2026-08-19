"use client";

import { useEffect, useState } from "react";
import AuthGuard from "../../components/AuthGuard";
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
    <section>
      <h1>My collection</h1>
      {error ? <p>{error}</p> : null}
      {items.length === 0 && !error ? (
        <p>Your saved books will appear here.</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, marginTop: 16 }}>
          {items.map((book) => (
            <li
              key={book.isbn}
              style={{
                marginBottom: 12,
                padding: "12px 0",
                borderBottom: "1px solid rgba(0,0,0,0.12)",
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

export default function CollectionPage() {
  return (
    <AuthGuard requireAuth redirectTo="/login">
      <CollectionContents />
    </AuthGuard>
  );
}
