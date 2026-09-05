"use client";

import { useEffect, useState } from "react";
import AuthGuard from "../../components/AuthGuard";
import BookCard from "../../components/BookCard";
import { fetchTopPicks, postInteraction, type Pick } from "../../lib/api";
import { getUserId } from "../../lib/auth";

function TopPicksContents() {
  const [userId, setUserId] = useState<number | null>(null);
  const [picks, setPicks] = useState<Pick[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    setUserId(getUserId());
  }, []);

  useEffect(() => {
    if (userId == null) return;

    let cancelled = false;
    fetchTopPicks(userId)
      .then((res) => {
        if (!cancelled) setPicks(res?.picks ?? []);
      })
      .catch(() => {
        if (!cancelled) setPicks([]);
      });

    return () => {
      cancelled = true;
    };
  }, [userId, refreshKey]);

  async function sendInteraction(
    isbn: string,
    eventType: "ADD_TO_CART" | "ADD_TO_COLLECTION"
  ) {
    const uid = getUserId();
    if (uid == null) return;
    await postInteraction({
      userId: uid,
      isbn,
      eventType,
      createdAtMs: Date.now(),
    });
    setRefreshKey((k) => k + 1);
  }

  return (
    <section className="inner-page padding-large">
      <div className="container">
        <div className="section-title mb-4">
          <h1 className="mb-2">Top picks for you</h1>
          <p className="text-black-50 mb-0">
            Your full personalized recommendation list.
          </p>
        </div>
        <div className="book-grid">
          {picks.map((p) => (
            <BookCard
              key={p.isbn}
              pick={p}
              onAddToCollection={() => {
                void sendInteraction(p.isbn, "ADD_TO_COLLECTION");
              }}
              onAddToCart={() => {
                void sendInteraction(p.isbn, "ADD_TO_CART");
              }}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function TopPicksPage() {
  return (
    <AuthGuard requireAuth redirectTo="/login">
      <TopPicksContents />
    </AuthGuard>
  );
}
