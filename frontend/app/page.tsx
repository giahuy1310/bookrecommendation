"use client";

import { useEffect, useState } from "react";
import SearchHero from "../components/SearchHero";
import TopPicksSection from "../components/TopPicksSection";
import {
  postInteraction,
  type BookRow,
  type Pick,
} from "../lib/api";
import { getUserId } from "../lib/auth";

export default function HomePage() {
  const [userId, setUserId] = useState<number | null>(null);
  const [ready, setReady] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    setUserId(getUserId());
    setReady(true);
  }, []);

  function bumpRefresh() {
    setRefreshKey((k) => k + 1);
  }

  async function sendInteraction(
    isbn: string,
    eventType: "READ" | "ADD_TO_CART" | "ADD_TO_COLLECTION"
  ) {
    const uid = getUserId();
    if (uid == null) return;
    await postInteraction({
      userId: uid,
      isbn,
      eventType,
      createdAtMs: Date.now(),
    });
    bumpRefresh();
  }

  async function handleSelectBook(book: BookRow) {
    await sendInteraction(book.isbn, "READ");
  }

  function handleAddToCollection(pick: Pick) {
    void sendInteraction(pick.isbn, "ADD_TO_COLLECTION");
  }

  function handleAddToCart(pick: Pick) {
    void sendInteraction(pick.isbn, "ADD_TO_CART");
  }

  return (
    <>
      <SearchHero onSelectBook={handleSelectBook} />
      {ready && userId != null ? (
        <TopPicksSection
          userId={userId}
          refreshKey={refreshKey}
          onAddToCollection={handleAddToCollection}
          onAddToCart={handleAddToCart}
        />
      ) : null}
    </>
  );
}
