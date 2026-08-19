"use client";

import { useEffect, useRef, useState } from "react";
import HorizontalPicksRow from "./HorizontalPicksRow";
import BookCard from "./BookCard";
import { fetchTopPicks, type Pick } from "../lib/api";

type Props = {
  userId?: number | null;
  picks?: Pick[];
  onPicksChange?: (picks: Pick[]) => void;
  onAddToCollection?: (pick: Pick) => void;
  onAddToCart?: (pick: Pick) => void;
  refreshKey?: number;
};

export default function TopPicksSection({
  userId = null,
  picks: controlledPicks,
  onPicksChange,
  onAddToCollection,
  onAddToCart,
  refreshKey = 0,
}: Props) {
  const [internalPicks, setInternalPicks] = useState<Pick[]>([]);
  const expandedRef = useRef<HTMLDivElement>(null);

  const picks = controlledPicks ?? internalPicks;

  useEffect(() => {
    if (userId == null) {
      setInternalPicks([]);
      onPicksChange?.([]);
      return;
    }

    let cancelled = false;
    Promise.resolve(fetchTopPicks(userId))
      .then((res) => {
        if (cancelled) return;
        const next = res?.picks ?? [];
        setInternalPicks(next);
        onPicksChange?.(next);
      })
      .catch(() => {
        if (cancelled) return;
        setInternalPicks([]);
        onPicksChange?.([]);
      });

    return () => {
      cancelled = true;
    };
    // refreshKey forces a re-fetch after interactions
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, refreshKey]);

  if (userId == null) {
    return null;
  }

  const showArrow = picks.length > 15;

  return (
    <section style={{ marginTop: 24 }}>
      <h2 style={{ marginBottom: 12 }}>Top picks for you</h2>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <HorizontalPicksRow
            picks={picks}
            onAddToCollection={onAddToCollection}
            onAddToCart={onAddToCart}
          />
        </div>
        {showArrow ? (
          <button
            type="button"
            aria-label="Show all top picks"
            onClick={() =>
              expandedRef.current?.scrollIntoView({ behavior: "smooth" })
            }
            style={{
              flexShrink: 0,
              width: 40,
              height: 40,
              borderRadius: 8,
              border: "1px solid #333",
              background: "rgba(255,255,255,0.7)",
              cursor: "pointer",
              fontSize: 20,
            }}
          >
            →
          </button>
        ) : null}
      </div>
      {showArrow ? (
        <div ref={expandedRef} style={{ marginTop: 32 }}>
          <h3 style={{ marginBottom: 12 }}>Top picks for you</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
              gap: 12,
            }}
          >
            {picks.map((p) => (
              <BookCard
                key={`expanded-${p.isbn}`}
                pick={p}
                onAddToCollection={
                  onAddToCollection
                    ? () => onAddToCollection(p)
                    : undefined
                }
                onAddToCart={onAddToCart ? () => onAddToCart(p) : undefined}
              />
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
