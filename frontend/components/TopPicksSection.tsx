"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import HorizontalPicksRow from "./HorizontalPicksRow";
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
        {picks.length > 0 ? (
          <Link
            href="/top-picks"
            aria-label="See all top picks"
            style={{
              flexShrink: 0,
              padding: "8px 12px",
              borderRadius: 8,
              border: "1px solid var(--palette-blue)",
              background: "var(--palette-blue)",
              color: "var(--color-text)",
              cursor: "pointer",
              fontSize: 14,
              fontFamily: "inherit",
              textDecoration: "none",
            }}
          >
            See all
          </Link>
        ) : null}
      </div>
    </section>
  );
}
