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
    <section id="best-selling-items" className="position-relative padding-large">
      <div className="container">
        <div className="section-title d-md-flex justify-content-between align-items-center mb-4">
          <h3 className="d-flex align-items-center">Top picks for you</h3>
          {picks.length > 0 ? (
            <Link href="/top-picks" className="btn" aria-label="See all top picks">
              See all
            </Link>
          ) : null}
        </div>
        <HorizontalPicksRow
          picks={picks}
          onAddToCollection={onAddToCollection}
          onAddToCart={onAddToCart}
        />
      </div>
    </section>
  );
}
