"use client";

import React from "react";

type Pick = {
  isbn: string;
  title: string;
  author: string;
  finalScore?: number;
};

type Props = {
  pick: Pick;
  onAddToCollection?: () => void;
  onAddToCart?: () => void;
};

export default function BookCard({
  pick,
  onAddToCollection,
  onAddToCart,
}: Props) {
  return (
    <div
      data-testid="book-card"
      style={{
        minWidth: 180,
        padding: 12,
        borderRadius: 10,
        background: "color-mix(in srgb, var(--palette-tan) 28%, white)",
        border: "1px solid var(--palette-tan)",
      }}
    >
      <div style={{ fontWeight: 700 }}>{pick.title}</div>
      <div style={{ opacity: 0.8, fontSize: 12 }}>{pick.author}</div>
      {(onAddToCollection || onAddToCart) && (
        <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
          {onAddToCollection ? (
            <button
              type="button"
              onClick={onAddToCollection}
              style={{
                fontSize: 11,
                padding: "4px 8px",
                borderRadius: 6,
                border: "1px solid var(--palette-pink)",
                background: "var(--palette-pink)",
                color: "var(--color-text)",
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Add to collection
            </button>
          ) : null}
          {onAddToCart ? (
            <button
              type="button"
              onClick={onAddToCart}
              style={{
                fontSize: 11,
                padding: "4px 8px",
                borderRadius: 6,
                border: "1px solid var(--palette-blue)",
                background: "var(--palette-blue)",
                color: "var(--color-text)",
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Add to cart
            </button>
          ) : null}
        </div>
      )}
    </div>
  );
}
