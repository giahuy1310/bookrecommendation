"use client";

import React from "react";
import BookCard from "./BookCard";

type Pick = {
  isbn: string;
  title: string;
  author: string;
  finalScore: number;
};

type Props = {
  picks: Pick[];
  onAddToCollection?: (pick: Pick) => void;
  onAddToCart?: (pick: Pick) => void;
};

export default function HorizontalPicksRow({
  picks,
  onAddToCollection,
  onAddToCart,
}: Props) {
  const visible = picks.slice(0, 15);
  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        overflowX: "auto",
        paddingBottom: 8,
      }}
    >
      {visible.map((p) => (
        <BookCard
          key={p.isbn}
          pick={p}
          onAddToCollection={
            onAddToCollection ? () => onAddToCollection(p) : undefined
          }
          onAddToCart={onAddToCart ? () => onAddToCart(p) : undefined}
        />
      ))}
    </div>
  );
}
