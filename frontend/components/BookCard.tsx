import React from "react";

type Pick = {
  isbn: string;
  title: string;
  author: string;
  finalScore: number;
};

export default function BookCard({ pick }: { pick: Pick }) {
  return (
    <div data-testid="book-card" style={{ minWidth: 180 }}>
      <div style={{ fontWeight: 700 }}>{pick.title}</div>
      <div style={{ opacity: 0.8, fontSize: 12 }}>{pick.author}</div>
    </div>
  );
}
