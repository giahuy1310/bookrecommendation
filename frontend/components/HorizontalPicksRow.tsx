import React from "react";
import BookCard from "./BookCard";

type Pick = {
  isbn: string;
  title: string;
  author: string;
  finalScore: number;
};

export default function HorizontalPicksRow({ picks }: { picks: Pick[] }) {
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
        <BookCard key={p.isbn} pick={p} />
      ))}
    </div>
  );
}
