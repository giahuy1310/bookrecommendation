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

function coverForIsbn(isbn: string) {
  let hash = 0;
  for (let i = 0; i < isbn.length; i++) {
    hash = (hash + isbn.charCodeAt(i) * (i + 1)) % 12;
  }
  const n = (hash % 12) + 1;
  return `/bookly/product-item${n}.png`;
}

export default function BookCard({
  pick,
  onAddToCollection,
  onAddToCart,
}: Props) {
  return (
    <div
      data-testid="book-card"
      className="card position-relative p-4 border rounded-3 h-100"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={coverForIsbn(pick.isbn)}
        className="img-fluid shadow-sm"
        alt=""
      />
      <h6 className="mt-4 mb-0 fw-bold">{pick.title}</h6>
      <div className="review-content d-flex">
        <p className="my-2 me-2 fs-6 text-black-50 mb-0">{pick.author}</p>
      </div>
      {(onAddToCollection || onAddToCart) && (
        <div className="card-concern position-absolute start-0 end-0 d-flex gap-2">
          {onAddToCart ? (
            <button
              type="button"
              className="btn btn-dark"
              onClick={onAddToCart}
              aria-label="Add to cart"
              title="Add to cart"
            >
              <svg className="cart">
                <use xlinkHref="#cart" />
              </svg>
              <span className="visually-hidden">Add to cart</span>
            </button>
          ) : null}
          {onAddToCollection ? (
            <button
              type="button"
              className="btn btn-dark"
              onClick={onAddToCollection}
              aria-label="Add to collection"
              title="Add to collection"
            >
              <svg className="wishlist">
                <use xlinkHref="#heart" />
              </svg>
              <span className="visually-hidden">Add to collection</span>
            </button>
          ) : null}
        </div>
      )}
    </div>
  );
}
