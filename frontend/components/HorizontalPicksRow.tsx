"use client";

import React from "react";
import { Navigation } from "swiper/modules";
import { Swiper, SwiperSlide } from "swiper/react";
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

  if (visible.length === 0) {
    return null;
  }

  return (
    <div className="position-relative">
      <div className="position-absolute top-50 end-0 pe-0 pe-xxl-5 me-0 me-xxl-5 swiper-next product-slider-button-next">
        <svg
          className="chevron-forward-circle d-flex justify-content-center align-items-center p-2"
          width="80"
          height="80"
        >
          <use xlinkHref="#alt-arrow-right-outline" />
        </svg>
      </div>
      <div className="position-absolute top-50 start-0 ps-0 ps-xxl-5 ms-0 ms-xxl-5 swiper-prev product-slider-button-prev">
        <svg
          className="chevron-back-circle d-flex justify-content-center align-items-center p-2"
          width="80"
          height="80"
        >
          <use xlinkHref="#alt-arrow-left-outline" />
        </svg>
      </div>
      <Swiper
        className="product-swiper"
        modules={[Navigation]}
        navigation={{
          nextEl: ".product-slider-button-next",
          prevEl: ".product-slider-button-prev",
        }}
        spaceBetween={20}
        breakpoints={{
          0: { slidesPerView: 1 },
          576: { slidesPerView: 2 },
          768: { slidesPerView: 3 },
          992: { slidesPerView: 4 },
        }}
      >
        {visible.map((p) => (
          <SwiperSlide key={p.isbn}>
            <BookCard
              pick={p}
              onAddToCollection={
                onAddToCollection ? () => onAddToCollection(p) : undefined
              }
              onAddToCart={onAddToCart ? () => onAddToCart(p) : undefined}
            />
          </SwiperSlide>
        ))}
      </Swiper>
    </div>
  );
}
