"use client";

import { Navigation } from "swiper/modules";
import { Swiper, SwiperSlide } from "swiper/react";

const SLIDES = [
  {
    title: "The Fine Print Book Collection",
    subtitle: "Find your next favorite read with UrFavBook",
    image: "/bookly/banner-image2.png",
    cta: "Search books",
  },
  {
    title: "How Innovation Works",
    subtitle: "Personalized picks based on what you love",
    image: "/bookly/banner-image1.png",
    cta: "Explore picks",
  },
  {
    title: "Your Heart is the Sea",
    subtitle: "Search, collect, and cart books in one place",
    image: "/bookly/banner-image.png",
    cta: "Start searching",
  },
];

export default function Billboard() {
  return (
    <section
      id="billboard"
      className="position-relative d-flex align-items-center py-5 bg-light-gray"
      style={{
        backgroundImage: "url(/bookly/banner-image-bg.jpg)",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        height: 800,
      }}
    >
      <div className="position-absolute end-0 pe-0 pe-xxl-5 me-0 me-xxl-5 swiper-next main-slider-button-next">
        <svg
          className="chevron-forward-circle d-flex justify-content-center align-items-center p-2"
          width="80"
          height="80"
        >
          <use xlinkHref="#alt-arrow-right-outline" />
        </svg>
      </div>
      <div className="position-absolute start-0 ps-0 ps-xxl-5 ms-0 ms-xxl-5 swiper-prev main-slider-button-prev">
        <svg
          className="chevron-back-circle d-flex justify-content-center align-items-center p-2"
          width="80"
          height="80"
        >
          <use xlinkHref="#alt-arrow-left-outline" />
        </svg>
      </div>
      <Swiper
        className="main-swiper"
        modules={[Navigation]}
        navigation={{
          nextEl: ".main-slider-button-next",
          prevEl: ".main-slider-button-prev",
        }}
        loop
        slidesPerView={1}
      >
        {SLIDES.map((slide) => (
          <SwiperSlide key={slide.title}>
            <div className="container">
              <div className="row d-flex flex-column-reverse flex-md-row align-items-center">
                <div className="col-md-5 offset-md-1 mt-5 mt-md-0 text-center text-md-start">
                  <div className="banner-content">
                    <h2>{slide.title}</h2>
                    <p>{slide.subtitle}</p>
                    <a href="#search-hero" className="btn mt-3">
                      {slide.cta}
                    </a>
                  </div>
                </div>
                <div className="col-md-6 text-center">
                  <div className="image-holder">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={slide.image}
                      className="img-fluid"
                      alt=""
                    />
                  </div>
                </div>
              </div>
            </div>
          </SwiperSlide>
        ))}
      </Swiper>
    </section>
  );
}
