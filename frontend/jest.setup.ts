import "@testing-library/jest-dom";

jest.mock("swiper/react", () => {
  const React = require("react");
  return {
    Swiper: ({ children }: { children: React.ReactNode }) =>
      React.createElement("div", { "data-testid": "swiper" }, children),
    SwiperSlide: ({ children }: { children: React.ReactNode }) =>
      React.createElement("div", { "data-testid": "swiper-slide" }, children),
  };
});

jest.mock("swiper/modules", () => ({
  Navigation: {},
}));
