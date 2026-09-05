import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import SiteHeader from "../components/SiteHeader";
import { getUserId, isLoggedIn } from "../lib/auth";

jest.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

jest.mock("../lib/auth", () => ({
  getUserId: jest.fn(),
  isLoggedIn: jest.fn(),
}));

test("logged-in nav includes Top picks linking to /top-picks", async () => {
  (isLoggedIn as jest.Mock).mockReturnValue(true);
  (getUserId as jest.Mock).mockReturnValue(123);

  render(<SiteHeader />);

  const link = await screen.findByRole("link", { name: /^Top picks$/i });
  expect(link).toHaveAttribute("href", "/top-picks");
});

test("logged-out nav does not include Top picks", async () => {
  (isLoggedIn as jest.Mock).mockReturnValue(false);
  (getUserId as jest.Mock).mockReturnValue(null);

  render(<SiteHeader />);

  await waitFor(() => {
    expect(screen.getByRole("link", { name: /Log-in/i })).toBeInTheDocument();
  });
  expect(screen.queryByRole("link", { name: /^Top picks$/i })).not.toBeInTheDocument();
});

test("brand shows UrFavBook", async () => {
  (isLoggedIn as jest.Mock).mockReturnValue(false);
  (getUserId as jest.Mock).mockReturnValue(null);

  render(<SiteHeader />);

  await waitFor(() => {
    expect(screen.getAllByText(/UrFavBook/i).length).toBeGreaterThan(0);
  });
});
