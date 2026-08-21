"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type CSSProperties } from "react";
import { getUserId, isLoggedIn } from "../lib/auth";

function navLinkStyle(active: boolean): CSSProperties {
  return {
    color: active ? "var(--palette-blue)" : "var(--color-text)",
    fontWeight: active ? 700 : 400,
    textDecoration: active ? "underline" : "none",
  };
}

export default function NavBar() {
  const pathname = usePathname();
  const [loggedIn, setLoggedIn] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setLoggedIn(isLoggedIn() && getUserId() != null);
    setReady(true);
  }, [pathname]);

  const navStyle: CSSProperties = {
    display: "flex",
    gap: 16,
    padding: "16px 24px",
    alignItems: "center",
    flexWrap: "wrap",
    borderBottom: "2px solid var(--palette-tan)",
    background: "var(--color-bg)",
  };

  if (!ready) {
    return (
      <nav style={navStyle}>
        <Link
          href="/"
          style={{
            fontWeight: 700,
            fontSize: 20,
            color: "var(--palette-blue)",
          }}
        >
          BookCrossing
        </Link>
      </nav>
    );
  }

  return (
    <nav style={navStyle}>
      <Link
        href="/"
        style={{
          fontWeight: 700,
          fontSize: 20,
          color: "var(--palette-blue)",
        }}
      >
        BookCrossing
      </Link>
      <Link href="/" style={navLinkStyle(pathname === "/")}>
        Home
      </Link>
      {loggedIn ? (
        <>
          <Link
            href="/top-picks"
            style={navLinkStyle(pathname === "/top-picks")}
          >
            Top picks
          </Link>
          <Link
            href="/collection"
            style={navLinkStyle(pathname === "/collection")}
          >
            My collection
          </Link>
          <Link href="/cart" style={navLinkStyle(pathname === "/cart")}>
            My cart
          </Link>
          <Link
            href="/profile"
            style={navLinkStyle(pathname === "/profile")}
          >
            User profile
          </Link>
        </>
      ) : (
        <Link href="/login" style={navLinkStyle(pathname === "/login")}>
          Log-in
        </Link>
      )}
    </nav>
  );
}
