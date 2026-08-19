"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getUserId, isLoggedIn } from "../lib/auth";

export default function NavBar() {
  const pathname = usePathname();
  const [loggedIn, setLoggedIn] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setLoggedIn(isLoggedIn() && getUserId() != null);
    setReady(true);
  }, [pathname]);

  if (!ready) {
    return (
      <nav
        style={{
          display: "flex",
          gap: 16,
          padding: "16px 24px",
          alignItems: "center",
        }}
      >
        <Link href="/" style={{ fontWeight: 700, fontSize: 20 }}>
          BookCrossing
        </Link>
      </nav>
    );
  }

  return (
    <nav
      style={{
        display: "flex",
        gap: 16,
        padding: "16px 24px",
        alignItems: "center",
        flexWrap: "wrap",
      }}
    >
      <Link href="/" style={{ fontWeight: 700, fontSize: 20 }}>
        BookCrossing
      </Link>
      <Link href="/">Home</Link>
      {loggedIn ? (
        <>
          <Link href="/collection">My collection</Link>
          <Link href="/cart">My cart</Link>
          <Link href="/profile">User profile</Link>
        </>
      ) : (
        <Link href="/login">Log-in</Link>
      )}
    </nav>
  );
}
