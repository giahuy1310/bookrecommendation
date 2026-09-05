"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getUserId, isLoggedIn } from "../lib/auth";

function navLinkClass(active: boolean) {
  return `nav-link me-4${active ? " active" : ""}`;
}

export default function SiteHeader() {
  const pathname = usePathname();
  const [loggedIn, setLoggedIn] = useState(false);
  const [ready, setReady] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    setLoggedIn(isLoggedIn() && getUserId() != null);
    setReady(true);
    setMenuOpen(false);
  }, [pathname]);

  const brand = (
    <Link className="navbar-brand d-flex align-items-center gap-2" href="/">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/bookly/main-logo.png" className="logo" alt="" />
      <span className="fw-bold text-uppercase">UrFavBook</span>
    </Link>
  );

  return (
    <header id="header" className="site-header">
      <div className="top-info border-bottom d-none d-md-block">
        <div className="container-fluid">
          <div className="row g-0">
            <div className="col-md-4">
              <p className="fs-6 my-2 text-center">
                Personalized picks from your reading taste
              </p>
            </div>
            <div className="col-md-4 border-start border-end">
              <p className="fs-6 my-2 text-center">
                Discover books you&apos;ll love with UrFavBook
              </p>
            </div>
            <div className="col-md-4">
              <p className="fs-6 my-2 text-center">
                Search, collect, and cart your next reads
              </p>
            </div>
          </div>
        </div>
      </div>

      <nav id="header-nav" className="navbar navbar-expand-lg py-3">
        <div className="container">
          {brand}
          <button
            className="navbar-toggler d-flex d-lg-none order-3 p-2"
            type="button"
            aria-controls="bdNavbar"
            aria-expanded={menuOpen}
            aria-label="Toggle navigation"
            onClick={() => setMenuOpen((o) => !o)}
          >
            <svg className="navbar-icon">
              <use xlinkHref="#navbar-icon" />
            </svg>
          </button>

          <div
            className={`offcanvas-lg offcanvas-end${menuOpen ? " show" : ""}`}
            tabIndex={-1}
            id="bdNavbar"
            aria-labelledby="bdNavbarOffcanvasLabel"
            style={
              menuOpen
                ? {
                    visibility: "visible",
                    position: "fixed",
                    inset: 0,
                    zIndex: 1045,
                    background: "var(--white-color, #fff)",
                    padding: "1rem",
                  }
                : undefined
            }
          >
            <div className="offcanvas-header px-4 pb-0 d-lg-none">
              {brand}
              <button
                type="button"
                className="btn-close"
                aria-label="Close"
                onClick={() => setMenuOpen(false)}
              />
            </div>
            <div className="offcanvas-body">
              <ul
                id="navbar"
                className="navbar-nav text-uppercase justify-content-start justify-content-lg-center align-items-start align-items-lg-center flex-grow-1"
              >
                <li className="nav-item">
                  <Link
                    className={navLinkClass(pathname === "/")}
                    href="/"
                  >
                    Home
                  </Link>
                </li>
                {ready && loggedIn ? (
                  <>
                    <li className="nav-item">
                      <Link
                        className={navLinkClass(pathname === "/top-picks")}
                        href="/top-picks"
                      >
                        Top picks
                      </Link>
                    </li>
                    <li className="nav-item">
                      <Link
                        className={navLinkClass(pathname === "/collection")}
                        href="/collection"
                      >
                        My collection
                      </Link>
                    </li>
                    <li className="nav-item">
                      <Link
                        className={navLinkClass(pathname === "/cart")}
                        href="/cart"
                      >
                        My cart
                      </Link>
                    </li>
                  </>
                ) : null}
              </ul>

              <div className="user-items d-flex">
                <ul className="d-flex justify-content-end list-unstyled mb-0 align-items-center">
                  <li className="search-item pe-3">
                    <a href="#search-hero" className="search-button" aria-label="Search">
                      <svg className="search">
                        <use xlinkHref="#search" />
                      </svg>
                    </a>
                  </li>
                  {ready && loggedIn ? (
                    <>
                      <li className="pe-3">
                        <Link href="/collection" aria-label="My collection">
                          <svg className="wishlist">
                            <use xlinkHref="#heart" />
                          </svg>
                        </Link>
                      </li>
                      <li className="pe-3">
                        <Link href="/cart" aria-label="My cart">
                          <svg className="cart">
                            <use xlinkHref="#cart" />
                          </svg>
                        </Link>
                      </li>
                      <li>
                        <Link href="/profile" aria-label="User profile">
                          <svg className="user">
                            <use xlinkHref="#user" />
                          </svg>
                        </Link>
                      </li>
                    </>
                  ) : ready ? (
                    <li>
                      <Link
                        className={navLinkClass(pathname === "/login")}
                        href="/login"
                      >
                        Log-in
                      </Link>
                    </li>
                  ) : null}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
}
