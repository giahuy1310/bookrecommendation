import Link from "next/link";

export default function SiteFooter() {
  return (
    <>
      <footer id="footer" className="padding-large">
        <div className="container">
          <div className="row">
            <div className="footer-top-area">
              <div className="row d-flex flex-wrap justify-content-between">
                <div className="col-lg-3 col-sm-6 pb-3">
                  <div className="footer-menu">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src="/bookly/main-logo.png"
                      alt="UrFavBook"
                      className="img-fluid mb-2"
                    />
                    <p className="fw-bold text-uppercase mb-2">UrFavBook</p>
                    <p>
                      Personalized book recommendations based on what you read,
                      collect, and add to your cart.
                    </p>
                    <div className="social-links">
                      <ul className="d-flex list-unstyled">
                        <li>
                          <a href="https://templatesjungle.com/" aria-label="Facebook">
                            <svg className="facebook">
                              <use xlinkHref="#facebook" />
                            </svg>
                          </a>
                        </li>
                        <li>
                          <a href="https://templatesjungle.com/" aria-label="Instagram">
                            <svg className="instagram">
                              <use xlinkHref="#instagram" />
                            </svg>
                          </a>
                        </li>
                        <li>
                          <a href="https://templatesjungle.com/" aria-label="Twitter">
                            <svg className="twitter">
                              <use xlinkHref="#twitter" />
                            </svg>
                          </a>
                        </li>
                        <li>
                          <a href="https://templatesjungle.com/" aria-label="LinkedIn">
                            <svg className="linkedin">
                              <use xlinkHref="#linkedin" />
                            </svg>
                          </a>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="col-lg-2 col-sm-6 pb-3">
                  <div className="footer-menu text-capitalize">
                    <h5 className="widget-title pb-2">Quick Links</h5>
                    <ul className="menu-list list-unstyled text-capitalize">
                      <li className="menu-item mb-1">
                        <Link href="/">Home</Link>
                      </li>
                      <li className="menu-item mb-1">
                        <Link href="/top-picks">Top picks</Link>
                      </li>
                      <li className="menu-item mb-1">
                        <Link href="/collection">My collection</Link>
                      </li>
                      <li className="menu-item mb-1">
                        <Link href="/cart">My cart</Link>
                      </li>
                      <li className="menu-item mb-1">
                        <Link href="/profile">Profile</Link>
                      </li>
                    </ul>
                  </div>
                </div>
                <div className="col-lg-3 col-sm-6 pb-3">
                  <div className="footer-menu text-capitalize">
                    <h5 className="widget-title pb-2">Help &amp; Info</h5>
                    <ul className="menu-list list-unstyled">
                      <li className="menu-item mb-1">
                        <Link href="/login">Log-in</Link>
                      </li>
                      <li className="menu-item mb-1">
                        <a href="#search-hero">Search books</a>
                      </li>
                    </ul>
                  </div>
                </div>
                <div className="col-lg-3 col-sm-6 pb-3">
                  <div className="footer-menu contact-item">
                    <h5 className="widget-title text-capitalize pb-2">
                      Contact Us
                    </h5>
                    <p>
                      Questions about recommendations? Explore UrFavBook and
                      keep reading.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </footer>
      <hr />
      <div id="footer-bottom" className="mb-2">
        <div className="container">
          <div className="d-flex flex-wrap justify-content-between">
            <div className="ship-and-payment d-flex gap-md-5 flex-wrap">
              <div className="shipping d-flex">
                <p>We ship with:</p>
                <div className="card-wrap ps-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/bookly/dhl.png" alt="DHL" />
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/bookly/shippingcard.png" alt="shipping" />
                </div>
              </div>
              <div className="payment-method d-flex">
                <p>Payment options:</p>
                <div className="card-wrap ps-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/bookly/visa.jpg" alt="visa" />
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/bookly/mastercard.jpg" alt="mastercard" />
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/bookly/paypal.jpg" alt="paypal" />
                </div>
              </div>
            </div>
            <div className="copyright">
              <p>
                © Copyright 2024 UrFavBook. HTML Template by{" "}
                <a
                  href="https://templatesjungle.com/"
                  target="_blank"
                  rel="noreferrer"
                >
                  TemplatesJungle
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
