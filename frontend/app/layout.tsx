import type { Metadata } from "next";
import { Nunito } from "next/font/google";
import IconSprites from "../components/bookly/IconSprites";
import SiteFooter from "../components/SiteFooter";
import SiteHeader from "../components/SiteHeader";
import "bootstrap/dist/css/bootstrap.min.css";
import "swiper/css";
import "swiper/css/navigation";
import "../styles/bookly.css";
import "../styles/globals.css";

const nunito = Nunito({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-nunito",
});

export const metadata: Metadata = {
  title: "UrFavBook",
  description: "Personalized book recommendations — UrFavBook",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={nunito.variable}>
      <body className={nunito.className}>
        <IconSprites />
        <SiteHeader />
        <main className="site-main">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
