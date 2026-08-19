import type { Metadata } from "next";
import NavBar from "../components/NavBar";
import "../styles/globals.css";

export const metadata: Metadata = {
  title: "BookCrossing Recommendations",
  description: "Book-Crossing recommendation UI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <NavBar />
        <main>{children}</main>
      </body>
    </html>
  );
}
