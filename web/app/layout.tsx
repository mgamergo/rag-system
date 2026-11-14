import type { Metadata } from "next";
import "./globals.css";
import { ConvexClientProvider } from "../providers/ConvexClientProvider";

export const metadata: Metadata = {
  title: "Field Notes RAG",
  description: "Ask grounded questions over your indexed research notes.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><ConvexClientProvider>{children}</ConvexClientProvider></body></html>;
}
