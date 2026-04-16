import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ChemAI Research | AI in Chemistry",
  description: "Accelerating chemical discovery through AI-driven research. Exploring Nickel Catalysis and computational chemistry with machine learning.",
  keywords: ["ChemAI", "AI Chemistry", "Nickel Catalysis", "NiCOBot", "Redox Ligands", "Computational Chemistry", "Machine Learning", "AI Research"],
  authors: [{ name: "ChemAI Research Team" }],
  icons: {
    icon: "/logo.svg",
  },
  openGraph: {
    title: "ChemAI Research | AI in Chemistry",
    description: "Accelerating chemical discovery through AI-driven research",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "ChemAI Research | AI in Chemistry",
    description: "Accelerating chemical discovery through AI-driven research",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
