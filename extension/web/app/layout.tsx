import type { Metadata } from 'next';
import { Providers } from './providers';
import './globals.css';

export const metadata: Metadata = {
  title: 'Water Forecast — PESU',
  description: '24h water demand forecasts for PESU campus tanks',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-waltr-bg text-slate-200">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
