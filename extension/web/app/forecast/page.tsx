import { ForecastDashboard } from '@/components/ForecastDashboard';

interface PageProps {
  searchParams: Promise<{ embed?: string }>;
}

export default async function ForecastPage({ searchParams }: PageProps) {
  const { embed } = await searchParams;
  const isEmbedded = embed === '1';

  return (
    <main className={isEmbedded ? 'p-4' : 'mx-auto max-w-6xl p-8'}>
      {!isEmbedded && (
        <header className="mb-6 border-b border-slate-800 pb-4">
          <p className="text-xs uppercase tracking-wider text-waltr-accent">Extension Module</p>
          <h1 className="text-2xl font-bold">Water Forecast Dashboard</h1>
          <p className="text-sm text-slate-400">
            24-hour per-tank demand forecasts across AutoGluon + PatchTST.
          </p>
        </header>
      )}
      <ForecastDashboard />
    </main>
  );
}
