'use client';

import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import type { ForecastResponse, TanksResponse } from '@/lib/types';
import { ForecastChart } from './ForecastChart';

const ALL_MODELS = ['autogluon', 'patchtst', 'anomaly_ensemble'] as const;

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { ...init, headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) } });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export function ForecastDashboard() {
  const tanksQ = useQuery({
    queryKey: ['tanks'],
    queryFn: () => fetchJSON<TanksResponse>('/api/tanks'),
  });

  const [tankId, setTankId] = useState<string>('');
  const [horizon, setHorizon] = useState(24);
  const [models, setModels] = useState<string[]>(['autogluon', 'patchtst']);

  const selectedTank = tankId || tanksQ.data?.tanks[0] || '';

  const forecast = useMutation({
    mutationFn: (body: { tank_id: string; prediction_length: number; model_keys: string[] }) =>
      fetchJSON<ForecastResponse>('/api/forecast', { method: 'POST', body: JSON.stringify(body) }),
  });

  const runForecast = () => {
    if (!selectedTank) return;
    forecast.mutate({ tank_id: selectedTank, prediction_length: horizon, model_keys: models });
  };

  const chartData = useMemo(() => forecast.data, [forecast.data]);

  return (
    <div className="space-y-6">
      <section className="grid gap-4 rounded-lg bg-waltr-panel p-4 md:grid-cols-[2fr_1fr_2fr_auto]">
        <label className="flex flex-col text-sm">
          <span className="mb-1 text-slate-400">Tank</span>
          <select
            className="rounded bg-slate-900 p-2"
            value={selectedTank}
            onChange={(e) => setTankId(e.target.value)}
            disabled={tanksQ.isLoading}
          >
            {tanksQ.data?.tanks.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col text-sm">
          <span className="mb-1 text-slate-400">Horizon (h)</span>
          <input
            type="number"
            min={1}
            max={48}
            value={horizon}
            onChange={(e) => setHorizon(Math.max(1, Math.min(48, Number(e.target.value))))}
            className="rounded bg-slate-900 p-2"
          />
        </label>

        <fieldset className="flex flex-col text-sm">
          <legend className="mb-1 text-slate-400">Models</legend>
          <div className="flex flex-wrap gap-3">
            {ALL_MODELS.map((m) => (
              <label key={m} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={models.includes(m)}
                  onChange={(e) =>
                    setModels((prev) =>
                      e.target.checked ? [...prev, m] : prev.filter((x) => x !== m),
                    )
                  }
                />
                <span>{m}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <button
          onClick={runForecast}
          disabled={!selectedTank || forecast.isPending}
          className="self-end rounded bg-waltr-accent px-4 py-2 font-semibold text-white disabled:opacity-50"
        >
          {forecast.isPending ? 'Running…' : 'Forecast'}
        </button>
      </section>

      {forecast.error && (
        <div className="rounded bg-red-950 p-3 text-sm text-red-200">
          {(forecast.error as Error).message}
        </div>
      )}

      {chartData && (
        <section className="rounded-lg bg-waltr-panel p-4">
          <ForecastChart data={chartData} />
          {chartData.warnings.length > 0 && (
            <ul className="mt-3 list-disc pl-5 text-xs text-amber-300">
              {chartData.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
          <dl className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-400 md:grid-cols-4">
            {Object.entries(chartData.forecast_sources).map(([k, v]) => (
              <div key={k}>
                <dt className="inline font-semibold">{k}:</dt>{' '}
                <dd className="inline">{v}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
    </div>
  );
}
