'use client';

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ForecastResponse } from '@/lib/types';

type Row = {
  timestamp: string;
  actual?: number;
} & Record<string, number | string | undefined>;

const MODEL_COLORS: Record<string, string> = {
  autogluon: '#3b82f6',
  patchtst: '#a855f7',
  anomaly_ensemble: '#10b981',
};

export function ForecastChart({ data }: { data: ForecastResponse }) {
  const rows = new Map<string, Row>();

  for (const h of data.history) {
    rows.set(h.timestamp, { timestamp: h.timestamp, actual: h['Outflow in KL'] });
  }
  for (const [model, points] of Object.entries(data.forecasts)) {
    for (const p of points) {
      const row = rows.get(p.timestamp) ?? { timestamp: p.timestamp };
      row[model] = p.pred_mean;
      rows.set(p.timestamp, row);
    }
  }

  const series = Array.from(rows.values()).sort((a, b) =>
    a.timestamp.localeCompare(b.timestamp),
  );

  return (
    <div className="h-96 w-full">
      <ResponsiveContainer>
        <LineChart data={series} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" stroke="#64748b" tick={{ fontSize: 10 }} minTickGap={40} />
          <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }}
            labelStyle={{ color: '#e2e8f0' }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#e2e8f0"
            strokeWidth={1.5}
            dot={false}
            name="Actual"
          />
          {Object.keys(data.forecasts).map((m) => (
            <Line
              key={m}
              type="monotone"
              dataKey={m}
              stroke={MODEL_COLORS[m] ?? '#f59e0b'}
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              name={m}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
