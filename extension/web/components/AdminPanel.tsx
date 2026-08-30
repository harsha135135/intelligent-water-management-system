'use client';

import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import type { TaskEnqueued } from '@/lib/types';

type TaskState = { state: string; ready: boolean; result?: unknown };

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { ...init, headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) } });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export function AdminPanel() {
  const [modelKey, setModelKey] = useState<'autogluon' | 'patchtst' | 'anomaly_ensemble'>('autogluon');
  const [timeLimit, setTimeLimit] = useState(1800);
  const [activeTask, setActiveTask] = useState<string | null>(null);

  const retrain = useMutation({
    mutationFn: () =>
      fetchJSON<TaskEnqueued>('/api/retrain', {
        method: 'POST',
        body: JSON.stringify({ model_key: modelKey, time_limit: timeLimit }),
      }),
    onSuccess: (d) => setActiveTask(d.task_id),
  });

  const sync = useMutation({
    mutationFn: () =>
      fetchJSON<TaskEnqueued>('/api/sync', {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    onSuccess: (d) => setActiveTask(d.task_id),
  });

  const taskQ = useQuery<TaskState>({
    queryKey: ['task', activeTask],
    queryFn: () => fetchJSON<TaskState>(`/api/task/${activeTask}`),
    enabled: !!activeTask,
    refetchInterval: (q) => (q.state.data?.ready ? false : 3000),
  });

  return (
    <div className="space-y-6">
      <section className="rounded-lg bg-waltr-panel p-4">
        <h2 className="mb-3 font-semibold">Retrain model</h2>
        <div className="flex flex-wrap gap-3">
          <select
            className="rounded bg-slate-900 p-2"
            value={modelKey}
            onChange={(e) => setModelKey(e.target.value as typeof modelKey)}
          >
            <option value="autogluon">AutoGluon</option>
            <option value="patchtst">PatchTST</option>
            <option value="anomaly_ensemble">Anomaly Ensemble</option>
          </select>
          <label className="flex items-center gap-2 text-sm">
            <span className="text-slate-400">time_limit(s)</span>
            <input
              type="number"
              min={300}
              value={timeLimit}
              onChange={(e) => setTimeLimit(Number(e.target.value))}
              className="w-28 rounded bg-slate-900 p-2"
            />
          </label>
          <button
            onClick={() => retrain.mutate()}
            disabled={retrain.isPending}
            className="rounded bg-waltr-accent px-4 py-2 font-semibold text-white disabled:opacity-50"
          >
            {retrain.isPending ? 'Queueing…' : 'Queue retrain'}
          </button>
        </div>
      </section>

      <section className="rounded-lg bg-waltr-panel p-4">
        <h2 className="mb-3 font-semibold">Waltr sync</h2>
        <button
          onClick={() => sync.mutate()}
          disabled={sync.isPending}
          className="rounded bg-waltr-accent px-4 py-2 font-semibold text-white disabled:opacity-50"
        >
          {sync.isPending ? 'Queueing…' : 'Queue sync'}
        </button>
      </section>

      {activeTask && (
        <section className="rounded-lg bg-waltr-panel p-4">
          <h2 className="mb-3 font-semibold">Task {activeTask}</h2>
          <pre className="overflow-auto text-xs text-slate-300">
            {JSON.stringify(taskQ.data ?? { state: 'loading…' }, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
