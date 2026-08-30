import { AdminPanel } from '@/components/AdminPanel';

export default function AdminPage() {
  return (
    <main className="mx-auto max-w-4xl p-8">
      <header className="mb-6 border-b border-slate-800 pb-4">
        <p className="text-xs uppercase tracking-wider text-waltr-accent">Admin</p>
        <h1 className="text-2xl font-bold">Retrain & Data Sync</h1>
        <p className="text-sm text-slate-400">
          Trigger model retraining and Waltr data syncs. Requires admin role.
        </p>
      </header>
      <AdminPanel />
    </main>
  );
}
