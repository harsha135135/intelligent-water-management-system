import { signUpstream } from './server-token';
import type { AuthedUser } from './types';

function apiBase(): string {
  return process.env.API_BASE_URL ?? 'http://api:8000';
}

type Method = 'GET' | 'POST';

export interface UpstreamOptions {
  method: Method;
  path: string;
  user: AuthedUser;
  body?: unknown;
  searchParams?: Record<string, string | number | undefined>;
}

export async function callUpstream<T>(opts: UpstreamOptions): Promise<{ status: number; data: T }> {
  const base = apiBase().replace(/\/$/, '');
  const qs = opts.searchParams
    ? '?' +
      Object.entries(opts.searchParams)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join('&')
    : '';

  const bodyText = opts.body !== undefined ? JSON.stringify(opts.body) : '';
  const sig = signUpstream(opts.method, opts.path, bodyText);

  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'x-user-sub': opts.user.sub,
    'x-user-role': opts.user.role,
    ...sig,
  };

  const resp = await fetch(`${base}${opts.path}${qs}`, {
    method: opts.method,
    headers,
    body: opts.method === 'POST' ? bodyText : undefined,
    cache: 'no-store',
  });

  const text = await resp.text();
  const data = text ? (JSON.parse(text) as T) : (undefined as unknown as T);
  return { status: resp.status, data };
}
