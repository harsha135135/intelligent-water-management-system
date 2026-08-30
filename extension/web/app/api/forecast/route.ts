import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { requireRole, verifyWaltrJwt } from '@/lib/waltr-auth';
import { callUpstream } from '@/lib/upstream';
import type { ForecastResponse } from '@/lib/types';

const schema = z.object({
  tank_id: z.string().min(1),
  prediction_length: z.number().int().positive().max(48).default(24),
  model_keys: z.array(z.string()).optional(),
});

export async function POST(req: NextRequest) {
  const user = await verifyWaltrJwt(req);
  const gate = requireRole(user, 'user');
  if (gate instanceof Response) return gate;

  const parsed = schema.safeParse(await req.json().catch(() => ({})));
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.format() }, { status: 400 });
  }

  const { status, data } = await callUpstream<ForecastResponse>({
    method: 'POST',
    path: '/forecast',
    user: gate,
    body: parsed.data,
  });
  return NextResponse.json(data, { status });
}
