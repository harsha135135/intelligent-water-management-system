import { NextRequest, NextResponse } from 'next/server';
import { requireRole, verifyWaltrJwt } from '@/lib/waltr-auth';
import { callUpstream } from '@/lib/upstream';
import type { TaskEnqueued } from '@/lib/types';

export async function POST(req: NextRequest) {
  const user = await verifyWaltrJwt(req);
  const gate = requireRole(user, 'admin');
  if (gate instanceof Response) return gate;

  const body = await req.json().catch(() => ({}));
  const { status, data } = await callUpstream<TaskEnqueued>({
    method: 'POST',
    path: '/sync',
    user: gate,
    body,
  });
  return NextResponse.json(data, { status });
}
