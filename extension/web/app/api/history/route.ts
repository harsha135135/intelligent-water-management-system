import { NextRequest, NextResponse } from 'next/server';
import { requireRole, verifyWaltrJwt } from '@/lib/waltr-auth';
import { callUpstream } from '@/lib/upstream';
import type { HistoryResponse } from '@/lib/types';

export async function GET(req: NextRequest) {
  const user = await verifyWaltrJwt(req);
  const gate = requireRole(user, 'user');
  if (gate instanceof Response) return gate;

  const tank_id = req.nextUrl.searchParams.get('tank_id');
  const hours = req.nextUrl.searchParams.get('hours') ?? '168';
  if (!tank_id) return NextResponse.json({ error: 'tank_id required' }, { status: 400 });

  const { status, data } = await callUpstream<HistoryResponse>({
    method: 'GET',
    path: '/history',
    user: gate,
    searchParams: { tank_id, hours },
  });
  return NextResponse.json(data, { status });
}
