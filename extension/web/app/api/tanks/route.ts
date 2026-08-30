import { NextRequest, NextResponse } from 'next/server';
import { requireRole, verifyWaltrJwt } from '@/lib/waltr-auth';
import { callUpstream } from '@/lib/upstream';
import type { TanksResponse } from '@/lib/types';

export async function GET(req: NextRequest) {
  const user = await verifyWaltrJwt(req);
  const gate = requireRole(user, 'user');
  if (gate instanceof Response) return gate;

  const { status, data } = await callUpstream<TanksResponse>({
    method: 'GET',
    path: '/tanks',
    user: gate,
  });
  return NextResponse.json(data, { status });
}
