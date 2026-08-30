import { NextRequest, NextResponse } from 'next/server';
import { requireRole, verifyWaltrJwt } from '@/lib/waltr-auth';
import { callUpstream } from '@/lib/upstream';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ taskId: string }> },
) {
  const user = await verifyWaltrJwt(req);
  const gate = requireRole(user, 'admin');
  if (gate instanceof Response) return gate;

  const { taskId } = await params;
  const { status, data } = await callUpstream<Record<string, unknown>>({
    method: 'GET',
    path: `/task/${encodeURIComponent(taskId)}`,
    user: gate,
  });
  return NextResponse.json(data, { status });
}
