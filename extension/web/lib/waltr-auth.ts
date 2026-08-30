import { createRemoteJWKSet, jwtVerify, type JWTPayload } from 'jose';
import type { NextRequest } from 'next/server';
import type { AuthedUser, Role } from './types';

let cachedJWKS: ReturnType<typeof createRemoteJWKSet> | null = null;

function getJWKS() {
  const url = process.env.WALTR_JWKS_URL;
  if (!url) return null;
  if (!cachedJWKS) cachedJWKS = createRemoteJWKSet(new URL(url));
  return cachedJWKS;
}

function extractBearer(req: NextRequest): string | null {
  const header = req.headers.get('authorization');
  if (header?.toLowerCase().startsWith('bearer ')) return header.slice(7).trim();
  if (header?.toLowerCase().startsWith('jwt ')) return header.slice(4).trim();
  const cookie = req.cookies.get('waltr_token')?.value;
  return cookie ?? null;
}

function readRole(payload: JWTPayload): Role {
  const namespaced = process.env.WALTR_ROLE_CLAIM;
  const raw =
    (namespaced ? (payload as Record<string, unknown>)[namespaced] : undefined) ??
    payload.role ??
    (payload as Record<string, unknown>)['https://waltr.in/role'];
  return raw === 'admin' ? 'admin' : 'user';
}

function payloadToUser(payload: JWTPayload): AuthedUser {
  return {
    sub: String(payload.sub ?? 'unknown'),
    role: readRole(payload),
    locationId: payload.location_id ? String(payload.location_id) : undefined,
  };
}

export async function verifyWaltrJwt(req: NextRequest): Promise<AuthedUser | null> {
  const token = extractBearer(req);
  if (!token) {
    if (process.env.DEV_AUTH_BYPASS === 'true' && !process.env.WALTR_JWKS_URL) {
      const role = (req.headers.get('x-dev-role') as Role | null) ?? 'user';
      return { sub: 'dev-user', role: role === 'admin' ? 'admin' : 'user' };
    }
    return null;
  }

  const jwks = getJWKS();
  if (!jwks) {
    if (process.env.DEV_AUTH_BYPASS === 'true') {
      const [, payload] = token.split('.');
      if (!payload) return null;
      try {
        const decoded = JSON.parse(Buffer.from(payload, 'base64url').toString('utf-8'));
        return payloadToUser(decoded);
      } catch {
        return null;
      }
    }
    return null;
  }

  try {
    const { payload } = await jwtVerify(token, jwks, {
      issuer: process.env.WALTR_JWT_ISSUER,
      audience: process.env.WALTR_JWT_AUDIENCE,
      algorithms: ['RS256'],
    });
    return payloadToUser(payload);
  } catch {
    return null;
  }
}

export function requireRole(user: AuthedUser | null, needed: Role): AuthedUser | Response {
  if (!user) return new Response('Unauthorized', { status: 401 });
  if (needed === 'admin' && user.role !== 'admin') {
    return new Response('Forbidden', { status: 403 });
  }
  return user;
}
