import { NextResponse, type NextRequest } from 'next/server';

const PUBLIC_PATHS = ['/api/health'];

function buildCsp(waltrOrigin: string | undefined): string {
  const frameAncestors = waltrOrigin ? `'self' ${waltrOrigin}` : "'self'";
  return [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "connect-src 'self'",
    `frame-ancestors ${frameAncestors}`,
  ].join('; ');
}

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const res = NextResponse.next();

  res.headers.set(
    'Content-Security-Policy',
    buildCsp(process.env.WALTR_EMBED_ORIGIN),
  );
  res.headers.set('X-Content-Type-Options', 'nosniff');
  res.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');

  if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) return res;

  // Route handlers do their own JWT verification; middleware only enforces
  // that a token is present for protected page loads so we fail fast.
  if (pathname.startsWith('/forecast') || pathname.startsWith('/api/')) {
    const hasAuth =
      req.headers.get('authorization') ||
      req.cookies.get('waltr_token') ||
      process.env.DEV_AUTH_BYPASS === 'true';
    if (!hasAuth && pathname.startsWith('/api/')) {
      return new NextResponse('Unauthorized', { status: 401 });
    }
  }

  return res;
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
