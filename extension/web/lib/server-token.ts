import { createHmac } from 'node:crypto';

function secret(): string {
  const s = process.env.INTERNAL_HMAC_SECRET;
  if (!s) throw new Error('INTERNAL_HMAC_SECRET is not set');
  return s;
}

export function signUpstream(method: string, path: string, body: string): {
  'x-internal-signature': string;
  'x-internal-timestamp': string;
} {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const mac = createHmac('sha256', secret());
  mac.update(method.toUpperCase());
  mac.update('\n');
  mac.update(path);
  mac.update('\n');
  mac.update(timestamp);
  mac.update('\n');
  mac.update(body);
  return {
    'x-internal-signature': mac.digest('hex'),
    'x-internal-timestamp': timestamp,
  };
}
