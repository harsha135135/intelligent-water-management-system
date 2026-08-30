# Waltr ↔ Water-Forecast Integration Spec

Contract document handed to the Waltr platform team. Everything the forecast extension needs from Waltr, and everything Waltr needs from us, lives here.

## 1. Topology

```
browser ──► app.waltr.in  (Waltr ingress / nginx)
                │
                ├── /*             → existing Waltr app
                └── /forecast/*    → water-forecast-web:3000   (this extension)
                                         │
                                         └── internal docker network
                                                   │
                                                   └── water-forecast-api:8000 (private)
```

Same origin. No CORS. Waltr's session/JWT auto-flows to `/forecast/*` because the browser already sent it to `app.waltr.in`.

## 2. What we need from Waltr

### 2.1 Ingress slot
Add to Waltr's nginx (or equivalent ingress):

```nginx
location /forecast/ {
    proxy_pass http://water-forecast-web:3000/;
    proxy_http_version 1.1;
    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Authorization     $http_authorization;
    proxy_set_header Cookie            $http_cookie;
    proxy_read_timeout 120s;
}
```

The `water-forecast-web` hostname resolves on the shared docker network created in `deploy/docker-compose.yml`. If Waltr runs on a separate network, attach both compose projects to a shared external network and we'll rename accordingly.

### 2.2 JWKS endpoint
- **URL**: e.g. `https://auth.waltr.in/.well-known/jwks.json` — we set this as `WALTR_JWKS_URL`.
- **Algorithm**: RS256 only. ES256 acceptable if you prefer; lock it to one.
- **Rotation**: JWKS cache in our Node tier refreshes on `kid` miss. No action needed from Waltr on rotation beyond publishing the new key.

### 2.3 JWT claim contract
Required claims on every user token forwarded to `/forecast/*`:

| Claim         | Type   | Notes                                                          |
|---------------|--------|----------------------------------------------------------------|
| `sub`         | string | Waltr user id. Used for audit log.                             |
| `role`        | string | `"user"` or `"admin"`. Admin unlocks retrain/sync.             |
| `location_id` | number | Waltr location. We filter tanks against `WALTR_DEFAULT_LOCATION_ID` (638). |
| `iss`         | string | Must match `WALTR_JWT_ISSUER`.                                 |
| `aud`         | string | Must match `WALTR_JWT_AUDIENCE`.                               |
| `exp`, `iat`  | number | Standard.                                                      |

If `role` lives under a namespaced claim (e.g. `https://waltr.in/role`), tell us the exact key and we'll adapt `web/lib/waltr-auth.ts`.

### 2.4 Service-account token
For hourly scheduled sync (Celery beat → `api.waltr.in`), we need a long-lived service JWT:
- Scope: read-only on `/v1/location/638/tanks` and historical readings.
- Delivered into PESU secret store as `WALTR_SERVICE_TOKEN`.
- Rotation cadence: quarterly is fine. Notify ops 7 days before expiry.

### 2.5 Shared internal HMAC secret
Between our own `web` and `api` containers (not Waltr). Waltr team does not need this; listed here only so no one is surprised by the `x-internal-signature` header inside our network.

## 3. What Waltr gets from us

- A single docker image pair: `ghcr.io/pesu/water-forecast-web:<tag>` and `water-forecast-api:<tag>`.
- `deploy/docker-compose.yml` brings up web, api, worker, beat, redis, postgres.
- Read-only page at `/forecast` for every authenticated Waltr user.
- Admin-gated page at `/forecast/admin` for retrain + manual sync.
- Health endpoint: `GET /forecast/api/health` → 200 when the Python tier is reachable.

## 4. Role gating

| Route                          | Who        | Enforced where                              |
|--------------------------------|------------|---------------------------------------------|
| `GET  /forecast`               | any JWT    | `web/middleware.ts`                         |
| `POST /forecast/api/forecast`  | any JWT    | Next route handler → `requireRole('user')`  |
| `POST /forecast/api/retrain`   | admin      | Next route handler + FastAPI `require_admin`|
| `POST /forecast/api/sync`      | admin      | Next route handler + FastAPI `require_admin`|

Defence in depth: the FastAPI layer re-checks `x-user-role` set by the Node tier, and rejects anything missing a valid `x-internal-signature`.

## 5. Observability handoff

- Logs: `docker compose logs -f web api worker` on the PESU host. JSON lines, `structlog` format on Python side.
- Metrics: `GET /metrics` on api (Prometheus text). Waltr's Prom can scrape if desired.
- Sentry DSN (optional): set `SENTRY_DSN` in `.env`; errors tagged `service=water-forecast-{web,api}`.

## 6. Failure modes Waltr should know about

- **AutoGluon predictor load fails** → API returns saved-predictions fallback with a warning flag. UI shows a banner; no 5xx.
- **Waltr sync 401** → we stop beat, alert ops, page stays up serving cached forecasts.
- **Retrain in progress** → `/forecast/admin` shows streaming log. Retrain runs in Celery worker; web tier stays responsive.

## 7. Open items to confirm with Waltr team

- [ ] Exact JWKS URL and `iss`/`aud` values.
- [ ] Claim key for role (flat `role` vs namespaced).
- [ ] Network name to share between Waltr ingress and our compose project.
- [ ] Service-account JWT issuance + rotation owner.
- [ ] Branding tokens (colors, font) — optional for v1.

Point of contact on our side: harshavardhan1305h@gmail.com.
