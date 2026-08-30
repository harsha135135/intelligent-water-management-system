# Deploy

Local smoke test and on-prem bring-up for the water-forecast extension.

## Prereqs
- Docker 24+ and Docker Compose v2.
- `DATA_ROOT` populated with `dataset/`, `results/`, and a writable `runtime/` subtree. On a fresh host:
  ```bash
  sudo mkdir -p /srv/water
  sudo chown -R 10001:10001 /srv/water
  rsync -a ../dataset  /srv/water/
  rsync -a ../results  /srv/water/
  rsync -a ../src      /srv/water/
  mkdir -p /srv/water/runtime
  ```
  The container runs as UID 10001; paths must be readable/writable by that uid.

## Local smoke test
```bash
cp .env.example .env          # edit secrets + DATA_ROOT
docker compose --env-file .env up -d --build
docker compose ps
curl -i http://localhost/forecast             # proxied through nginx -> web
```
Set `DEV_AUTH_BYPASS=true` in `.env` and send `x-dev-role: admin` to exercise admin routes without a real Waltr JWT.

## Production bring-up (PESU on-prem)
1. Provision Ubuntu 22.04 VM, install Docker, open 443 inbound from the college LAN.
2. `git clone` the repo to `/opt/water-forecast`, `cd deploy`.
3. Write `/etc/water-forecast/.env` from the secret store; symlink `ln -s /etc/water-forecast/.env .env`.
4. Issue TLS: `certbot certonly --webroot -w /var/www/certbot -d forecast.pes.edu`, copy chain+key to `$TLS_CERT_DIR`, uncomment the 443 block in `nginx.conf`.
5. `export IMAGE_TAG=v1.0.0 && docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env pull && docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env up -d`.
6. Verify: `curl -fsS https://forecast.pes.edu/forecast/api/tanks` through an admin-JWT'd session.

## CI/CD
- `.github/workflows/ci.yml` — lint + test + build both images on every push; pushes `sha-<short>` tags to GHCR on `main`/`test`.
- `.github/workflows/deploy.yml` — on git tag `v*.*.*` (or `workflow_dispatch`) it pushes the release image tags and SSHes into the PESU host to `docker compose pull && up -d`.
- Required GitHub secrets on the `production` environment: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_PORT` (optional), `GHCR_READ_USER`, `GHCR_READ_TOKEN` (read:packages PAT).
- Host prep: `/opt/water-forecast` is a `git clone` of this repo; `/etc/water-forecast/.env` holds prod secrets and is symlinked to `deploy/.env`.

## Rollback
Tags are immutable in GHCR. To roll back:
```bash
export IMAGE_TAG=v1.0.3
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env up -d web api worker beat
```

## Logs & health
- `docker compose logs -f api worker` — inference + retrain logs.
- `curl http://localhost:8000/health` (inside the Docker network) — API health.
- Retrain log tail is served via `GET /api/task/<id>` (admin-only through the web tier).
