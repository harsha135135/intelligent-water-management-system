from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import install_hmac_middleware

from .conftest import sign_request


def _make_app() -> FastAPI:
    app = FastAPI()
    install_hmac_middleware(app)

    @app.get("/echo")
    def echo():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def test_health_exempt_from_hmac(hmac_secret):
    client = TestClient(_make_app())
    resp = client.get("/health")
    assert resp.status_code == 200


def test_missing_signature_rejected(hmac_secret):
    client = TestClient(_make_app())
    resp = client.get("/echo")
    assert resp.status_code == 401


def test_bad_signature_rejected(hmac_secret):
    client = TestClient(_make_app())
    resp = client.get(
        "/echo",
        headers={"x-internal-signature": "deadbeef", "x-internal-timestamp": "1"},
    )
    assert resp.status_code == 401


def test_valid_signature_accepted(hmac_secret):
    client = TestClient(_make_app())
    headers = sign_request(hmac_secret, "GET", "/echo")
    resp = client.get("/echo", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
