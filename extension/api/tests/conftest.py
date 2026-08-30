from __future__ import annotations

import hashlib
import hmac
import time

import pytest


@pytest.fixture
def hmac_secret(monkeypatch) -> str:
    secret = "test-secret"
    monkeypatch.setenv("INTERNAL_HMAC_SECRET", secret)
    from app.config import get_settings
    get_settings.cache_clear()
    return secret


def sign_request(secret: str, method: str, path: str, body: bytes = b"") -> dict[str, str]:
    timestamp = str(int(time.time()))
    mac = hmac.new(secret.encode(), digestmod=hashlib.sha256)
    mac.update(method.upper().encode())
    mac.update(b"\n")
    mac.update(path.encode())
    mac.update(b"\n")
    mac.update(timestamp.encode())
    mac.update(b"\n")
    mac.update(body)
    return {
        "x-internal-signature": mac.hexdigest(),
        "x-internal-timestamp": timestamp,
    }
