from __future__ import annotations

import logging

import sentry_sdk
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from sentry_sdk.integrations.fastapi import FastApiIntegration

from .auth import install_hmac_middleware
from .config import get_settings
from .routes import admin, anomaly, forecast
from .services import list_tank_ids, model_availability


def create_app() -> FastAPI:
    settings = get_settings()

    logging.basicConfig(level=settings.log_level)

    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.05,
        )

    app = FastAPI(
        title="Water Forecast API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    install_hmac_middleware(app)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    app.include_router(forecast.router)
    app.include_router(anomaly.router)
    app.include_router(admin.router)

    @app.get("/health", include_in_schema=True)
    def health() -> dict:
        try:
            tanks = list_tank_ids()
            tank_count = len(tanks)
        except Exception:
            tank_count = 0
        return {
            "status": "ok",
            "models": model_availability(),
            "tank_count": tank_count,
        }

    return app


app = create_app()
