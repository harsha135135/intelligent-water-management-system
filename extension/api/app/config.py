from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default=Path(__file__).resolve().parents[2])
    dataset_dir: Path = Field(default=Path("/srv/water/dataset"))
    results_dir: Path = Field(default=Path("/srv/water/results"))
    models_src_dir: Path = Field(default=Path(__file__).resolve().parents[2] / "src" / "models")
    runtime_dir: Path = Field(default=Path("/srv/water/runtime"))

    internal_hmac_secret: str = Field(default="dev-only-change-me")
    hmac_max_skew_seconds: int = 300

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    waltr_base_url: str = "https://api.waltr.in"
    waltr_default_location_id: str = "638"
    waltr_service_token: str = Field(default="")

    sentry_dsn: str = ""
    log_level: str = "INFO"

    cors_allow_origins: list[str] = Field(default_factory=list)


@lru_cache
def get_settings() -> Settings:
    return Settings()
