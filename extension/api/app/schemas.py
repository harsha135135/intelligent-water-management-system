from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    models: dict[str, dict[str, Any]]
    tank_count: int


class TanksResponse(BaseModel):
    tanks: list[str]


class HistoryResponse(BaseModel):
    tank_id: str
    hours: int
    history: list[dict[str, Any]]


class ForecastRequest(BaseModel):
    tank_id: str
    prediction_length: int = 24
    model_keys: list[str] | None = None


class ForecastResponse(BaseModel):
    tank_id: str
    prediction_length: int
    history: list[dict[str, Any]]
    forecasts: dict[str, list[dict[str, Any]]]
    forecast_sources: dict[str, str]
    warnings: list[str]


class AnomalyPreviewRequest(BaseModel):
    tank_id: str
    hours: int = 24 * 7
    rolling_window: int = 24
    z_threshold: float = 4.5
    anomaly_context_hours: int = 24 * 180
    practical_max_kl: float | None = None
    practical_max_multiplier: float = 1.3
    jump_z_threshold: float = 3.0


class RetrainRequest(BaseModel):
    model_key: str = "autogluon"
    prediction_length: int = 24
    time_limit: int = 1800
    target_col: str = "Outflow in KL"
    min_history_hours: int = 24 * 14
    presets: str = "high_quality"
    max_epochs: int = 200
    model_dir: str | None = None
    rolling_window: int = 24
    z_threshold: float = 4.5
    practical_max_kl: float | None = None
    practical_max_multiplier: float = 1.3
    jump_z_threshold: float = 3.0
    anomaly_context_hours: int = 24 * 180
    training_profile: str = "heavy"
    num_val_windows: int = 3
    refit_full: bool = True
    enable_deep_models: bool = True


class SyncRequest(BaseModel):
    token: str | None = None
    location_id: str | None = None
    start_date: str = "2025-01-01"
    end_date: str | None = None


class TaskEnqueuedResponse(BaseModel):
    status: str = "queued"
    task_id: str
    task_name: str
