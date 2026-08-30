from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnomalyPreviewRequest,
    ForecastRequest,
    RetrainRequest,
    SyncRequest,
)


def test_forecast_request_defaults():
    req = ForecastRequest(tank_id="BE_BLOCK_OHT")
    assert req.prediction_length == 24
    assert req.model_keys is None


def test_retrain_request_defaults():
    req = RetrainRequest()
    assert req.model_key == "autogluon"
    assert req.presets == "high_quality"
    assert req.refit_full is True


def test_anomaly_request_requires_tank():
    with pytest.raises(ValidationError):
        AnomalyPreviewRequest()  # type: ignore[call-arg]


def test_sync_request_optional_fields():
    req = SyncRequest()
    assert req.token is None
    assert req.start_date == "2025-01-01"
