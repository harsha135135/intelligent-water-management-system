from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..schemas import (
    ForecastRequest,
    ForecastResponse,
    HistoryResponse,
    TanksResponse,
)
from ..services import (
    forecast_next_24h,
    get_recent_history,
    list_tank_ids,
)

router = APIRouter(tags=["forecast"])


@router.get("/tanks", response_model=TanksResponse)
def tanks() -> TanksResponse:
    return TanksResponse(tanks=list_tank_ids())


@router.get("/history", response_model=HistoryResponse)
def history(
    tank_id: str = Query(..., min_length=1),
    hours: int = Query(24 * 7, ge=1, le=24 * 90),
) -> HistoryResponse:
    try:
        df = get_recent_history(tank_id=tank_id, hours=hours)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return HistoryResponse(
        tank_id=tank_id,
        hours=hours,
        history=df.to_dict(orient="records"),
    )


@router.post("/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest) -> ForecastResponse:
    if payload.prediction_length <= 0:
        raise HTTPException(status_code=400, detail="prediction_length must be > 0")
    if not payload.tank_id.strip():
        raise HTTPException(status_code=400, detail="tank_id is required")

    try:
        result = forecast_next_24h(
            tank_id=payload.tank_id,
            model_keys=payload.model_keys,
            prediction_length=payload.prediction_length,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")

    return ForecastResponse(**result)
