from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import AnomalyPreviewRequest
from ..services import anomaly_clean_preview, compare_anomaly_model_runs

router = APIRouter(tags=["anomaly"])


@router.post("/anomaly-preview")
def anomaly_preview(payload: AnomalyPreviewRequest) -> dict:
    if not payload.tank_id.strip():
        raise HTTPException(status_code=400, detail="tank_id is required")

    try:
        return anomaly_clean_preview(
            tank_id=payload.tank_id,
            hours=payload.hours,
            rolling_window=payload.rolling_window,
            z_threshold=payload.z_threshold,
            anomaly_context_hours=payload.anomaly_context_hours,
            practical_max_kl=payload.practical_max_kl,
            practical_max_multiplier=payload.practical_max_multiplier,
            jump_z_threshold=payload.jump_z_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/model-run-comparison")
def model_run_comparison() -> dict:
    return {"status": "ok", "comparison": compare_anomaly_model_runs()}
