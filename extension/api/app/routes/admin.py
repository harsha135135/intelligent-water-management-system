from __future__ import annotations

from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_admin
from ..schemas import (
    RetrainRequest,
    SyncRequest,
    TaskEnqueuedResponse,
)
from ..services import (
    ANOMALY_BASELINE_DIR,
    ANOMALY_HEAVY_DIR,
    MODEL_REGISTRY,
    refresh_runtime_cache,
)
from ..tasks import celery_app, retrain_model_task, sync_waltr_task

router = APIRouter(tags=["admin"])


@router.post("/refresh")
def refresh(_: str = Depends(require_admin)) -> dict:
    cache_info = refresh_runtime_cache()
    return {"status": "ok", "cache": cache_info}


@router.post("/retrain", response_model=TaskEnqueuedResponse)
def retrain(payload: RetrainRequest, _: str = Depends(require_admin)) -> TaskEnqueuedResponse:
    if payload.model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unsupported model_key: {payload.model_key}")
    if payload.prediction_length <= 0 or payload.time_limit <= 0 or payload.min_history_hours <= 0:
        raise HTTPException(status_code=400, detail="Invalid parameters for retraining request.")

    if payload.model_dir:
        final_dir = Path(payload.model_dir).resolve()
    elif payload.model_key == "anomaly_ensemble":
        final_dir = ANOMALY_HEAVY_DIR if payload.training_profile == "heavy" else ANOMALY_BASELINE_DIR
    else:
        final_dir = MODEL_REGISTRY[payload.model_key]["path"]

    kwargs = payload.model_dump()
    kwargs.pop("model_dir", None)
    kwargs["final_model_dir"] = str(final_dir)

    async_result = retrain_model_task.apply_async(kwargs=kwargs)
    return TaskEnqueuedResponse(task_id=async_result.id, task_name="retrain_model")


@router.post("/sync", response_model=TaskEnqueuedResponse)
def sync(payload: SyncRequest, _: str = Depends(require_admin)) -> TaskEnqueuedResponse:
    async_result = sync_waltr_task.apply_async(kwargs=payload.model_dump())
    return TaskEnqueuedResponse(task_id=async_result.id, task_name="sync_waltr")


@router.get("/task/{task_id}")
def task_status(task_id: str, _: str = Depends(require_admin)) -> dict:
    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "state": result.state,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "result": result.result if result.ready() else None,
    }
