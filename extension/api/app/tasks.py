from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from celery import Celery
from celery.schedules import crontab

from .config import get_settings
from .waltr_sync import sync_waltr_dataset_incremental


_settings = get_settings()

celery_app = Celery(
    "water_forecast",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
)
celery_app.conf.task_track_started = True
celery_app.conf.task_time_limit = 60 * 60 * 6  # 6h hard cap
celery_app.conf.worker_prefetch_multiplier = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retrain_log_file() -> Path:
    log_file = _settings.runtime_dir / "retrain_model.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


def _append_log(msg: str) -> None:
    with _retrain_log_file().open("a", encoding="utf-8") as f:
        f.write(msg if msg.endswith("\n") else msg + "\n")


def _promote_staged_model(staged: Path, final: Path) -> None:
    """Atomic swap: promote staged dir to active, backup+drop previous."""
    if not (staged / "predictor.pkl").exists():
        _append_log(f"[{_utc_now_iso()}] Staged model missing predictor.pkl at {staged}; keeping previous active model.")
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        return

    backup = final.parent / f"{final.name}_backup_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    try:
        if final.exists():
            final.rename(backup)
        staged.rename(final)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        _append_log(f"[{_utc_now_iso()}] Promoted staged model to active: {final}")
    except Exception as exc:
        if not final.exists() and backup.exists():
            backup.rename(final)
        _append_log(f"[{_utc_now_iso()}] Model promotion failed, kept previous active model: {exc}")


@celery_app.task(bind=True, name="water_forecast.retrain_model")
def retrain_model_task(
    self,
    model_key: str,
    final_model_dir: str,
    prediction_length: int = 24,
    time_limit: int = 1800,
    target_col: str = "Outflow in KL",
    min_history_hours: int = 24 * 14,
    presets: str = "high_quality",
    max_epochs: int = 200,
    rolling_window: int = 24,
    z_threshold: float = 4.5,
    practical_max_kl: float | None = None,
    practical_max_multiplier: float = 1.3,
    jump_z_threshold: float = 3.0,
    anomaly_context_hours: int = 24 * 180,
    training_profile: str = "heavy",
    num_val_windows: int = 3,
    refit_full: bool = True,
    enable_deep_models: bool = True,
) -> dict[str, Any]:
    final_path = Path(final_model_dir).resolve()
    staged = (
        final_path.parent
        / "_retrain_staging"
        / f"{model_key}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    staged.parent.mkdir(parents=True, exist_ok=True)

    models_src = _settings.models_src_dir
    if model_key == "autogluon":
        script = models_src / "autogluon_hourly_forecasting.py"
        cmd = [
            sys.executable, str(script),
            "--dataset-dir", str(_settings.dataset_dir),
            "--model-dir", str(staged),
            "--prediction-length", str(prediction_length),
            "--time-limit", str(time_limit),
            "--target-col", target_col,
            "--min-history-hours", str(min_history_hours),
            "--presets", presets,
        ]
    elif model_key == "patchtst":
        script = models_src / "patchtst_hourly_forecasting.py"
        cmd = [
            sys.executable, str(script),
            "--dataset-dir", str(_settings.dataset_dir),
            "--model-dir", str(staged),
            "--prediction-length", str(prediction_length),
            "--time-limit", str(time_limit),
            "--target-col", target_col,
            "--min-history-hours", str(min_history_hours),
            "--max-epochs", str(max_epochs),
            "--presets", presets,
        ]
    elif model_key == "anomaly_ensemble":
        script = models_src / "anomaly_aware_ensemble_forecasting.py"
        cmd = [
            sys.executable, str(script),
            "--dataset-dir", str(_settings.dataset_dir),
            "--model-dir", str(staged),
            "--prediction-length", str(prediction_length),
            "--time-limit", str(time_limit),
            "--target-col", target_col,
            "--rolling-window", str(rolling_window),
            "--z-threshold", str(z_threshold),
            "--practical-max-multiplier", str(practical_max_multiplier),
            "--jump-z-threshold", str(jump_z_threshold),
            "--min-history-hours", str(min_history_hours),
            "--anomaly-context-hours", str(anomaly_context_hours),
            "--max-epochs", str(max_epochs),
            "--training-profile", training_profile,
            "--num-val-windows", str(num_val_windows),
            "--presets", presets,
        ]
        if refit_full:
            cmd.append("--refit-full")
        if enable_deep_models:
            cmd.append("--enable-deep-models")
        if practical_max_kl is not None:
            cmd.extend(["--practical-max-kl", str(practical_max_kl)])
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}")

    _append_log(f"\n[{_utc_now_iso()}] Starting retrain: {model_key}")
    _append_log(f"Active: {final_path}")
    _append_log(f"Staged: {staged}")
    _append_log("Command: " + " ".join(cmd))

    with _retrain_log_file().open("a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_settings.project_root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        proc.wait()
        exit_code = proc.returncode

    if exit_code == 0:
        _promote_staged_model(staged, final_path)
    else:
        _append_log(f"[{_utc_now_iso()}] Retrain exited {exit_code}; discarding staged dir.")
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)

    return {
        "model_key": model_key,
        "exit_code": exit_code,
        "final_model_dir": str(final_path),
        "staged_model_dir": str(staged),
    }


@celery_app.task(name="water_forecast.sync_waltr")
def sync_waltr_task(
    token: str | None = None,
    location_id: str | None = None,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
) -> dict[str, Any]:
    resolved_token = token or _settings.waltr_service_token
    if not resolved_token:
        raise ValueError("No Waltr token provided (arg or WALTR_SERVICE_TOKEN env)")

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date else date.today()

    summary = sync_waltr_dataset_incremental(
        token=resolved_token,
        dataset_dir=_settings.dataset_dir,
        location_id=location_id or _settings.waltr_default_location_id,
        start_date=start,
        end_date=end,
    )
    return summary


@celery_app.task(name="water_forecast.smoke_forecast")
def smoke_forecast_task() -> dict[str, Any]:
    """Daily assertion that every available model still predicts."""
    from .services import forecast_next_24h, list_tank_ids, model_availability

    tanks = list_tank_ids()
    if not tanks:
        return {"status": "no_tanks"}

    target_tank = tanks[0]
    available = [k for k, v in model_availability().items() if v["available"]]
    if not available:
        return {"status": "no_models"}

    result = forecast_next_24h(
        tank_id=target_tank,
        model_keys=available,
        prediction_length=24,
    )
    return {
        "status": "ok",
        "tank_id": target_tank,
        "models": available,
        "warnings": result.get("warnings", []),
    }


celery_app.conf.beat_schedule = {
    "hourly-waltr-sync": {
        "task": "water_forecast.sync_waltr",
        "schedule": crontab(minute=5),
    },
    "daily-smoke-forecast": {
        "task": "water_forecast.smoke_forecast",
        "schedule": crontab(hour=3, minute=0),
    },
}
celery_app.conf.timezone = "UTC"
