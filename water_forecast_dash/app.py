from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from datetime import date
import shutil

from flask import Flask, jsonify, render_template, request

try:
    from .services import (
        MODEL_REGISTRY,
        PROJECT_ROOT,
        ANOMALY_BASELINE_DIR,
        ANOMALY_HEAVY_DIR,
        anomaly_clean_preview,
        compare_anomaly_model_runs,
        refresh_runtime_cache,
        list_tank_ids,
        model_availability,
        forecast_next_24h,
        get_recent_history,
    )
except ImportError:
    from services import (  # type: ignore
        MODEL_REGISTRY,
        PROJECT_ROOT,
        ANOMALY_BASELINE_DIR,
        ANOMALY_HEAVY_DIR,
        anomaly_clean_preview,
        compare_anomaly_model_runs,
        refresh_runtime_cache,
        list_tank_ids,
        model_availability,
        forecast_next_24h,
        get_recent_history,
    )

try:
    from .waltr_sync import sync_waltr_dataset_incremental
except ImportError:
    from waltr_sync import sync_waltr_dataset_incremental  # type: ignore


RETRAIN_LOG_FILE = Path(__file__).resolve().parent / "runtime" / "retrain_model.log"
_RETRAIN_STATE: dict[str, object] = {
    "process": None,
    "log_handle": None,
    "started_at": None,
    "last_exit_code": None,
    "last_command": None,
    "model_key": None,
    "final_model_dir": None,
    "staged_model_dir": None,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _update_retrain_state() -> None:
    process = _RETRAIN_STATE.get("process")
    if process is None:
        return

    assert isinstance(process, subprocess.Popen)
    if process.poll() is not None:
        _RETRAIN_STATE["last_exit_code"] = process.returncode

        staged_model_dir = _RETRAIN_STATE.get("staged_model_dir")
        final_model_dir = _RETRAIN_STATE.get("final_model_dir")

        staged_path = Path(staged_model_dir) if isinstance(staged_model_dir, str) and staged_model_dir else None
        final_path = Path(final_model_dir) if isinstance(final_model_dir, str) and final_model_dir else None

        # Atomic model swap: only successful training promotes staged artifacts.
        if process.returncode == 0 and staged_path is not None and final_path is not None:
            predictor_exists = (staged_path / "predictor.pkl").exists()
            if predictor_exists:
                backup_path = final_path.parent / f"{final_path.name}_backup_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                try:
                    if final_path.exists():
                        final_path.rename(backup_path)
                    staged_path.rename(final_path)
                    if backup_path.exists():
                        shutil.rmtree(backup_path, ignore_errors=True)
                    with RETRAIN_LOG_FILE.open("a", encoding="utf-8") as log_f:
                        log_f.write(f"[{_utc_now_iso()}] Promoted staged model to active: {final_path}\n")
                except Exception as exc:
                    # Roll back if promotion fails.
                    if not final_path.exists() and backup_path.exists():
                        backup_path.rename(final_path)
                    with RETRAIN_LOG_FILE.open("a", encoding="utf-8") as log_f:
                        log_f.write(f"[{_utc_now_iso()}] Model promotion failed, kept previous active model: {exc}\n")
            else:
                with RETRAIN_LOG_FILE.open("a", encoding="utf-8") as log_f:
                    log_f.write(f"[{_utc_now_iso()}] Staged model missing predictor.pkl, keeping previous active model.\n")
                if staged_path.exists():
                    shutil.rmtree(staged_path, ignore_errors=True)
        else:
            if staged_path is not None and staged_path.exists():
                shutil.rmtree(staged_path, ignore_errors=True)
            if process.returncode != 0:
                with RETRAIN_LOG_FILE.open("a", encoding="utf-8") as log_f:
                    log_f.write(f"[{_utc_now_iso()}] Retrain failed/stopped, active model kept unchanged.\n")

        log_handle = _RETRAIN_STATE.get("log_handle")
        if log_handle is not None:
            log_handle.close()
            _RETRAIN_STATE["log_handle"] = None
        _RETRAIN_STATE["process"] = None
        _RETRAIN_STATE["model_key"] = None
        _RETRAIN_STATE["staged_model_dir"] = None
        _RETRAIN_STATE["final_model_dir"] = None


def _read_log_tail(max_lines: int = 60) -> list[str]:
    if not RETRAIN_LOG_FILE.exists():
        return []

    lines = RETRAIN_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def _build_retrain_status() -> dict:
    _update_retrain_state()
    process = _RETRAIN_STATE.get("process")
    running = process is not None
    pid = process.pid if running else None
    return {
        "running": running,
        "pid": pid,
        "model_key": _RETRAIN_STATE.get("model_key"),
        "final_model_dir": _RETRAIN_STATE.get("final_model_dir"),
        "staged_model_dir": _RETRAIN_STATE.get("staged_model_dir"),
        "started_at": _RETRAIN_STATE.get("started_at"),
        "last_exit_code": _RETRAIN_STATE.get("last_exit_code"),
        "last_command": _RETRAIN_STATE.get("last_command"),
        "log_file": str(RETRAIN_LOG_FILE),
        "log_tail": _read_log_tail(),
    }


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        tanks = list_tank_ids()
        availability = model_availability()
        visible_model_keys = ["autogluon", "patchtst"]
        # Jinja's `tojson` requires plain JSON-serializable types.
        model_registry_view = {
            key: {
                "label": cfg.get("label", key),
                "path": str(cfg.get("path", "")),
                "available": availability.get(key, {}).get("available", False),
            }
            for key, cfg in MODEL_REGISTRY.items()
            if key in visible_model_keys
        }
        return render_template(
            "index.html",
            tanks=tanks,
            model_registry=model_registry_view,
        )

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "models": model_availability(),
                "tank_count": len(list_tank_ids()),
            }
        )

    @app.get("/api/tanks")
    def tanks():
        return jsonify({"tanks": list_tank_ids()})

    @app.post("/api/refresh")
    def refresh():
        cache_info = refresh_runtime_cache()
        return jsonify(
            {
                "status": "ok",
                "cache": cache_info,
                "models": model_availability(),
                "tanks": list_tank_ids(),
            }
        )

    @app.post("/api/sync-data")
    def sync_data():
        payload = request.get_json(silent=True) or {}
        token = str(payload.get("token", "")).strip()
        location_raw = payload.get("location_id")
        location_id = str(location_raw).strip() if location_raw is not None else ""

        if not token:
            return jsonify({"error": "token is required"}), 400

        start_date_raw = str(payload.get("start_date", "2025-01-01")).strip()
        end_date_raw = str(payload.get("end_date", date.today().isoformat())).strip()

        try:
            start_date = date.fromisoformat(start_date_raw)
            end_date = date.fromisoformat(end_date_raw)
        except ValueError:
            return jsonify({"error": "start_date/end_date must be in YYYY-MM-DD format"}), 400

        try:
            sync_summary = sync_waltr_dataset_incremental(
                token=token,
                dataset_dir=PROJECT_ROOT / "dataset",
                location_id=location_id or None,
                start_date=start_date,
                end_date=end_date,
            )
            cache_info = refresh_runtime_cache()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except PermissionError as exc:
            status_code = 401 if "unauthorized" in str(exc).lower() else 403
            return jsonify({"error": str(exc)}), status_code
        except Exception as exc:
            return jsonify({"error": f"Data sync failed: {exc}"}), 500

        return jsonify(
            {
                "status": "ok",
                "sync": sync_summary,
                "cache": cache_info,
                "tanks": list_tank_ids(),
            }
        )

    @app.get("/api/retrain-status")
    def retrain_status():
        return jsonify({"status": "ok", "retrain": _build_retrain_status()})

    @app.get("/api/model-run-comparison")
    @app.get("/api/compare-third-model")
    def compare_model_runs():
        return jsonify({"status": "ok", "comparison": compare_anomaly_model_runs()})

    @app.post("/api/retrain-model")
    @app.post("/api/retrain-third-model")
    def retrain_model():
        status = _build_retrain_status()
        if status["running"]:
            return jsonify({"error": "Model training is already running.", "retrain": status}), 409

        payload = request.get_json(silent=True) or {}

        model_key = str(payload.get("model_key", "autogluon")).strip()
        if model_key not in MODEL_REGISTRY:
            return jsonify({"error": f"Unsupported model_key: {model_key}"}), 400

        prediction_length = int(payload.get("prediction_length", 24))
        time_limit = int(payload.get("time_limit", 1800))
        min_history_hours = int(payload.get("min_history_hours", 24 * 14))
        presets = str(payload.get("presets", "high_quality"))
        target_col = str(payload.get("target_col", "Outflow in KL"))

        rolling_window = int(payload.get("rolling_window", 24))
        z_threshold = float(payload.get("z_threshold", 4.5))
        practical_max_kl = payload.get("practical_max_kl")
        practical_max_kl = None if practical_max_kl in (None, "", "null") else float(practical_max_kl)
        practical_max_multiplier = float(payload.get("practical_max_multiplier", 1.3))
        jump_z_threshold = float(payload.get("jump_z_threshold", 3.0))
        anomaly_context_hours = int(payload.get("anomaly_context_hours", 24 * 180))
        max_epochs = int(payload.get("max_epochs", 200))
        training_profile = str(payload.get("training_profile", "heavy"))
        num_val_windows = int(payload.get("num_val_windows", 3))
        refit_full = _as_bool(payload.get("refit_full", True), default=True)
        enable_deep_models = _as_bool(payload.get("enable_deep_models", True), default=True)
        default_model_dir = ANOMALY_HEAVY_DIR if training_profile == "heavy" else ANOMALY_BASELINE_DIR
        final_model_dir = Path(
            str(
                payload.get(
                    "model_dir",
                    MODEL_REGISTRY[model_key]["path"] if model_key != "anomaly_ensemble" else default_model_dir,
                )
            )
        ).resolve()
        staged_model_dir = (
            final_model_dir.parent
            / "_retrain_staging"
            / f"{model_key}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )

        if (
            prediction_length <= 0
            or time_limit <= 0
            or min_history_hours <= 0
        ):
            return jsonify({"error": "Invalid parameters for retraining request."}), 400

        cmd: list[str] = []
        script_path: Path | None = None

        if model_key == "autogluon":
            script_path = PROJECT_ROOT / "src/models/autogluon_hourly_forecasting.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--dataset-dir",
                str(PROJECT_ROOT / "dataset"),
                "--model-dir",
                str(staged_model_dir),
                "--prediction-length",
                str(prediction_length),
                "--time-limit",
                str(time_limit),
                "--target-col",
                target_col,
                "--min-history-hours",
                str(min_history_hours),
                "--presets",
                presets,
            ]
        elif model_key == "patchtst":
            script_path = PROJECT_ROOT / "src/models/patchtst_hourly_forecasting.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--dataset-dir",
                str(PROJECT_ROOT / "dataset"),
                "--model-dir",
                str(staged_model_dir),
                "--prediction-length",
                str(prediction_length),
                "--time-limit",
                str(time_limit),
                "--target-col",
                target_col,
                "--min-history-hours",
                str(min_history_hours),
                "--max-epochs",
                str(max_epochs),
                "--presets",
                presets,
            ]
        else:
            if (
                rolling_window < 2
                or z_threshold <= 0
                or (practical_max_kl is not None and practical_max_kl <= 0)
                or practical_max_multiplier <= 0
                or jump_z_threshold <= 0
                or num_val_windows <= 0
                or anomaly_context_hours <= 0
                or training_profile not in {"baseline", "heavy"}
            ):
                return jsonify({"error": "Invalid anomaly-model parameters for retraining request."}), 400

            script_path = PROJECT_ROOT / "src/models/anomaly_aware_ensemble_forecasting.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--dataset-dir",
                str(PROJECT_ROOT / "dataset"),
                "--model-dir",
                str(staged_model_dir),
                "--prediction-length",
                str(prediction_length),
                "--time-limit",
                str(time_limit),
                "--target-col",
                target_col,
                "--rolling-window",
                str(rolling_window),
                "--z-threshold",
                str(z_threshold),
                "--practical-max-multiplier",
                str(practical_max_multiplier),
                "--jump-z-threshold",
                str(jump_z_threshold),
                "--min-history-hours",
                str(min_history_hours),
                "--anomaly-context-hours",
                str(anomaly_context_hours),
                "--max-epochs",
                str(max_epochs),
                "--training-profile",
                training_profile,
                "--num-val-windows",
                str(num_val_windows),
                "--presets",
                presets,
            ]

            if refit_full:
                cmd.append("--refit-full")
            if enable_deep_models:
                cmd.append("--enable-deep-models")
            if practical_max_kl is not None:
                cmd.extend(["--practical-max-kl", str(practical_max_kl)])

        if script_path is None or not script_path.exists():
            return jsonify({"error": f"Training script not found: {script_path}"}), 500

        RETRAIN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with RETRAIN_LOG_FILE.open("a", encoding="utf-8") as log_f:
            log_f.write(f"\n[{_utc_now_iso()}] Starting model retrain: {model_key}\n")
            log_f.write(f"Active model dir: {final_model_dir}\n")
            log_f.write(f"Staging model dir: {staged_model_dir}\n")
            log_f.write("Command: " + " ".join(cmd) + "\n")

        log_handle = RETRAIN_LOG_FILE.open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        _RETRAIN_STATE["process"] = process
        _RETRAIN_STATE["log_handle"] = log_handle
        _RETRAIN_STATE["started_at"] = _utc_now_iso()
        _RETRAIN_STATE["last_exit_code"] = None
        _RETRAIN_STATE["last_command"] = " ".join(cmd)
        _RETRAIN_STATE["model_key"] = model_key
        _RETRAIN_STATE["final_model_dir"] = str(final_model_dir)
        _RETRAIN_STATE["staged_model_dir"] = str(staged_model_dir)

        return jsonify({"status": "started", "retrain": _build_retrain_status()})

    @app.post("/api/retrain-stop")
    def retrain_stop():
        _update_retrain_state()
        process = _RETRAIN_STATE.get("process")
        if process is None:
            return jsonify({"status": "idle", "retrain": _build_retrain_status()})

        assert isinstance(process, subprocess.Popen)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

        _update_retrain_state()

        with RETRAIN_LOG_FILE.open("a", encoding="utf-8") as log_f:
            log_f.write(f"[{_utc_now_iso()}] Retrain stopped by user.\n")

        return jsonify({"status": "stopped", "retrain": _build_retrain_status()})

    @app.post("/api/anomaly-preview")
    def anomaly_preview():
        payload = request.get_json(silent=True) or {}
        tank_id = str(payload.get("tank_id", "")).strip()
        hours = int(payload.get("hours", 24 * 7))
        rolling_window = int(payload.get("rolling_window", 24))
        z_threshold = float(payload.get("z_threshold", 4.5))
        anomaly_context_hours = int(payload.get("anomaly_context_hours", 24 * 180))
        practical_max_kl = payload.get("practical_max_kl")
        practical_max_kl = None if practical_max_kl in (None, "", "null") else float(practical_max_kl)
        practical_max_multiplier = float(payload.get("practical_max_multiplier", 1.3))
        jump_z_threshold = float(payload.get("jump_z_threshold", 3.0))

        if not tank_id:
            return jsonify({"error": "tank_id is required"}), 400

        try:
            preview = anomaly_clean_preview(
                tank_id=tank_id,
                hours=hours,
                rolling_window=rolling_window,
                z_threshold=z_threshold,
                anomaly_context_hours=anomaly_context_hours,
                practical_max_kl=practical_max_kl,
                practical_max_multiplier=practical_max_multiplier,
                jump_z_threshold=jump_z_threshold,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(preview)

    @app.get("/api/history")
    def history():
        tank_id = request.args.get("tank_id", type=str)
        hours = request.args.get("hours", default=24 * 7, type=int)

        if not tank_id:
            return jsonify({"error": "tank_id is required"}), 400

        try:
            history_df = get_recent_history(tank_id=tank_id, hours=hours)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "tank_id": tank_id,
                "hours": int(hours),
                "history": history_df.to_dict(orient="records"),
            }
        )

    @app.post("/api/forecast")
    def forecast():
        payload = request.get_json(silent=True) or {}
        tank_id = str(payload.get("tank_id", "")).strip()
        prediction_length = int(payload.get("prediction_length", 24))
        model_keys = payload.get("model_keys")

        if not tank_id:
            return jsonify({"error": "tank_id is required"}), 400

        if model_keys is not None and not isinstance(model_keys, list):
            return jsonify({"error": "model_keys must be an array of model keys"}), 400

        try:
            result = forecast_next_24h(
                tank_id=tank_id,
                model_keys=model_keys,
                prediction_length=prediction_length,
            )
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Forecast failed: {exc}"}), 500

        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
