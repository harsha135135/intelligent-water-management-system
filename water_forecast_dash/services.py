from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Resolve project root and import existing feature/loader utilities from src/models.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_SRC_DIR = PROJECT_ROOT / "src" / "models"
AUTOGLUON_UTILS_PATH = MODELS_SRC_DIR / "autogluon_hourly_forecasting.py"


def _load_autogluon_utils_module():
    spec = importlib.util.spec_from_file_location(
        "autogluon_hourly_forecasting",
        AUTOGLUON_UTILS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path: {AUTOGLUON_UTILS_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ag_utils = _load_autogluon_utils_module()

ITEM_ID_COL = _ag_utils.ITEM_ID_COL
KNOWN_COVARIATES = _ag_utils.KNOWN_COVARIATES
TARGET_COL = _ag_utils.TARGET_COL
TIMESTAMP_COL = _ag_utils.TIMESTAMP_COL
add_leakage_safe_features = _ag_utils.add_leakage_safe_features
load_hourly_json_dataset = _ag_utils.load_hourly_json_dataset
make_tsdf = _ag_utils.make_tsdf

DATASET_DIR = PROJECT_ROOT / "dataset"
ANOMALY_BASELINE_DIR = PROJECT_ROOT / "results" / "autogluon" / "anomaly_ensemble_full"
ANOMALY_HEAVY_DIR = PROJECT_ROOT / "results" / "autogluon" / "anomaly_ensemble_heavy"
MODEL_REGISTRY = {
    "autogluon": {
        "label": "AutoGluon",
        "path": PROJECT_ROOT / "results" / "autogluon" / "hourly_forecasting",
        "future_file": "future_predictions_readable.csv",
        "holdout_file": "holdout_predictions_readable.csv",
    },
    "patchtst": {
        "label": "PatchTST",
        "path": PROJECT_ROOT / "results" / "patchtst" / "hourly_forecasting",
        "future_file": "future_predictions_readable.csv",
        "holdout_file": "holdout_predictions_readable.csv",
    },
    "anomaly_ensemble": {
        "label": "Anomaly-Aware Ensemble",
        "path": ANOMALY_BASELINE_DIR,
        "future_file": "future_predictions_readable.csv",
        "holdout_file": "holdout_predictions_readable.csv",
    },
}


_RUNTIME_CACHE: dict[str, Any] = {
    "dataset_snapshot": None,
    "dataset_df": None,
    "featured_df": None,
}
_PREDICTOR_CACHE: dict[str, tuple[int, TimeSeriesPredictor]] = {}


def _dataset_snapshot() -> tuple[int, int]:
    json_files = list(DATASET_DIR.glob("*/*.json"))
    if not json_files:
        return (0, 0)

    latest_mtime_ns = max(p.stat().st_mtime_ns for p in json_files)
    return (len(json_files), latest_mtime_ns)


def _ensure_dataset_cache() -> None:
    snapshot = _dataset_snapshot()
    cached_snapshot = _RUNTIME_CACHE["dataset_snapshot"]
    if snapshot == cached_snapshot and _RUNTIME_CACHE["dataset_df"] is not None:
        return

    dataset_df = load_hourly_json_dataset(DATASET_DIR)
    featured_df = add_leakage_safe_features(dataset_df)

    _RUNTIME_CACHE["dataset_snapshot"] = snapshot
    _RUNTIME_CACHE["dataset_df"] = dataset_df
    _RUNTIME_CACHE["featured_df"] = featured_df


def refresh_runtime_cache() -> dict[str, Any]:
    _RUNTIME_CACHE["dataset_snapshot"] = None
    _RUNTIME_CACHE["dataset_df"] = None
    _RUNTIME_CACHE["featured_df"] = None
    _PREDICTOR_CACHE.clear()
    _ensure_dataset_cache()
    return {
        "dataset_snapshot": _RUNTIME_CACHE["dataset_snapshot"],
        "tank_count": len(list_tank_ids()),
    }


def get_dataset() -> pd.DataFrame:
    """Load cached dataset and refresh automatically when new JSON files are added."""
    _ensure_dataset_cache()
    return _RUNTIME_CACHE["dataset_df"]


def get_featured_dataset() -> pd.DataFrame:
    """Add deterministic calendar covariates used by trained models."""
    _ensure_dataset_cache()
    return _RUNTIME_CACHE["featured_df"]


def _predictor_mtime_ns(model_path: Path) -> int:
    predictor_file = model_path / "predictor.pkl"
    if predictor_file.exists():
        return predictor_file.stat().st_mtime_ns
    return model_path.stat().st_mtime_ns


def get_predictor(model_key: str) -> TimeSeriesPredictor:
    model_path = get_model_path(model_key)
    mtime_ns = _predictor_mtime_ns(model_path)

    cached = _PREDICTOR_CACHE.get(model_key)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]

    predictor = TimeSeriesPredictor.load(str(model_path), require_version_match=False)
    _PREDICTOR_CACHE[model_key] = (mtime_ns, predictor)
    return predictor


def get_model_path(model_key: str) -> Path:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_key}")

    if model_key == "anomaly_ensemble":
        heavy_predictor = ANOMALY_HEAVY_DIR / "predictor.pkl"
        if heavy_predictor.exists():
            return ANOMALY_HEAVY_DIR

    model_path = MODEL_REGISTRY[model_key]["path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    return model_path


def list_tank_ids() -> list[str]:
    df = get_dataset()
    return sorted(df[ITEM_ID_COL].astype(str).unique().tolist())


def get_recent_history(tank_id: str, hours: int = 168) -> pd.DataFrame:
    df = get_dataset()
    tank_df = df[df[ITEM_ID_COL].astype(str) == str(tank_id)].copy()
    if tank_df.empty:
        raise ValueError(f"Unknown tank_id: {tank_id}")

    tank_df = tank_df.sort_values(TIMESTAMP_COL)
    if hours > 0:
        tank_df = tank_df.tail(hours)

    tank_df[TIMESTAMP_COL] = tank_df[TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M:%S")

    keep_cols = [ITEM_ID_COL, TIMESTAMP_COL, TARGET_COL]
    return tank_df[keep_cols].reset_index(drop=True)


def anomaly_clean_preview(
    tank_id: str,
    hours: int,
    rolling_window: int,
    z_threshold: float,
    anomaly_context_hours: int,
    practical_max_kl: float | None,
    practical_max_multiplier: float,
    jump_z_threshold: float,
) -> dict[str, Any]:
    """Return original vs anomaly-cleaned series for interactive UI comparison."""
    if hours <= 0:
        raise ValueError("hours must be > 0")
    if rolling_window < 2:
        raise ValueError("rolling_window must be >= 2")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be > 0")
    if anomaly_context_hours <= 0:
        raise ValueError("anomaly_context_hours must be > 0")
    if practical_max_kl is not None and practical_max_kl <= 0:
        raise ValueError("practical_max_kl must be > 0 when provided")
    if practical_max_multiplier <= 0:
        raise ValueError("practical_max_multiplier must be > 0")
    if jump_z_threshold <= 0:
        raise ValueError("jump_z_threshold must be > 0")

    df = get_dataset()
    tank_df = df[df[ITEM_ID_COL].astype(str) == str(tank_id)].copy()
    if tank_df.empty:
        raise ValueError(f"Unknown tank_id: {tank_id}")

    tank_df = tank_df.sort_values(TIMESTAMP_COL).tail(hours).reset_index(drop=True)

    y = tank_df[TARGET_COL].astype(float)
    context_y = y.tail(max(1, anomaly_context_hours))
    rolling_mean = y.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).mean()
    rolling_std = y.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).std()

    fallback_mean = y.rolling(window=rolling_window, min_periods=1).mean()
    rolling_mean = rolling_mean.fillna(fallback_mean)

    median_std = float(rolling_std.median()) if rolling_std.notna().any() else 0.0
    if not np.isfinite(median_std) or median_std <= 0:
        std_val = float(context_y.std()) if np.isfinite(context_y.std()) else 0.0
        median_std = std_val if std_val > 0 else 1.0
    rolling_std = rolling_std.fillna(median_std)

    low = rolling_mean - z_threshold * rolling_std
    high = rolling_mean + z_threshold * rolling_std
    q_low = context_y.quantile(0.01)
    q_high = context_y.quantile(0.99)

    practical_cap = float(practical_max_kl) if practical_max_kl is not None else float(max(q_high * practical_max_multiplier, 1.0))
    practical_mask = y > practical_cap

    diff = y.diff().abs().fillna(0.0)
    diff_roll_mean = diff.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).mean()
    diff_roll_std = diff.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).std()

    diff_fallback_mean = diff.rolling(window=rolling_window, min_periods=1).mean()
    diff_roll_mean = diff_roll_mean.fillna(diff_fallback_mean)

    diff_median_std = float(diff_roll_std.median()) if diff_roll_std.notna().any() else 0.0
    if not np.isfinite(diff_median_std) or diff_median_std <= 0:
        diff_std_val = float(diff.std()) if np.isfinite(diff.std()) and diff.std() > 0 else 1.0
        diff_median_std = diff_std_val
    diff_roll_std = diff_roll_std.fillna(diff_median_std)

    jump_threshold = diff_roll_mean + jump_z_threshold * diff_roll_std
    jump_mask = (diff > jump_threshold).fillna(False)

    # Practical-only policy: preserve statistical demand spikes and replace only
    # implausible values.
    non_physical_negative = (y < 0).fillna(False)
    hard_practical_anomaly = (practical_mask | non_physical_negative).fillna(False)
    jump_practical_anomaly = (jump_mask & practical_mask).fillna(False)

    anomaly_mask = (hard_practical_anomaly | jump_practical_anomaly).fillna(False)
    cleaned = y.where(~anomaly_mask, rolling_mean).clip(lower=0)

    result_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(tank_df[TIMESTAMP_COL]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual": y,
            "cleaned": cleaned,
            "is_anomaly": anomaly_mask.astype(int),
            "low": low,
            "high": high,
            "practical_cap": practical_cap,
            "jump_threshold": jump_threshold,
        }
    )

    return {
        "tank_id": str(tank_id),
        "hours": int(hours),
        "rolling_window": int(rolling_window),
        "z_threshold": float(z_threshold),
        "anomaly_context_hours": int(anomaly_context_hours),
        "practical_max_kl": None if practical_max_kl is None else float(practical_max_kl),
        "practical_max_multiplier": float(practical_max_multiplier),
        "jump_z_threshold": float(jump_z_threshold),
        "anomaly_count": int(result_df["is_anomaly"].sum()),
        "series": result_df.to_dict(orient="records"),
    }


def _create_future_covariates_for_item(
    history_tsdf: TimeSeriesDataFrame,
    prediction_length: int,
) -> TimeSeriesDataFrame:
    """Generate next-horizon calendar covariates for each item in history_tsdf."""
    future_rows = []

    for item_id in history_tsdf.item_ids:
        ts_index = history_tsdf.loc[item_id].index
        start_ts = ts_index.max()
        future_dates = pd.date_range(
            start=start_ts + pd.Timedelta(hours=1),
            periods=prediction_length,
            freq="h",
        )
        future_rows.append(
            pd.DataFrame(
                {
                    ITEM_ID_COL: item_id,
                    TIMESTAMP_COL: future_dates,
                }
            )
        )

    future_df = pd.concat(future_rows, ignore_index=True)
    future_df = add_leakage_safe_features(future_df)

    columns = [ITEM_ID_COL, TIMESTAMP_COL] + KNOWN_COVARIATES
    return TimeSeriesDataFrame.from_data_frame(
        future_df[columns],
        id_column=ITEM_ID_COL,
        timestamp_column=TIMESTAMP_COL,
    )


def _normalize_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.rename(columns={"mean": "pred_mean"}).copy()

    if "0.1" in pred_df.columns:
        pred_df = pred_df.rename(columns={"0.1": "pred_p10"})
    if "0.5" in pred_df.columns:
        pred_df = pred_df.rename(columns={"0.5": "pred_p50"})
    if "0.9" in pred_df.columns:
        pred_df = pred_df.rename(columns={"0.9": "pred_p90"})

    return pred_df


def _load_saved_predictions(
    file_path: Path,
    tank_id: str,
    prediction_length: int,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Saved prediction file not found: {file_path}")

    df = pd.read_csv(file_path)
    required = {ITEM_ID_COL, "pred_mean"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Saved predictions missing columns {missing}: {file_path}")

    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")

    tank_df = df[df[ITEM_ID_COL].astype(str) == str(tank_id)].copy()
    if tank_df.empty:
        raise ValueError(f"No saved predictions found for tank_id={tank_id} in {file_path}")

    if TIMESTAMP_COL in tank_df.columns:
        tank_df = tank_df.sort_values(TIMESTAMP_COL)

    return tank_df.head(prediction_length).reset_index(drop=True)


def _build_fallback_forecast_from_saved_files(
    model_key: str,
    tank_id: str,
    prediction_length: int,
    history_end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    cfg = MODEL_REGISTRY[model_key]
    model_path = Path(cfg["path"])

    future_file = model_path / cfg["future_file"]
    if future_file.exists():
        saved_df = _load_saved_predictions(future_file, tank_id, prediction_length)
        source = "saved_future_predictions"
    else:
        holdout_file = model_path / cfg["holdout_file"]
        holdout_df = _load_saved_predictions(holdout_file, tank_id, prediction_length)

        # Holdout timestamps are historical; remap them onto the next horizon.
        future_dates = pd.date_range(
            start=history_end_ts + pd.Timedelta(hours=1),
            periods=min(prediction_length, len(holdout_df)),
            freq="h",
        )
        holdout_df = holdout_df.head(len(future_dates)).copy()
        holdout_df[TIMESTAMP_COL] = future_dates
        saved_df = holdout_df
        source = "saved_holdout_proxy"

    # Always align fallback timestamps to the current history horizon so stale
    # saved files don't surface old calendar dates after new sync data arrives.
    fallback_horizon = min(prediction_length, len(saved_df))
    aligned_future_dates = pd.date_range(
        start=history_end_ts + pd.Timedelta(hours=1),
        periods=fallback_horizon,
        freq="h",
    )
    saved_df = saved_df.head(fallback_horizon).copy()
    saved_df[TIMESTAMP_COL] = aligned_future_dates

    saved_df = _normalize_predictions(saved_df)
    keep_cols = [ITEM_ID_COL, TIMESTAMP_COL, "pred_mean"]
    for col in ["pred_p10", "pred_p50", "pred_p90"]:
        if col in saved_df.columns:
            keep_cols.append(col)

    saved_df = saved_df[keep_cols].copy()
    saved_df[TIMESTAMP_COL] = pd.to_datetime(saved_df[TIMESTAMP_COL]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return saved_df, source


def forecast_next_24h(
    tank_id: str,
    model_keys: list[str] | None = None,
    prediction_length: int = 24,
) -> dict:
    """Run next-horizon forecasts from one or more trained models for one tank."""
    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0")

    if model_keys is None or not model_keys:
        model_keys = list(MODEL_REGISTRY.keys())

    invalid_models = [key for key in model_keys if key not in MODEL_REGISTRY]
    if invalid_models:
        raise ValueError(f"Unsupported model keys: {invalid_models}")

    feat_df = get_featured_dataset()
    tank_feat_df = feat_df[feat_df[ITEM_ID_COL].astype(str) == str(tank_id)].copy()
    if tank_feat_df.empty:
        raise ValueError(f"Unknown tank_id: {tank_id}")

    tank_feat_df = tank_feat_df.sort_values(TIMESTAMP_COL)
    ts_data = make_tsdf(tank_feat_df, TARGET_COL)
    known_covariates = _create_future_covariates_for_item(ts_data, prediction_length)

    forecasts: dict[str, list[dict]] = {}
    forecast_sources: dict[str, str] = {}
    warnings: list[str] = []
    history_end_ts = pd.to_datetime(tank_feat_df[TIMESTAMP_COL]).max()

    for model_key in model_keys:
        try:
            predictor = get_predictor(model_key)
            pred_tsdf = predictor.predict(ts_data, known_covariates=known_covariates)
            pred_df = _normalize_predictions(pred_tsdf.reset_index())
            pred_df[TIMESTAMP_COL] = pd.to_datetime(pred_df[TIMESTAMP_COL]).dt.strftime("%Y-%m-%d %H:%M:%S")
            source = "live_model_inference"
        except Exception as exc:
            pred_df, source = _build_fallback_forecast_from_saved_files(
                model_key=model_key,
                tank_id=tank_id,
                prediction_length=prediction_length,
                history_end_ts=history_end_ts,
            )
            warnings.append(
                f"{model_key}: using fallback ({source}) due to inference error: {exc}"
            )

        keep_cols = [ITEM_ID_COL, TIMESTAMP_COL, "pred_mean"]
        for col in ["pred_p10", "pred_p50", "pred_p90"]:
            if col in pred_df.columns:
                keep_cols.append(col)

        forecasts[model_key] = (
            pred_df[keep_cols]
            .sort_values(TIMESTAMP_COL)
            .to_dict(orient="records")
        )
        forecast_sources[model_key] = source

    history_df = get_recent_history(tank_id=tank_id, hours=24 * 7)

    return {
        "tank_id": str(tank_id),
        "prediction_length": int(prediction_length),
        "history": history_df.to_dict(orient="records"),
        "forecasts": forecasts,
        "forecast_sources": forecast_sources,
        "warnings": warnings,
    }


def model_availability() -> dict[str, dict]:
    status = {}
    for model_key, cfg in MODEL_REGISTRY.items():
        model_path = get_model_path(model_key) if model_key == "anomaly_ensemble" else Path(cfg["path"])
        status[model_key] = {
            "label": cfg["label"],
            "path": str(model_path),
            "available": model_path.exists() and (model_path / "predictor.pkl").exists(),
        }
    return status


def _read_training_summary(model_dir: Path) -> dict[str, Any] | None:
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        return None

    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_mase_from_sources(model_dir: Path, summary: dict[str, Any] | None) -> float | None:
    if not summary:
        summary = {}

    mase = summary.get("best_mase")
    if mase is not None:
        try:
            return float(mase)
        except (TypeError, ValueError):
            pass

    flipped = summary.get("best_score_val_flipped")
    if flipped is not None:
        try:
            return float(-float(flipped))
        except (TypeError, ValueError):
            return None

    leaderboard_path = model_dir / "leaderboard.csv"
    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            if not leaderboard_df.empty and "score_val" in leaderboard_df.columns:
                best_flipped = float(leaderboard_df.iloc[0]["score_val"])
                return -best_flipped
        except Exception:
            return None
    return None


def compare_anomaly_model_runs() -> dict[str, Any]:
    baseline_summary = _read_training_summary(ANOMALY_BASELINE_DIR)
    heavy_summary = _read_training_summary(ANOMALY_HEAVY_DIR)

    baseline_mase = _extract_mase_from_sources(ANOMALY_BASELINE_DIR, baseline_summary)
    heavy_mase = _extract_mase_from_sources(ANOMALY_HEAVY_DIR, heavy_summary)

    better_profile = None
    mase_delta = None
    improvement_pct = None
    if baseline_mase is not None and heavy_mase is not None:
        mase_delta = heavy_mase - baseline_mase
        if baseline_mase != 0:
            improvement_pct = ((baseline_mase - heavy_mase) / baseline_mase) * 100.0
        better_profile = "heavy" if heavy_mase < baseline_mase else "baseline"

    return {
        "baseline": {
            "path": str(ANOMALY_BASELINE_DIR),
            "available": (ANOMALY_BASELINE_DIR / "predictor.pkl").exists(),
            "summary": baseline_summary,
            "best_mase": baseline_mase,
        },
        "heavy": {
            "path": str(ANOMALY_HEAVY_DIR),
            "available": (ANOMALY_HEAVY_DIR / "predictor.pkl").exists(),
            "summary": heavy_summary,
            "best_mase": heavy_mase,
        },
        "comparison": {
            "metric": "MASE (lower is better)",
            "better_profile": better_profile,
            "mase_delta_heavy_minus_baseline": mase_delta,
            "improvement_pct_heavy_vs_baseline": improvement_pct,
            "active_for_dashboard": "heavy" if (ANOMALY_HEAVY_DIR / "predictor.pkl").exists() else "baseline",
        },
    }
