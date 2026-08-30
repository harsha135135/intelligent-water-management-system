import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


logger = logging.getLogger(__name__)

TARGET_COL = "Outflow in KL"
TIMESTAMP_COL = "timestamp"
ITEM_ID_COL = "item_id"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWN_COVARIATES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# Centralised constant so version-specific name changes are one-line fixes.
CHRONOS_MODEL_KEY = "Chronos"


def _safe_float(value, field_name: str = "", source_hint: str = "") -> float:
    """Convert a value to float, returning NaN on failure and logging a warning."""
    try:
        return float(value)
    except (TypeError, ValueError):
        if field_name or source_hint:
            logger.warning(
                "Could not convert value %r to float (field=%r, source=%r). Substituting NaN.",
                value,
                field_name,
                source_hint,
            )
        return np.nan


def load_hourly_json_dataset(
    dataset_dir: Path,
    max_tanks: int | None = None,
    max_files_per_tank: int | None = None,
) -> pd.DataFrame:
    """Load all tank/day JSON files into a single long DataFrame.

    Timestamps are parsed in bulk after row collection for better performance.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    rows: List[Dict] = []
    nan_counts: Dict[str, int] = {}

    tank_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if max_tanks is not None:
        tank_dirs = tank_dirs[:max_tanks]

    numeric_fields = [
        "Inflow in KL",
        "Outflow in KL",
        "Opening Value in KL",
        "Closing Value in KL",
    ]

    for tank_dir in tank_dirs:
        item_id = tank_dir.name
        json_files = sorted(tank_dir.glob("*.json"))
        if max_files_per_tank is not None:
            json_files = json_files[:max_files_per_tank]

        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.warning("Skipping %s: %s", json_file, exc)
                continue

            tank_name = payload.get("Tank Name", item_id)
            tank_type = payload.get("Tank Type", "unknown")
            tank_shape = payload.get("Tank Shape", "unknown")
            source_hint = str(json_file)

            for rec in payload.get("data", []):
                row: Dict = {
                    ITEM_ID_COL: item_id,
                    # Store raw string; parse in bulk below.
                    TIMESTAMP_COL: rec.get("Date Time"),
                    "tank_name": tank_name,
                    "tank_type": tank_type,
                    "tank_shape": tank_shape,
                }
                for field in numeric_fields:
                    val = _safe_float(rec.get(field), field_name=field, source_hint=source_hint)
                    row[field] = val
                    if np.isnan(val):
                        nan_counts[field] = nan_counts.get(field, 0) + 1
                rows.append(row)

    if not rows:
        raise ValueError(f"No hourly rows found under: {dataset_dir}")

    df = pd.DataFrame(rows)

    # Bulk timestamp parse — much faster than calling pd.to_datetime row-by-row.
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], format="%Y-%m-%d %H", errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL])

    df = df.drop_duplicates(subset=[ITEM_ID_COL, TIMESTAMP_COL], keep="last")
    df = df.sort_values([ITEM_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)

    logger.info(
        "Ingestion NaN summary: %s",
        {k: v for k, v in nan_counts.items() if v > 0} or "none",
    )

    return df


def add_leakage_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add only deterministic calendar features that are known for future timestamps.

    We intentionally avoid using contemporaneous process variables
    (opening/closing/inflow) as known covariates because they are unknown at
    forecast time and can cause leakage.
    """
    df = df.copy()
    ts = df[TIMESTAMP_COL]

    df["hour"] = ts.dt.hour.astype(int)
    df["day_of_week"] = ts.dt.dayofweek.astype(int)
    df["day_of_month"] = ts.dt.day.astype(int)
    df["month"] = ts.dt.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    return df


def filter_viable_series(
    df: pd.DataFrame,
    prediction_length: int,
    min_history: int = 24 * 14,
) -> pd.DataFrame:
    """Keep only tanks with enough history for robust training and evaluation."""
    min_points = prediction_length + min_history
    counts = df.groupby(ITEM_ID_COL).size()
    keep_ids = counts[counts >= min_points].index
    filtered = df[df[ITEM_ID_COL].isin(keep_ids)].copy()

    dropped_ids = counts[counts < min_points].index.tolist()
    logger.info(
        "Series filter summary: kept=%d, dropped=%d, min_points_required=%d",
        len(keep_ids),
        len(dropped_ids),
        min_points,
    )

    if filtered.empty:
        raise ValueError(
            "No time series has enough history after filtering. "
            f"Need at least {min_points} points per tank."
        )

    return filtered


def split_train_test(
    df: pd.DataFrame,
    prediction_length: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split per tank: train = all but last horizon; test = full series."""
    train_parts = []
    test_parts = []

    # sort=True (default) makes the inner sort_values call redundant.
    for _, g in df.groupby(ITEM_ID_COL, sort=True):
        g = g.sort_values(TIMESTAMP_COL)
        train_parts.append(g.iloc[:-prediction_length])
        test_parts.append(g)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def _validate_covariate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if any expected covariate column is missing."""
    missing = [col for col in KNOWN_COVARIATES if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing expected covariate columns: {missing}. "
            "Ensure add_leakage_safe_features() has been called before make_tsdf()."
        )


def make_tsdf(df: pd.DataFrame, target_col: str) -> TimeSeriesDataFrame:
    _validate_covariate_columns(df)
    cols = [ITEM_ID_COL, TIMESTAMP_COL, target_col] + KNOWN_COVARIATES
    return TimeSeriesDataFrame.from_data_frame(
        df[cols], id_column=ITEM_ID_COL, timestamp_column=TIMESTAMP_COL
    )


def build_future_known_covariates(
    test_df: pd.DataFrame,
    prediction_length: int,
) -> TimeSeriesDataFrame:
    """Construct known covariates only for each item's forecast horizon.

    Note: only the calendar features for the held-out window are passed here.
    The predictor itself is called with train_tsdf so it never sees future
    target values.
    """
    _validate_covariate_columns(test_df)
    future = (
        test_df.sort_values([ITEM_ID_COL, TIMESTAMP_COL])
        .groupby(ITEM_ID_COL, as_index=False, group_keys=False)
        .tail(prediction_length)
    )
    cols = [ITEM_ID_COL, TIMESTAMP_COL] + KNOWN_COVARIATES
    return TimeSeriesDataFrame.from_data_frame(
        future[cols], id_column=ITEM_ID_COL, timestamp_column=TIMESTAMP_COL
    )


def _rmse(series: pd.Series) -> float:
    """Root mean squared error helper for use inside groupby aggregations."""
    return float(np.sqrt(np.mean(np.square(series))))


def evaluate_forecast(
    predictions: TimeSeriesDataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    prediction_length: int,
) -> Dict[str, float]:
    pred_df = predictions.reset_index()[[ITEM_ID_COL, TIMESTAMP_COL, "mean"]]
    actual_df = (
        test_df.sort_values([ITEM_ID_COL, TIMESTAMP_COL])
        .groupby(ITEM_ID_COL, as_index=False, group_keys=False)
        .tail(prediction_length)[[ITEM_ID_COL, TIMESTAMP_COL, target_col]]
    )

    merged = actual_df.merge(pred_df, on=[ITEM_ID_COL, TIMESTAMP_COL], how="inner")
    merged = merged.dropna(subset=[target_col, "mean"])

    if merged.empty:
        raise ValueError("No aligned prediction/actual rows to evaluate.")

    # Guard against timestamp misalignment silently swallowing rows.
    expected_rows = actual_df[ITEM_ID_COL].nunique() * prediction_length
    if len(merged) < expected_rows * 0.9:
        logger.warning(
            "Evaluation merge retained only %d / %d expected rows. "
            "Check for timestamp misalignment between predictions and test_df.",
            len(merged),
            expected_rows,
        )

    err = merged[target_col] - merged["mean"]
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))

    return {
        "rows_evaluated": int(len(merged)),
        "rmse": rmse,
        "mae": mae,
    }


def build_readable_holdout_predictions(
    predictions: TimeSeriesDataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    prediction_length: int,
) -> pd.DataFrame:
    """Create a compact, human-readable holdout table with errors and intervals."""
    pred_df = predictions.reset_index().copy()
    actual_df = (
        test_df.sort_values([ITEM_ID_COL, TIMESTAMP_COL])
        .groupby(ITEM_ID_COL, as_index=False, group_keys=False)
        .tail(prediction_length)[[ITEM_ID_COL, TIMESTAMP_COL, target_col]]
        .rename(columns={target_col: "actual"})
    )

    merged = actual_df.merge(pred_df, on=[ITEM_ID_COL, TIMESTAMP_COL], how="inner")
    merged = merged.rename(columns={"mean": "pred_mean"})

    quantile_renames = {"0.1": "pred_p10", "0.5": "pred_p50", "0.9": "pred_p90"}
    merged = merged.rename(columns={k: v for k, v in quantile_renames.items() if k in merged.columns})

    merged["error"] = merged["actual"] - merged["pred_mean"]
    merged["abs_error"] = merged["error"].abs()

    if {"pred_p10", "pred_p90"}.issubset(merged.columns):
        merged["actual_in_p10_p90"] = (
            (merged["actual"] >= merged["pred_p10"]) & (merged["actual"] <= merged["pred_p90"])
        ).astype(int)

    keep_cols = [ITEM_ID_COL, TIMESTAMP_COL, "actual", "pred_mean", "error", "abs_error"]
    for col in ["pred_p10", "pred_p50", "pred_p90", "actual_in_p10_p90"]:
        if col in merged.columns:
            keep_cols.append(col)

    return merged[keep_cols].sort_values([ITEM_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)


def build_per_tank_metrics(readable_pred_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate holdout errors per tank for quick model diagnosis."""
    grouped = readable_pred_df.groupby(ITEM_ID_COL, as_index=False)
    summary = grouped.agg(
        rows_evaluated=("abs_error", "size"),
        mae=("abs_error", "mean"),
        # Named function instead of lambda: more readable and slightly faster.
        rmse=("error", _rmse),
        mean_actual=("actual", "mean"),
        mean_pred=("pred_mean", "mean"),
    )

    if "actual_in_p10_p90" in readable_pred_df.columns:
        coverage = (
            readable_pred_df.groupby(ITEM_ID_COL)["actual_in_p10_p90"]
            .mean()
            .reset_index()
            .rename(columns={"actual_in_p10_p90": "p10_p90_coverage"})
        )
        summary = summary.merge(coverage, on=ITEM_ID_COL, how="left")

    return summary.sort_values("mae", ascending=False).reset_index(drop=True)


def _build_base_hyperparameters() -> dict:
    return {
        "Naive": {},
        "SeasonalNaive": {},
        "ETS": {},
        "Theta": {},
        "DynamicOptimizedTheta": {},
        "NPTS": {},
    }


def _build_online_hyperparameters() -> dict:
    return {
        CHRONOS_MODEL_KEY: {},
        "TemporalFusionTransformer": {},
        "PatchTST": {},
        "DeepAR": {},
        "TiDE": {},
    }


def train_autogluon_hourly_forecaster(
    dataset_dir: Path,
    model_dir: Path,
    prediction_length: int,
    time_limit: int,
    target_col: str,
    max_tanks: int | None,
    max_files_per_tank: int | None,
    min_history_hours: int,
    allow_online_models: bool,
    presets: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("Loading JSON data from: %s", dataset_dir)
    raw_df = load_hourly_json_dataset(
        dataset_dir,
        max_tanks=max_tanks,
        max_files_per_tank=max_files_per_tank,
    )
    logger.info(
        "Ingestion summary: rows=%d, tanks=%d, time_range=[%s -> %s]",
        len(raw_df),
        raw_df[ITEM_ID_COL].nunique(),
        raw_df[TIMESTAMP_COL].min(),
        raw_df[TIMESTAMP_COL].max(),
    )

    logger.info("Adding leakage-safe time features...")
    feat_df = add_leakage_safe_features(raw_df)

    logger.info("Filtering short series...")
    feat_df = filter_viable_series(
        feat_df,
        prediction_length=prediction_length,
        min_history=min_history_hours,
    )

    logger.info("Creating train/test split...")
    train_df, test_df = split_train_test(feat_df, prediction_length=prediction_length)

    train_tsdf = make_tsdf(train_df, target_col=target_col)
    # future_known_cov contains only the calendar features for the held-out
    # window. The predictor is called with train_tsdf so it never sees future
    # target values.
    future_known_cov = build_future_known_covariates(test_df, prediction_length=prediction_length)

    model_dir.mkdir(parents=True, exist_ok=True)

    predictor = TimeSeriesPredictor(
        path=str(model_dir),
        target=target_col,
        prediction_length=prediction_length,
        freq="h",
        eval_metric="MASE",
        known_covariates_names=KNOWN_COVARIATES,
    )

    hyperparameters = _build_base_hyperparameters()
    if allow_online_models:
        # These models may download checkpoints from Hugging Face.
        hyperparameters.update(_build_online_hyperparameters())

    def _fit_with(hparams: dict) -> None:
        predictor.fit(
            train_data=train_tsdf,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=hparams,
            num_val_windows=1,
            verbosity=2,
        )

    logger.info("Training AutoGluon TimeSeries predictor...")
    try:
        _fit_with(hyperparameters)
    except ValueError as exc:
        # Fallback for version-specific model-name mismatches.
        if "is not supported" in str(exc):
            logger.warning(
                "Model name compatibility issue: %s. "
                "Falling back to offline-only model set.",
                exc,
            )
            _fit_with(_build_base_hyperparameters())
        else:
            raise
    except Exception as exc:
        # Network/cache issues for Chronos should not abort the full run.
        msg = str(exc).lower()
        chronos_related = (
            CHRONOS_MODEL_KEY.lower() in msg
            or "huggingface" in msg
            or "hf_hub" in msg
        )
        if allow_online_models and chronos_related and CHRONOS_MODEL_KEY in hyperparameters:
            logger.warning(
                "Chronos download/runtime failed (%s). Retrying without Chronos.", exc
            )
            retry_hyperparameters = {
                k: v for k, v in hyperparameters.items() if k != CHRONOS_MODEL_KEY
            }
            _fit_with(retry_hyperparameters)
        else:
            raise

    logger.info("Forecasting on holdout horizon...")
    predictions = predictor.predict(train_tsdf, known_covariates=future_known_cov)

    metrics = evaluate_forecast(
        predictions=predictions,
        test_df=test_df,
        target_col=target_col,
        prediction_length=prediction_length,
    )

    logger.info("Evaluation metrics on holdout horizon:\n%s", json.dumps(metrics, indent=2))

    pred_out = model_dir / "holdout_predictions.csv"
    pred_readable_out = model_dir / "holdout_predictions_readable.csv"
    per_tank_metrics_out = model_dir / "holdout_per_tank_metrics.csv"
    metrics_out = model_dir / "holdout_metrics.json"
    prepared_out = model_dir / "prepared_hourly_data.csv"

    predictions.reset_index().to_csv(pred_out, index=False)

    readable_pred_df = build_readable_holdout_predictions(
        predictions=predictions,
        test_df=test_df,
        target_col=target_col,
        prediction_length=prediction_length,
    )
    readable_pred_df.to_csv(pred_readable_out, index=False)

    per_tank_metrics_df = build_per_tank_metrics(readable_pred_df)
    per_tank_metrics_df.to_csv(per_tank_metrics_out, index=False)

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    feat_df.to_csv(prepared_out, index=False)

    logger.info("Saved predictions         : %s", pred_out)
    logger.info("Saved readable predictions: %s", pred_readable_out)
    logger.info("Saved per-tank metrics    : %s", per_tank_metrics_out)
    logger.info("Saved metrics             : %s", metrics_out)
    logger.info("Saved prepared data       : %s", prepared_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe AutoGluon hourly forecasting from tank JSON files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Root dataset directory containing one subfolder per tank.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "results/autogluon/hourly_forecasting",
        help="Directory where model artifacts and outputs are stored.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=24,
        help="Forecast horizon in hours.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=1800,
        help="AutoGluon training time limit in seconds.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=TARGET_COL,
        help="Target column to forecast.",
    )
    parser.add_argument(
        "--max-tanks",
        type=int,
        default=None,
        help="Optional: limit number of tank folders for quick smoke tests.",
    )
    parser.add_argument(
        "--max-files-per-tank",
        type=int,
        default=None,
        help="Optional: limit number of day JSON files per tank for quick smoke tests.",
    )
    parser.add_argument(
        "--min-history-hours",
        type=int,
        default=24 * 14,
        help="Minimum historical points per tank required before holdout split.",
    )
    parser.add_argument(
        "--allow-online-models",
        action="store_true",
        help="Enable internet-dependent models (Chronos/deep models) in addition to offline baselines.",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="medium_quality",
        choices=["fast_training", "medium_quality", "high_quality", "best_quality"],
        help="AutoGluon preset. Use high/best quality when online models are enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_autogluon_hourly_forecaster(
        dataset_dir=args.dataset_dir.resolve(),
        model_dir=args.model_dir.resolve(),
        prediction_length=args.prediction_length,
        time_limit=args.time_limit,
        target_col=args.target_col,
        max_tanks=args.max_tanks,
        max_files_per_tank=args.max_files_per_tank,
        min_history_hours=args.min_history_hours,
        allow_online_models=args.allow_online_models,
        presets=args.presets,
    )


if __name__ == "__main__":
    main()