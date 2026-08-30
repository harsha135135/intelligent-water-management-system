import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor

from autogluon_hourly_forecasting import (
    ITEM_ID_COL,
    KNOWN_COVARIATES,
    PROJECT_ROOT,
    TARGET_COL,
    TIMESTAMP_COL,
    add_leakage_safe_features,
    build_future_known_covariates,
    build_readable_holdout_predictions,
    evaluate_forecast,
    filter_viable_series,
    load_hourly_json_dataset,
    make_tsdf,
    split_train_test,
)


logger = logging.getLogger(__name__)

# Matches any character that is not alphanumeric, hyphen, or underscore.
_UNSAFE_PATH_CHARS = re.compile(r"[^\w\-]")


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Configure the root logger so that shared-module loggers (e.g.
    # autogluon_hourly_forecasting) also write to the log file.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # Prevent double-emission if a parent logger also has handlers.
    logger.propagate = False


def _rmse(series: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(series))))


def build_per_tank_metrics(readable_pred_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate holdout MAE and RMSE per tank, consistent with the base module."""
    summary = (
        readable_pred_df
        .groupby(ITEM_ID_COL, as_index=False)
        .agg(
            rows_evaluated=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("error", _rmse),
            mean_actual=("actual", "mean"),
            mean_pred=("pred_mean", "mean"),
        )
        .sort_values("mae", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def _safe_tank_filename(tank_id: str) -> str:
    """Sanitise a tank ID so it is safe as a filename on all platforms."""
    return _UNSAFE_PATH_CHARS.sub("_", str(tank_id))


def save_holdout_predictions_per_tank(readable_pred_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for tank_id, tank_df in readable_pred_df.groupby(ITEM_ID_COL, sort=True):
        safe_tank = _safe_tank_filename(tank_id)
        tank_out = output_dir / f"{safe_tank}_holdout_predictions.csv"
        tank_df.sort_values(TIMESTAMP_COL).to_csv(tank_out, index=False)


def train_patchtst_hourly_forecaster(
    dataset_dir: Path,
    model_dir: Path,
    prediction_length: int,
    time_limit: int,
    target_col: str,
    max_tanks: int | None,
    max_files_per_tank: int | None,
    min_history_hours: int,
    presets: str,
    max_epochs: int,
) -> None:
    log_file = model_dir / "patchtst_training.log"
    configure_logging(log_file)

    logger.info("Starting PatchTST hourly forecasting run")
    logger.info("Dataset directory: %s", dataset_dir)
    logger.info("Model directory  : %s", model_dir)

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

    feat_df = add_leakage_safe_features(raw_df)
    feat_df = filter_viable_series(
        feat_df,
        prediction_length=prediction_length,
        min_history=min_history_hours,
    )

    train_df, test_df = split_train_test(feat_df, prediction_length=prediction_length)

    # Leakage guard: model is fit strictly on history; only horizon covariates are provided.
    train_tsdf = make_tsdf(train_df, target_col=target_col)
    future_known_cov = build_future_known_covariates(
        test_df,
        prediction_length=prediction_length,
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    predictor = TimeSeriesPredictor(
        path=str(model_dir),
        target=target_col,
        prediction_length=prediction_length,
        freq="h",
        eval_metric="MAE",
        known_covariates_names=KNOWN_COVARIATES,
    )

    hyperparameters = {
        "PatchTST": {
            "max_epochs": max_epochs,
        }
    }

    logger.info("Training PatchTST model (time_limit=%ds, max_epochs=%d)...", time_limit, max_epochs)
    try:
        predictor.fit(
            train_data=train_tsdf,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=hyperparameters,
            num_val_windows=1,
            verbosity=2,
        )
    except Exception:
        logger.exception("PatchTST training failed. See traceback above.")
        raise

    logger.info("Generating holdout forecasts...")
    predictions = predictor.predict(train_tsdf, known_covariates=future_known_cov)

    overall_metrics = evaluate_forecast(
        predictions=predictions,
        test_df=test_df,
        target_col=target_col,
        prediction_length=prediction_length,
    )

    readable_pred_df = build_readable_holdout_predictions(
        predictions=predictions,
        test_df=test_df,
        target_col=target_col,
        prediction_length=prediction_length,
    )

    per_tank_metrics_df = build_per_tank_metrics(readable_pred_df)

    pred_out = model_dir / "holdout_predictions.csv"
    pred_readable_out = model_dir / "holdout_predictions_readable.csv"
    per_tank_metrics_out = model_dir / "holdout_per_tank_metrics.csv"
    metrics_out = model_dir / "holdout_metrics.json"
    prepared_out = model_dir / "prepared_hourly_data.csv"
    per_tank_pred_dir = model_dir / "holdout_predictions_per_tank"

    predictions.reset_index().to_csv(pred_out, index=False)
    readable_pred_df.to_csv(pred_readable_out, index=False)
    per_tank_metrics_df.to_csv(per_tank_metrics_out, index=False)
    feat_df.to_csv(prepared_out, index=False)
    save_holdout_predictions_per_tank(readable_pred_df, per_tank_pred_dir)

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    logger.info("Overall holdout metrics: %s", json.dumps(overall_metrics))
    logger.info("Per-tank MAE / RMSE:")
    for row in per_tank_metrics_df.itertuples(index=False):
        row_dict = row._asdict()
        logger.info(
            "tank=%s rows=%d mae=%.6f rmse=%.6f",
            row_dict[ITEM_ID_COL],
            int(row_dict["rows_evaluated"]),
            float(row_dict["mae"]),
            float(row_dict["rmse"]),
        )

    logger.info("Saved predictions         : %s", pred_out)
    logger.info("Saved readable predictions: %s", pred_readable_out)
    logger.info("Saved per-tank metrics    : %s", per_tank_metrics_out)
    logger.info("Saved metrics             : %s", metrics_out)
    logger.info("Saved prepared data       : %s", prepared_out)
    logger.info("Saved per-tank predictions: %s", per_tank_pred_dir)
    logger.info("Saved log file            : %s", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe PatchTST hourly forecasting from tank JSON files."
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
        default=PROJECT_ROOT / "results/patchtst/hourly_forecasting",
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
        default=3600,
        help="Training time limit in seconds.",
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
        "--presets",
        type=str,
        default="high_quality",
        choices=["fast_training", "medium_quality", "high_quality", "best_quality"],
        help="AutoGluon training preset.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum PatchTST training epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_patchtst_hourly_forecaster(
        dataset_dir=args.dataset_dir.resolve(),
        model_dir=args.model_dir.resolve(),
        prediction_length=args.prediction_length,
        time_limit=args.time_limit,
        target_col=args.target_col,
        max_tanks=args.max_tanks,
        max_files_per_tank=args.max_files_per_tank,
        min_history_hours=args.min_history_hours,
        presets=args.presets,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()