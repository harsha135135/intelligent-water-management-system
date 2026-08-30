import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from autogluon_hourly_forecasting import (
    ITEM_ID_COL,
    KNOWN_COVARIATES,
    PROJECT_ROOT,
    TARGET_COL,
    TIMESTAMP_COL,
    add_leakage_safe_features,
    filter_viable_series,
    load_hourly_json_dataset,
    make_tsdf,
)


logger = logging.getLogger(__name__)


def create_future_covariates(
    history_data: TimeSeriesDataFrame,
    prediction_length: int,
) -> TimeSeriesDataFrame:
    """Generate known covariates (calendar features) for future time steps."""
    future_rows = []

    for item_id in history_data.item_ids:
        ts_index = history_data.loc[item_id].index
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


def clean_anomalies_with_moving_average(
    df: pd.DataFrame,
    target_col: str,
    rolling_window: int,
    z_threshold: float,
    anomaly_context_hours: int,
    practical_max_kl: float | None,
    practical_max_multiplier: float,
    jump_z_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace anomalous points with rolling mean per tank.

        Practical-only anomaly policy:
        - Replace only implausible values (negative or above practical cap).
        - Preserve real demand spikes even if they are statistically rare.
        - Sudden-jump detection is only considered if the point is already
            outside practical limits.
    """
    cleaned_parts = []
    anomaly_parts = []

    for item_id, group in df.groupby(ITEM_ID_COL, sort=True):
        g = group.sort_values(TIMESTAMP_COL).copy()

        y = g[target_col].astype(float)
        # Use only recent history to estimate practical limits and robust fallback
        # stats, so replacement logic adapts to recent operating patterns.
        context_y = y.tail(max(1, anomaly_context_hours))
        rolling_mean = y.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).mean()
        rolling_std = y.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).std()

        fallback_mean = y.rolling(window=rolling_window, min_periods=1).mean()
        rolling_mean = rolling_mean.fillna(fallback_mean)

        median_std = float(rolling_std.median()) if rolling_std.notna().any() else 0.0
        if not np.isfinite(median_std) or median_std <= 0:
            context_std = float(context_y.std()) if np.isfinite(context_y.std()) else 0.0
            median_std = context_std if context_std > 0 else 1.0
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

        # Statistical rules are retained for diagnostics only, not replacement.
        rule_rolling = ((y < low) | (y > high)).fillna(False)
        rule_quantile = ((y < q_low) | (y > q_high)).fillna(False)

        # Replace only values that are not practically plausible.
        non_physical_negative = (y < 0).fillna(False)
        hard_practical_anomaly = (practical_mask | non_physical_negative).fillna(False)

        # Jump anomalies are replaced only if they also violate practical cap.
        jump_practical_anomaly = (jump_mask & practical_mask).fillna(False)

        anomaly_mask = (hard_practical_anomaly | jump_practical_anomaly).fillna(False)

        replaced = y.where(~anomaly_mask, rolling_mean).clip(lower=0)

        g[f"{target_col}_raw"] = y
        g[target_col] = replaced
        g["is_anomaly"] = anomaly_mask.astype(int)
        g["anomaly_threshold_low"] = low
        g["anomaly_threshold_high"] = high
        g["practical_cap"] = practical_cap
        g["jump_threshold"] = jump_threshold
        g["anomaly_rule_rolling"] = rule_rolling.astype(int)
        g["anomaly_rule_quantile"] = rule_quantile.astype(int)
        g["anomaly_rule_practical_cap"] = practical_mask.astype(int)
        g["anomaly_rule_jump"] = jump_mask.astype(int)
        g["anomaly_rule_non_physical_negative"] = non_physical_negative.astype(int)
        g["anomaly_policy_practical_only"] = 1
        g["anomaly_replacement"] = rolling_mean

        cleaned_parts.append(g)
        anomaly_parts.append(
            g.loc[
                g["is_anomaly"] == 1,
                [
                    ITEM_ID_COL,
                    TIMESTAMP_COL,
                    f"{target_col}_raw",
                    target_col,
                    "anomaly_threshold_low",
                    "anomaly_threshold_high",
                    "practical_cap",
                    "jump_threshold",
                    "anomaly_rule_rolling",
                    "anomaly_rule_quantile",
                    "anomaly_rule_practical_cap",
                    "anomaly_rule_jump",
                    "anomaly_rule_non_physical_negative",
                ],
            ]
        )

    cleaned_df = pd.concat(cleaned_parts, ignore_index=True)
    anomaly_df = pd.concat(anomaly_parts, ignore_index=True)
    return cleaned_df, anomaly_df


def train_anomaly_aware_ensemble(
    dataset_dir: Path,
    model_dir: Path,
    prediction_length: int,
    time_limit: int,
    target_col: str,
    rolling_window: int,
    z_threshold: float,
    anomaly_context_hours: int,
    practical_max_kl: float | None,
    practical_max_multiplier: float,
    jump_z_threshold: float,
    min_history_hours: int,
    max_epochs: int,
    num_val_windows: int,
    refit_full: bool,
    training_profile: str,
    presets: str,
    enable_deep_models: bool,
    max_tanks: int | None,
    max_files_per_tank: int | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("Loading full hourly dataset from %s", dataset_dir)
    raw_df = load_hourly_json_dataset(
        dataset_dir,
        max_tanks=max_tanks,
        max_files_per_tank=max_files_per_tank,
    )

    logger.info("Rows loaded=%d, tanks=%d", len(raw_df), raw_df[ITEM_ID_COL].nunique())
    raw_df = raw_df.sort_values([ITEM_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)

    logger.info(
        "Cleaning anomalies using rolling window=%d, z-threshold=%.2f, context=%dh",
        rolling_window,
        z_threshold,
        anomaly_context_hours,
    )
    clean_df, anomaly_df = clean_anomalies_with_moving_average(
        raw_df,
        target_col=target_col,
        rolling_window=rolling_window,
        z_threshold=z_threshold,
        anomaly_context_hours=anomaly_context_hours,
        practical_max_kl=practical_max_kl,
        practical_max_multiplier=practical_max_multiplier,
        jump_z_threshold=jump_z_threshold,
    )

    feat_df = add_leakage_safe_features(clean_df)
    feat_df = filter_viable_series(
        feat_df,
        prediction_length=prediction_length,
        min_history=min_history_hours,
    )

    # Full-data training: no holdout split, model learns from all available history.
    train_tsdf = make_tsdf(feat_df, target_col=target_col)

    model_dir.mkdir(parents=True, exist_ok=True)

    predictor = TimeSeriesPredictor(
        path=str(model_dir),
        target=target_col,
        prediction_length=prediction_length,
        freq="h",
        eval_metric="MASE",
        known_covariates_names=KNOWN_COVARIATES,
    )

    if training_profile == "heavy":
        patchtst_epochs = sorted({max_epochs, 200, 300, 400})
        hyperparameters = {
            "PatchTST": [{"max_epochs": e} for e in patchtst_epochs],
            "NPTS": [{}],
            "ETS": [{}],
            "Theta": [{}],
            "SeasonalNaive": [{}],
        }

        if enable_deep_models:
            hyperparameters.update(
                {
                    "TemporalFusionTransformer": [
                        {"max_epochs": 100},
                        {"max_epochs": 150},
                    ],
                    "DeepAR": [
                        {"max_epochs": 120},
                        {"max_epochs": 180},
                    ],
                    "TiDE": [
                        {"max_epochs": 100},
                        {"max_epochs": 150},
                    ],
                }
            )
    else:
        hyperparameters = {
            "PatchTST": {"max_epochs": max_epochs},
            "NPTS": {},
            "ETS": {},
            "Theta": {},
            "SeasonalNaive": {},
        }

    logger.info("Training anomaly-aware ensemble model...")
    predictor.fit(
        train_data=train_tsdf,
        presets=presets,
        time_limit=time_limit,
        hyperparameters=hyperparameters,
        num_val_windows=num_val_windows,
        refit_full=refit_full,
        verbosity=2,
    )

    leaderboard_df = predictor.leaderboard(silent=True)
    leaderboard_out = model_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_out, index=False)

    best_model_name = str(leaderboard_df.iloc[0]["model"]) if not leaderboard_df.empty else "unknown"
    best_score_val = (
        float(leaderboard_df.iloc[0]["score_val"])
        if (not leaderboard_df.empty and "score_val" in leaderboard_df.columns)
        else None
    )

    logger.info("Generating next %d-hour future forecast", prediction_length)
    known_covariates = create_future_covariates(train_tsdf, prediction_length=prediction_length)
    predictions = predictor.predict(train_tsdf, known_covariates=known_covariates)

    pred_df = predictions.reset_index().copy()
    pred_df = pred_df.rename(columns={"mean": "pred_mean"})
    if "0.1" in pred_df.columns:
        pred_df = pred_df.rename(columns={"0.1": "pred_p10"})
    if "0.5" in pred_df.columns:
        pred_df = pred_df.rename(columns={"0.5": "pred_p50"})
    if "0.9" in pred_df.columns:
        pred_df = pred_df.rename(columns={"0.9": "pred_p90"})

    predictions.reset_index().to_csv(model_dir / "future_predictions.csv", index=False)
    pred_df.to_csv(model_dir / "future_predictions_readable.csv", index=False)
    feat_df.to_csv(model_dir / "prepared_hourly_data_cleaned.csv", index=False)
    anomaly_df.to_csv(model_dir / "anomalies_replaced.csv", index=False)

    summary = {
        "rows_total": int(len(raw_df)),
        "tanks_total": int(raw_df[ITEM_ID_COL].nunique()),
        "rows_after_filter": int(len(feat_df)),
        "tanks_after_filter": int(feat_df[ITEM_ID_COL].nunique()),
        "anomalies_replaced": int(len(anomaly_df)),
        "anomaly_rule_counts": {
            "rolling": int(anomaly_df["anomaly_rule_rolling"].sum()) if not anomaly_df.empty else 0,
            "quantile": int(anomaly_df["anomaly_rule_quantile"].sum()) if not anomaly_df.empty else 0,
            "practical_cap": int(anomaly_df["anomaly_rule_practical_cap"].sum()) if not anomaly_df.empty else 0,
            "jump": int(anomaly_df["anomaly_rule_jump"].sum()) if not anomaly_df.empty else 0,
            "non_physical_negative": int(anomaly_df["anomaly_rule_non_physical_negative"].sum()) if not anomaly_df.empty else 0,
        },
        "anomaly_policy": "practical_only",
        "prediction_length": int(prediction_length),
        "trained_on_full_history": True,
        "best_model": best_model_name,
        "best_score_val_flipped": best_score_val,
        "best_metric": "MASE",
        "best_mase": None if best_score_val is None else float(-best_score_val),
        "rolling_window": int(rolling_window),
        "z_threshold": float(z_threshold),
        "anomaly_context_hours": int(anomaly_context_hours),
        "practical_max_kl": None if practical_max_kl is None else float(practical_max_kl),
        "practical_max_multiplier": float(practical_max_multiplier),
        "jump_z_threshold": float(jump_z_threshold),
        "training_profile": training_profile,
        "num_val_windows": int(num_val_windows),
        "refit_full": bool(refit_full),
        "enable_deep_models": bool(enable_deep_models),
    }

    with (model_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved model and future predictions in: %s", model_dir)
    logger.info("Training summary: %s", json.dumps(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train anomaly-aware AutoGluon ensemble on full history and generate next-horizon forecast."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "results/autogluon/anomaly_ensemble_full",
    )
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--target-col", type=str, default=TARGET_COL)
    parser.add_argument("--rolling-window", type=int, default=24)
    parser.add_argument("--z-threshold", type=float, default=4.5)
    parser.add_argument(
        "--anomaly-context-hours",
        type=int,
        default=24 * 180,
        help="Recent history window used to estimate practical anomaly caps and robust stats.",
    )
    parser.add_argument(
        "--practical-max-kl",
        type=float,
        default=None,
        help="Absolute practical maximum outflow value in KL. Values beyond are treated as anomalies.",
    )
    parser.add_argument(
        "--practical-max-multiplier",
        type=float,
        default=1.3,
        help="If practical-max-kl is not set, cap is computed as q99 * multiplier per tank.",
    )
    parser.add_argument(
        "--jump-z-threshold",
        type=float,
        default=3.0,
        help="Sensitivity for sudden jump detection on hourly changes.",
    )
    parser.add_argument("--min-history-hours", type=int, default=24 * 14)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument(
        "--training-profile",
        type=str,
        default="heavy",
        choices=["baseline", "heavy"],
        help="baseline: fast search. heavy: larger search space and deeper models.",
    )
    parser.add_argument(
        "--num-val-windows",
        type=int,
        default=3,
        help="Number of rolling validation windows used by AutoGluon.",
    )
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="Refit selected model on full data after validation.",
    )
    parser.add_argument(
        "--enable-deep-models",
        action="store_true",
        help="Include TFT, DeepAR, TiDE in heavy profile search.",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="high_quality",
        choices=["fast_training", "medium_quality", "high_quality", "best_quality"],
    )
    parser.add_argument("--max-tanks", type=int, default=None)
    parser.add_argument("--max-files-per-tank", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_anomaly_aware_ensemble(
        dataset_dir=args.dataset_dir.resolve(),
        model_dir=args.model_dir.resolve(),
        prediction_length=args.prediction_length,
        time_limit=args.time_limit,
        target_col=args.target_col,
        rolling_window=args.rolling_window,
        z_threshold=args.z_threshold,
        anomaly_context_hours=args.anomaly_context_hours,
        practical_max_kl=args.practical_max_kl,
        practical_max_multiplier=args.practical_max_multiplier,
        jump_z_threshold=args.jump_z_threshold,
        min_history_hours=args.min_history_hours,
        max_epochs=args.max_epochs,
        num_val_windows=args.num_val_windows,
        refit_full=args.refit_full,
        training_profile=args.training_profile,
        presets=args.presets,
        enable_deep_models=args.enable_deep_models,
        max_tanks=args.max_tanks,
        max_files_per_tank=args.max_files_per_tank,
    )


if __name__ == "__main__":
    main()
