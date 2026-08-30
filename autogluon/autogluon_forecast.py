"""
autogluon_forecast.py — Reports AutoGluon TimeSeries forecasting results.
Uses the existing trained model's holdout predictions to compute overall MAE.

The AutoGluon model was previously trained with:
  - Hourly data across 24 tanks
  - prediction_length=24 (24-hour holdout)
  - eval_metric=MAE
  - presets=medium_quality
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = os.path.dirname(__file__)
AG_RESULTS = os.path.join(BASE, "waltr_data", "results", "autogluon", "hourly_forecasting")

# ── Load existing results ─────────────────────────────────────────────
print("Loading AutoGluon holdout results...")

# Per-tank metrics
per_tank = pd.read_csv(os.path.join(AG_RESULTS, "holdout_per_tank_metrics.csv"))
print(f"  Per-tank results: {len(per_tank)} tanks")

# Holdout predictions (contains mean predictions + quantiles)
preds = pd.read_csv(os.path.join(AG_RESULTS, "holdout_predictions.csv"))
print(f"  Holdout predictions: {len(preds)} rows")

# Load the prepared hourly data to get actual values for the holdout period
hourly_data = pd.read_csv(
    os.path.join(AG_RESULTS, "prepared_hourly_data.csv"),
    parse_dates=["timestamp"]
)

target_col = "Outflow in KL"
tanks = hourly_data["item_id"].unique()
prediction_length = 24

# ── Compute actuals vs predictions for holdout ────────────────────────
print("\nComputing holdout metrics from predictions...")

all_preds = []
for tank in sorted(tanks):
    tank_hourly = hourly_data[hourly_data["item_id"] == tank].sort_values("timestamp")
    actuals = tank_hourly[target_col].values[-prediction_length:]
    timestamps = tank_hourly["timestamp"].values[-prediction_length:]

    tank_preds = preds[preds["item_id"] == tank].sort_values("timestamp")
    predicted = np.maximum(tank_preds["mean"].values, 0)  # clip negatives

    for i in range(min(len(actuals), len(predicted))):
        all_preds.append({
            "tank": tank,
            "timestamp": timestamps[i],
            "actual": actuals[i],
            "predicted": predicted[i],
        })

results_df = pd.DataFrame(all_preds)

# ── Per-Tank Results ──────────────────────────────────────────────────
print("\n" + "="*70)
print("AutoGluon RESULTS — Per-Tank Holdout (Last 24 Hours)")
print("="*70)
print(per_tank[["item_id", "mae", "rmse", "mean_actual", "mean_pred"]].to_string(index=False))

# ── Overall Metrics ───────────────────────────────────────────────────
overall_mae = mean_absolute_error(results_df["actual"], results_df["predicted"])
overall_rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["predicted"]))

print(f"\n{'='*70}")
print(f"OVERALL METRICS (all tanks combined)")
print(f"{'='*70}")
print(f"  MAE:  {overall_mae:.6f}")
print(f"  RMSE: {overall_rmse:.6f}")
print(f"  Total test rows: {len(results_df)}")
print(f"  Tanks evaluated: {results_df['tank'].nunique()}")

# ── Comparison with LightGBM ─────────────────────────────────────────
lgbm_path = os.path.join(BASE, "lgbm_predictions.csv")
if os.path.exists(lgbm_path):
    lgbm_preds = pd.read_csv(lgbm_path)
    lgbm_mae = mean_absolute_error(lgbm_preds["actual"], lgbm_preds["predicted"])
    lgbm_rmse = np.sqrt(mean_squared_error(lgbm_preds["actual"], lgbm_preds["predicted"]))

    print(f"\n{'='*70}")
    print(f"COMPARISON: AutoGluon vs LightGBM")
    print(f"{'='*70}")
    print(f"  AutoGluon:  MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
    print(f"  LightGBM:   MAE={lgbm_mae:.4f}, RMSE={lgbm_rmse:.4f}")
    improvement_mae = (overall_mae - lgbm_mae) / overall_mae * 100
    improvement_rmse = (overall_rmse - lgbm_rmse) / overall_rmse * 100
    print(f"  LightGBM MAE  improvement over AutoGluon: {improvement_mae:+.1f}%")
    print(f"  LightGBM RMSE improvement over AutoGluon: {improvement_rmse:+.1f}%")

print("\nDone!")
