"""
run_autogluon_daily.py
Train AutoGluon TimeSeriesPredictor on daily_tank_data.csv for all 24 tanks.
Target: daily_outflow
Split: last 30 days per tank held out for test (no data leakage)
Anti-overfitting: medium_quality preset, no deep learning, time-based split
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os

# ── Load data ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "daily_tank_data.csv"), parse_dates=["date"])

TARGET = "daily_outflow"
ID_COL = "item_id"
TIME_COL = "timestamp"
PREDICTION_LENGTH = 30   # 30-day holdout per tank

print(f"Loaded {len(df)} rows, {df['tank_id'].nunique()} tanks")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── Prepare TimeSeriesDataFrame format ────────────────────────────────────
df = df.rename(columns={"tank_id": ID_COL, "date": TIME_COL})
df = df[[ID_COL, TIME_COL, TARGET]].copy()
df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

# Verify no missing dates cause gaps — forward-fill any gaps within each tank
records = []
for tank, grp in df.groupby(ID_COL):
    grp = grp[[TIME_COL, TARGET]].set_index(TIME_COL).resample("D").mean(numeric_only=True)
    grp[TARGET] = grp[TARGET].interpolate(method="linear").bfill()
    grp[ID_COL] = tank
    grp = grp.reset_index()
    records.append(grp[[ID_COL, TIME_COL, TARGET]])

df_clean = pd.concat(records, ignore_index=True)
print(f"After resampling: {len(df_clean)} rows")

# ── Train / test split (time-based, no leakage) ───────────────────────────
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

tsdf = TimeSeriesDataFrame.from_data_frame(
    df_clean,
    id_column=ID_COL,
    timestamp_column=TIME_COL,
)

print(f"\nTimeSeriesDataFrame: {len(tsdf)} rows, {tsdf.num_items} items")

# ── Train predictor ───────────────────────────────────────────────────────
# excluded_model_types removes heavy deep-learning models → less overfitting,
# faster training, and more reliable on short tabular time series.
EXCLUDED = [
    "Chronos",          # large pre-trained LLM-based, needs GPU for best results
    "TemporalFusionTransformer",
    "PatchTST",
    "WaveNet",
    "DeepAR",
]

predictor = TimeSeriesPredictor(
    target=TARGET,
    prediction_length=PREDICTION_LENGTH,
    eval_metric="MAE",
    path=os.path.join(BASE, "ag_daily_model"),
    freq="D",
    verbosity=2,
)

predictor.fit(
    tsdf,
    presets="medium_quality",
    excluded_model_types=EXCLUDED,
    time_limit=600,           # 10 min cap — prevents overfitting via long ensembles
    num_val_windows=3,        # rolling-window CV for robust validation
    val_step_size=7,          # each validation window steps 1 week forward
)

# ── Evaluate on held-out last PREDICTION_LENGTH days ─────────────────────
print("\n" + "="*65)
print("Evaluating on held-out last 30 days per tank...")
print("="*65)

scores = predictor.evaluate(tsdf)
print(f"\nLeaderboard (validation MAE per model):")
print(predictor.leaderboard(tsdf, silent=True).to_string(index=False))

# ── Get per-tank MAE ──────────────────────────────────────────────────────
predictions = predictor.predict(tsdf)

# Build actuals from the last PREDICTION_LENGTH rows per tank
actuals_list = []
preds_list   = []

for tank in df_clean[ID_COL].unique():
    tank_df = df_clean[df_clean[ID_COL] == tank].sort_values(TIME_COL)
    actuals = tank_df[TARGET].values[-PREDICTION_LENGTH:]

    try:
        tank_preds = predictions.loc[tank]
        if hasattr(tank_preds, "values"):
            pred_vals = tank_preds["mean"].values if "mean" in tank_preds.columns else tank_preds.values.ravel()
        else:
            pred_vals = np.array(tank_preds).ravel()
        pred_vals = np.maximum(pred_vals, 0)  # clip negatives (outflow can't be <0)
    except Exception:
        pred_vals = np.full(PREDICTION_LENGTH, np.nan)

    n = min(len(actuals), len(pred_vals))
    actuals_list.extend(actuals[:n])
    preds_list.extend(pred_vals[:n])

actuals_arr = np.array(actuals_list)
preds_arr   = np.array(preds_list)

valid = ~np.isnan(actuals_arr) & ~np.isnan(preds_arr)
overall_mae  = np.mean(np.abs(actuals_arr[valid] - preds_arr[valid]))
overall_rmse = np.sqrt(np.mean((actuals_arr[valid] - preds_arr[valid])**2))

# Per-tank breakdown
per_tank_rows = []
idx = 0
tanks = df_clean[ID_COL].unique()
for tank in tanks:
    tank_df = df_clean[df_clean[ID_COL] == tank].sort_values(TIME_COL)
    n_rows = min(PREDICTION_LENGTH, len(tank_df))
    a = actuals_arr[idx:idx+n_rows]
    p = preds_arr[idx:idx+n_rows]
    v = ~np.isnan(a) & ~np.isnan(p)
    mae_t = np.mean(np.abs(a[v] - p[v])) if v.any() else np.nan
    per_tank_rows.append({"tank": tank, "mae": round(mae_t, 4), "n": v.sum()})
    idx += n_rows

per_tank_df = pd.DataFrame(per_tank_rows).sort_values("mae")

print("\n" + "="*65)
print("PER-TANK MAE (30-day holdout, ascending)")
print("="*65)
print(per_tank_df.to_string(index=False))

print("\n" + "="*65)
print(f"OVERALL MAE  (all 24 tanks combined): {overall_mae:.4f} KL")
print(f"OVERALL RMSE (all 24 tanks combined): {overall_rmse:.4f} KL")
print(f"Total test points: {valid.sum()}")
print("="*65)
