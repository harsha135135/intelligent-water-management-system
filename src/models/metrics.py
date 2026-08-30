"""Forecast accuracy metrics: MAE, RMSE, MASE, RMSSE.

Why scaled metrics matter for this dataset
------------------------------------------
Per-tank mean hourly outflow spans 0.0012 KL/h (``NEW_BLOCK_RO``) to 1.67 KL/h
(``BE_BLOCK_OHT``) - three orders of magnitude. A raw MAE average across tanks is therefore
dominated by the four largest tanks, and a dead sensor scores a *perfect* MAE of 0.0 simply by
never moving. MASE and RMSSE divide each series' error by that series' own seasonal-naive error,
which makes the numbers comparable across tanks and immune to that failure mode.

Definitions
-----------
For a series with in-sample history ``y_1..y_n`` and seasonal period ``m``::

    scale_mae = mean_{t>m} |y_t - y_{t-m}|
    scale_mse = mean_{t>m} (y_t - y_{t-m})^2

    MAE   = mean |y - yhat|
    RMSE  = sqrt(mean (y - yhat)^2)
    MASE  = mean( |y - yhat| / scale_mae )
    RMSSE = sqrt( mean( (y - yhat)^2 / scale_mse ) )

The scale is always computed on the history *preceding the forecast origin*, never on the
evaluation window. ``m = 24`` for hourly campus water demand (daily seasonality).

Degenerate series
-----------------
A constant series has ``scale_mae == 0`` and its MASE is undefined (not infinite, not zero).
``NEW_BLOCK_RO`` is constant-zero for 96% of its history. Such series are **excluded** from
scaled-metric aggregates and reported separately - never silently patched with an epsilon,
which would manufacture an arbitrary and flattering number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASONAL_PERIOD = 24

PRED_SCHEMA = [
    "model", "item_id", "origin", "horizon", "step", "timestamp", "actual", "pred",
]


def seasonal_scales(history: np.ndarray, m: int = SEASONAL_PERIOD) -> tuple[float, float]:
    """Seasonal-naive scaling denominators from an in-sample history.

    NaNs are ignored pairwise, so gaps in the sensor record reduce the sample size rather than
    poisoning the whole scale.
    """
    h = np.asarray(history, dtype=float)
    if h.size <= m:
        return np.nan, np.nan
    diffs = h[m:] - h[:-m]
    if np.all(np.isnan(diffs)):
        return np.nan, np.nan
    scale_mae = float(np.nanmean(np.abs(diffs)))
    scale_mse = float(np.nanmean(np.square(diffs)))
    return scale_mae, scale_mse


def attach_scales(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    target_col: str = "Outflow in KL",
    m: int = SEASONAL_PERIOD,
) -> pd.DataFrame:
    """Attach (scale_mae, scale_mse) to each (item_id, origin) using pre-origin history only."""
    keys = predictions[["item_id", "origin"]].drop_duplicates()
    panel = panel.sort_values(["item_id", "timestamp"])
    by_item = {i: g for i, g in panel.groupby("item_id", sort=False)}

    rows = []
    for item_id, origin in keys.itertuples(index=False):
        g = by_item[item_id]
        history = g.loc[g["timestamp"] <= origin, target_col].to_numpy()
        s_mae, s_mse = seasonal_scales(history, m=m)
        rows.append({
            "item_id": item_id, "origin": origin,
            "scale_mae": s_mae, "scale_mse": s_mse,
            "history_len": len(history),
        })
    return predictions.merge(pd.DataFrame(rows), on=["item_id", "origin"], how="left")


def _agg(errors: pd.DataFrame) -> pd.Series:
    err = errors["actual"] - errors["pred"]
    abs_err = err.abs()
    sq_err = err.pow(2)

    usable = errors["scale_mae"].notna() & (errors["scale_mae"] > 0)
    usable_sq = errors["scale_mse"].notna() & (errors["scale_mse"] > 0)

    return pd.Series({
        "n": int(len(errors)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "mase": float((abs_err[usable] / errors.loc[usable, "scale_mae"]).mean())
        if usable.any() else np.nan,
        "rmsse": float(np.sqrt((sq_err[usable_sq] / errors.loc[usable_sq, "scale_mse"]).mean()))
        if usable_sq.any() else np.nan,
        "mean_actual": float(errors["actual"].mean()),
        "mean_pred": float(errors["pred"].mean()),
        "scaled_ok": bool(usable.any()),
    })


def per_tank_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Metrics for every (model, horizon, item_id)."""
    scored = scored.dropna(subset=["actual", "pred"])
    out = (
        scored.groupby(["model", "horizon", "item_id"], sort=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return out


def aggregate_metrics(per_tank: pd.DataFrame) -> pd.DataFrame:
    """Roll per-tank metrics up to a headline table per (model, horizon).

    Two aggregations are reported side by side because they answer different questions:

    * ``macro_*``  - unweighted mean across tanks. The honest headline for MASE/RMSSE, since
      those are already scale-free. Every tank counts once.
    * ``vw_mae`` / ``vw_rmse`` - weighted by each tank's mean actual demand. Answers "how many
      KL are we wrong by across campus", which is what an operator cares about.

    ``n_tanks_scaled`` reports how many tanks contributed to the scaled metrics, so an excluded
    degenerate series is always visible rather than silently dropped.
    """
    rows = []
    for (model, horizon), g in per_tank.groupby(["model", "horizon"], sort=False):
        w = g["mean_actual"].to_numpy()
        wsum = w.sum()
        scaled = g[g["scaled_ok"]]
        rows.append({
            "model": model,
            "horizon": horizon,
            "n_tanks": int(len(g)),
            "n_tanks_scaled": int(len(scaled)),
            "rows_evaluated": int(g["n"].sum()),
            "macro_mae": float(g["mae"].mean()),
            "macro_rmse": float(g["rmse"].mean()),
            "macro_mase": float(scaled["mase"].mean()) if len(scaled) else np.nan,
            "macro_rmsse": float(scaled["rmsse"].mean()) if len(scaled) else np.nan,
            "vw_mae": float((g["mae"] * w).sum() / wsum) if wsum > 0 else np.nan,
            "vw_rmse": float((g["rmse"] * w).sum() / wsum) if wsum > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["horizon", "macro_mase"]).reset_index(drop=True)


def coverage(scored: pd.DataFrame, lo: str = "0.1", hi: str = "0.9") -> pd.DataFrame:
    """Empirical coverage of a predictive interval - a probabilistic model that is well
    calibrated should land near the nominal 0.8 for a p10-p90 band."""
    if lo not in scored.columns or hi not in scored.columns:
        return pd.DataFrame()
    s = scored.dropna(subset=["actual", lo, hi])
    inside = (s["actual"] >= s[lo]) & (s["actual"] <= s[hi])
    return (
        s.assign(inside=inside.astype(int))
        .groupby(["model", "horizon"], sort=False)["inside"]
        .mean().reset_index().rename(columns={"inside": "p10_p90_coverage"})
    )
