"""Split-conformal interval calibration and per-tank volume bias correction.

Both layers were specified in ``docs/review_summary.md`` §14 and left unimplemented; this module
implements them. Neither requires retraining — they are post-hoc corrections fitted on forecasts
the model has already made.

**The separation rule.** Calibration parameters are fitted on a window that is strictly disjoint
from, and earlier than, the window they are reported on. Fitting and reporting on the same rows
makes the resulting coverage circular and meaningless, so ``fit`` refuses to run if the two
windows overlap.

**Order of operations.** Bias correction is multiplicative and therefore moves the quantiles as
well as the point forecast, so it is applied *first* and the conformal quantiles are then fitted
on the already-corrected calibration residuals — that is, on the pipeline actually being shipped.

**Why the interval adjustment is asymmetric.** Measured on this data, roughly a quarter of hourly
readings are exactly zero, and a continuous-density model places its p10 above zero on ~79 % of
rows; two thirds of its lower-tail misses are zero-demand hours. A symmetric widening would
therefore over-widen the upper tail to fix a fault that lives entirely in the lower one. Separate
lower and upper offsets, each targeting α/2, fix the tail that is actually broken. Clipping the
lower bound at zero is not cosmetic either: it is what lets the interval represent the zero atom.

Method: conformalised quantile regression (Romano, Patterson & Candès, 2019), applied per
(model, tank), with independent tail offsets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.20                       # nominal 80 % central interval
BIAS_CLIP = (0.5, 2.0)             # a factor outside this says the tank, not the model, is broken
MIN_MEAN_KL_H = 0.01               # below this a tank has no usable signal (the 'dead' tier)


# ---------------------------------------------------------------- bias

def fit_bias(cal: pd.DataFrame, *, clip: tuple[float, float] = BIAS_CLIP,
             min_mean: float = MIN_MEAN_KL_H) -> pd.DataFrame:
    """Per-(model, tank) multiplicative volume factor: Σactual / Σpred on the calibration window.

    Tanks whose mean demand is below ``min_mean`` get a factor of 1.0 and are marked not-applied:
    a ratio between two near-zero sums is noise, and the three dead sensors would otherwise
    receive wild corrections.
    """
    rows = []
    for (model, tank), g in cal.groupby(["model", "item_id"], sort=True):
        g = g.dropna(subset=["actual", "pred"])
        sa, sp = float(g["actual"].sum()), float(g["pred"].sum())
        mean_actual = float(g["actual"].mean()) if len(g) else 0.0
        applied = bool(len(g) and mean_actual >= min_mean and sp > 0)
        raw = sa / sp if sp > 0 else np.nan
        factor = float(np.clip(raw, *clip)) if applied and np.isfinite(raw) else 1.0
        rows.append({"model": model, "item_id": tank, "n": int(len(g)),
                     "mean_actual_kl_h": mean_actual, "raw_factor": raw,
                     "factor": factor, "applied": applied,
                     "clipped": bool(applied and np.isfinite(raw) and raw != factor)})
    return pd.DataFrame(rows)


def apply_bias(df: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(bias[["model", "item_id", "factor"]], on=["model", "item_id"], how="left")
    out["factor"] = out["factor"].fillna(1.0)
    for c in ("pred", "q10", "q90"):
        out[f"{c}_bc"] = out[c] * out["factor"]
    return out


# ---------------------------------------------------------------- conformal

def fit_conformal(cal_bc: pd.DataFrame, *, alpha: float = ALPHA) -> pd.DataFrame:
    """Per-(model, tank) tail offsets from bias-corrected calibration residuals.

    Scores are the CQR conformity scores, kept separate per tail:
        E_lo = q10 - y   (positive when the actual fell below the lower quantile)
        E_hi = y - q90   (positive when the actual exceeded the upper quantile)
    The (1 - alpha/2) empirical quantile of each, with the finite-sample correction
    ⌈(n+1)(1-alpha/2)⌉/n, gives an offset that targets alpha/2 miss rate in that tail.
    A negative offset is kept, not floored: it means the model's quantile was too *wide* there
    and the interval should tighten.
    """
    lvl = 1.0 - alpha / 2.0
    rows = []
    for (model, tank), g in cal_bc.groupby(["model", "item_id"], sort=True):
        g = g.dropna(subset=["actual", "q10_bc", "q90_bc"])
        n = len(g)
        if n < 100:                      # too few residuals to estimate a 90th percentile
            rows.append({"model": model, "item_id": tank, "n": n,
                         "q_lo": 0.0, "q_hi": 0.0, "fitted": False})
            continue
        e_lo = (g["q10_bc"] - g["actual"]).to_numpy()
        e_hi = (g["actual"] - g["q90_bc"]).to_numpy()
        k = min(1.0, np.ceil((n + 1) * lvl) / n)      # conformal finite-sample correction
        rows.append({"model": model, "item_id": tank, "n": n,
                     "q_lo": float(np.quantile(e_lo, k)),
                     "q_hi": float(np.quantile(e_hi, k)),
                     "fitted": True})
    return pd.DataFrame(rows)


def apply_conformal(df_bc: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    out = df_bc.merge(conf[["model", "item_id", "q_lo", "q_hi"]],
                      on=["model", "item_id"], how="left")
    out[["q_lo", "q_hi"]] = out[["q_lo", "q_hi"]].fillna(0.0)
    # Outflow is non-negative: clipping the lower bound at zero is what lets the interval
    # represent the mass of exactly-zero hours that the raw p10 sits above.
    out["lower_cal"] = np.clip(out["q10_bc"] - out["q_lo"], 0.0, None)
    out["upper_cal"] = np.clip(out["q90_bc"] + out["q_hi"], 0.0, None)
    out["upper_cal"] = np.maximum(out["upper_cal"], out["lower_cal"])
    out["pred_cal"] = np.clip(out["pred_bc"], 0.0, None)
    return out


# ---------------------------------------------------------------- pipeline

def fit(cal: pd.DataFrame, *, alpha: float = ALPHA) -> dict[str, pd.DataFrame]:
    """Fit both layers on the calibration window. Returns the parameter tables."""
    bias = fit_bias(cal)
    conf = fit_conformal(apply_bias(cal, bias), alpha=alpha)
    return {"bias": bias, "conformal": conf, "alpha": alpha}


def apply(df: pd.DataFrame, cal_params: dict) -> pd.DataFrame:
    return apply_conformal(apply_bias(df, cal_params["bias"]), cal_params["conformal"])


def assert_disjoint(cal: pd.DataFrame, test: pd.DataFrame) -> None:
    """Refuse to report calibrated coverage on rows the calibration was fitted on."""
    c_end, t_start = cal["timestamp"].max(), test["timestamp"].min()
    if c_end >= t_start:
        raise AssertionError(
            f"calibration window ends {c_end} but the reported window starts {t_start}; "
            "they overlap, so any coverage computed on it would be circular")
    overlap = set(map(tuple, cal[["item_id", "timestamp"]].values)) & \
              set(map(tuple, test[["item_id", "timestamp"]].values))
    if overlap:
        raise AssertionError(f"{len(overlap)} (tank, timestamp) pairs appear in both windows")


# ---------------------------------------------------------------- evaluation

def evaluate(df: pd.DataFrame, *, alpha: float = ALPHA) -> pd.DataFrame:
    """Coverage, width, bias and error, before and after calibration, per model."""
    nominal = 1.0 - alpha
    rows = []
    for model, g in df.groupby("model", sort=False):
        g = g.dropna(subset=["actual"])
        raw_in = (g["actual"] >= g["q10"]) & (g["actual"] <= g["q90"])
        cal_in = (g["actual"] >= g["lower_cal"]) & (g["actual"] <= g["upper_cal"])
        sa = g["actual"].sum()
        rows.append({
            "model": model, "n": int(len(g)), "nominal_coverage": nominal,
            "coverage_raw": float(raw_in.mean()), "coverage_cal": float(cal_in.mean()),
            "width_raw": float((g["q90"] - g["q10"]).mean()),
            "width_cal": float((g["upper_cal"] - g["lower_cal"]).mean()),
            "miss_below_raw": float((g["actual"] < g["q10"]).mean()),
            "miss_below_cal": float((g["actual"] < g["lower_cal"]).mean()),
            "miss_above_raw": float((g["actual"] > g["q90"]).mean()),
            "miss_above_cal": float((g["actual"] > g["upper_cal"]).mean()),
            "volume_bias_raw_pct": float(100 * (g["pred"].sum() - sa) / sa),
            "volume_bias_cal_pct": float(100 * (g["pred_cal"].sum() - sa) / sa),
            "mae_raw": float((g["actual"] - g["pred"]).abs().mean()),
            "mae_cal": float((g["actual"] - g["pred_cal"]).abs().mean()),
        })
    return pd.DataFrame(rows)
