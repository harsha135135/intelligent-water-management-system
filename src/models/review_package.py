"""Build the final capstone review package from the COMPLETED benchmark.

This module is strictly a *reader*. It re-scores nothing, refits nothing, and changes no
methodology: it consumes ``results/chronos2/predictions_*.parquet`` exactly as produced by the
completed rolling-origin run and derives every table, CSV and figure from those rows.

Scope decisions, stated once:

* **PatchTST is excluded.** It was never run on this grid; ``results/patchtst/`` comes from a
  different, non-comparable evaluation (24 series / 576 rows, different holdout dates).
* The headline comparison is **Chronos-2 zero-shot vs NPTS**, with **SeasonalNaive-24** as the
  reference floor, because NPTS is the model currently shipping in ``results/autogluon`` (its
  saved WeightedEnsemble carries weight 1.0 on NPTS).
* The **final holdout** is the last origin of the completed grid (2026-04-15 23:00), which
  forecasts the final 7 contiguous days of the dataset. It is a designated subset of the
  benchmark, not a separate experiment, and is labelled as such everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.curate import load_curated_hourly
from .metrics import attach_scales

RESULTS = Path("results/chronos2")
OUT = RESULTS / "review"
PLOTS = OUT / "plots"

HORIZONS = [6, 12, 24, 48, 72, 168]
HLABEL = {6: "6h", 12: "12h", 24: "24h", 48: "48h", 72: "72h", 168: "168h"}

CHRONOS = "Chronos2-ZS"
INCUMBENT = "NPTS"
REFERENCE = "SeasonalNaive"
HEADLINE = [CHRONOS, INCUMBENT, REFERENCE]
VARIANTS = ["Chronos2-ZS", "Chronos2-COV", "Chronos2-COV-LEAN", "Chronos2-COV-XL"]

# Excluded from the review package by scope, not by outcome.
EXCLUDED_MODELS = ["ETS", "Theta", "DynamicOptimizedTheta"]


# ── loading ──────────────────────────────────────────────────────────────────

def load_scored() -> pd.DataFrame:
    """All completed predictions, with the seasonal-naive scales attached."""
    frames = [pd.read_parquet(p) for p in sorted(RESULTS.glob("predictions_*.parquet"))]
    preds = pd.concat(frames, ignore_index=True)
    panel = load_curated_hourly(with_features=False)
    scored = attach_scales(preds, panel)
    scored = scored.dropna(subset=["actual", "pred"]).copy()
    scored["error"] = scored["actual"] - scored["pred"]
    scored["abs_error"] = scored["error"].abs()
    return scored


def _metrics(g: pd.DataFrame) -> dict:
    err, ae = g["error"], g["abs_error"]
    ok = g["scale_mae"] > 0
    ok2 = g["scale_mse"] > 0
    return {
        "n": int(len(g)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mase": float((ae[ok] / g.loc[ok, "scale_mae"]).mean()) if ok.any() else np.nan,
        "rmsse": float(np.sqrt(((err[ok2] ** 2) / g.loc[ok2, "scale_mse"]).mean())) if ok2.any() else np.nan,
        "mean_actual": float(g["actual"].mean()),
        "bias": float(err.mean()),
        "worst_abs_error": float(ae.max()),
    }


def per_tank_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon, item), g in scored.groupby(["model", "horizon", "item_id"], sort=False):
        rows.append({"model": model, "horizon": int(horizon), "item_id": item, **_metrics(g)})
    return pd.DataFrame(rows)


def macro(per_tank: pd.DataFrame) -> pd.DataFrame:
    """Macro (equal-weight across tanks) plus volume-weighted absolute errors."""
    rows = []
    for (model, horizon), g in per_tank.groupby(["model", "horizon"], sort=False):
        w = g["mean_actual"].to_numpy()
        ws = w.sum()
        rows.append({
            "model": model, "horizon": int(horizon),
            "n_tanks": int(len(g)), "rows": int(g["n"].sum()),
            "mae": g["mae"].mean(), "rmse": g["rmse"].mean(),
            "mase": g["mase"].mean(), "rmsse": g["rmsse"].mean(),
            "vw_mae": float((g["mae"] * w).sum() / ws) if ws else np.nan,
            "vw_rmse": float((g["rmse"] * w).sum() / ws) if ws else np.nan,
        })
    return pd.DataFrame(rows)


def coverage(scored: pd.DataFrame) -> pd.DataFrame:
    """Empirical p10-p90 coverage. Nominal is 0.80."""
    s = scored.dropna(subset=["0.1", "0.9"])
    inside = ((s["actual"] >= s["0.1"]) & (s["actual"] <= s["0.9"])).astype(int)
    return (s.assign(inside=inside)
              .groupby(["model", "horizon"], sort=False)["inside"].mean()
              .reset_index().rename(columns={"inside": "p10_p90_coverage"}))


# ── §1 headline comparison ───────────────────────────────────────────────────

def headline_table(mac: pd.DataFrame) -> pd.DataFrame:
    """Chronos-2 vs NPTS vs SeasonalNaive at every horizon, with both improvement forms."""
    w = mac[mac["model"].isin(HEADLINE)].pivot(index="horizon", columns="model")
    rows = []
    for h in HORIZONS:
        r = {"horizon": h, "horizon_label": HLABEL[h]}
        for m in ("mae", "rmse", "mase", "rmsse"):
            c = w[(m, CHRONOS)][h]
            n = w[(m, INCUMBENT)][h]
            s = w[(m, REFERENCE)][h]
            r[f"chronos2_{m}"] = c
            r[f"npts_{m}"] = n
            r[f"seasonalnaive_{m}"] = s
            r[f"abs_impr_vs_npts_{m}"] = n - c
            r[f"pct_impr_vs_npts_{m}"] = 100 * (n - c) / n if n else np.nan
            r[f"abs_impr_vs_naive_{m}"] = s - c
            r[f"pct_impr_vs_naive_{m}"] = 100 * (s - c) / s if s else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── §2 per-tank Chronos vs NPTS ──────────────────────────────────────────────

def per_tank_comparison(per_tank: pd.DataFrame, horizons=(6, 24, 168)) -> pd.DataFrame:
    c = per_tank[per_tank["model"] == CHRONOS]
    n = per_tank[per_tank["model"] == INCUMBENT]
    m = c.merge(n, on=["horizon", "item_id"], suffixes=("_chronos2", "_npts"))
    m = m[m["horizon"].isin(horizons)].copy()
    for k in ("mae", "rmse", "mase", "rmsse"):
        m[f"pct_impr_{k}"] = 100 * (m[f"{k}_npts"] - m[f"{k}_chronos2"]) / m[f"{k}_npts"]
    m["winner"] = np.where(m["mase_chronos2"] < m["mase_npts"], "Chronos-2", "NPTS")
    m["beats_seasonal_naive"] = m["mase_chronos2"] < 1.0
    cols = (["item_id", "horizon", "winner", "beats_seasonal_naive", "mean_actual_chronos2"]
            + [f"{k}_{s}" for k in ("mae", "rmse", "mase", "rmsse")
               for s in ("chronos2", "npts")]
            + [f"pct_impr_{k}" for k in ("mae", "rmse", "mase", "rmsse")])
    out = m[cols].rename(columns={"mean_actual_chronos2": "mean_actual_kl_h"})
    return out.sort_values(["horizon", "mase_chronos2"], ascending=[True, False])


# ── §5 practical accuracy ────────────────────────────────────────────────────

def practical_accuracy(scored: pd.DataFrame, model: str, horizon: int,
                       trust: dict) -> pd.DataFrame:
    """Operator-facing accuracy, including within-tolerance rates.

    Percentage-within-tolerance is only defined where the actual is non-zero: with actual = 0 a
    relative error is either 0 (exact) or undefined (any non-zero prediction). 26.5% of observed
    hourly readings are exactly zero, so those rows are **excluded from the tolerance columns
    only** and counted in ``n_zero_actual``. Every other column uses all rows.
    """
    s = scored[(scored["model"] == model) & (scored["horizon"] == horizon)]
    rows = []
    for item, g in s.groupby("item_id", sort=True):
        nz = g[g["actual"] > 0]
        rel = (nz["abs_error"] / nz["actual"]) if len(nz) else pd.Series(dtype=float)
        base = _metrics(g)
        rows.append({
            "item_id": item,
            "trust": trust.get(item, "unknown"),
            "mean_actual_kl_h": base["mean_actual"],
            "mae": base["mae"],
            "rmse": base["rmse"],
            "mae_pct_of_mean": (100 * base["mae"] / base["mean_actual"]
                                if base["mean_actual"] > 0 else np.nan),
            "mase": base["mase"],
            "rmsse": base["rmsse"],
            "bias_mean_error": base["bias"],
            "worst_abs_error": base["worst_abs_error"],
            "n_rows": base["n"],
            "n_zero_actual": int((g["actual"] == 0).sum()),
            "n_used_for_tolerance": int(len(nz)),
            "pct_within_10": float(100 * (rel <= 0.10).mean()) if len(nz) else np.nan,
            "pct_within_20": float(100 * (rel <= 0.20).mean()) if len(nz) else np.nan,
            "pct_within_30": float(100 * (rel <= 0.30).mean()) if len(nz) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_actual_kl_h", ascending=False)


# ── §6 operational volume accuracy ───────────────────────────────────────────

def daily_volume_accuracy(scored: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Per-tank accuracy of the *volume* over the ``horizon``-hour window after each origin.

    MASE answers "is this better than naive". It does not answer "can we size a refill with it",
    which is the question the water-management system actually asks. This aggregates each
    forecast window to a single total and scores that.

    ``err_pct_of_window`` is the mean absolute window error as a percentage of that tank's mean
    window demand. It is meaningless where a tank draws almost nothing - a tank that uses
    0.09 KL/day cannot be forecast to a percentage - so ``mean_actual_kl_per_window`` is kept
    alongside it and must be read first. ``bias_pct`` is signed: negative = under-forecast.
    """
    s = scored[scored["horizon"] == horizon]
    windows = s.groupby(["model", "item_id", "origin"])[["actual", "pred"]].sum().reset_index()
    windows["error"] = windows["pred"] - windows["actual"]
    rows = []
    for (model, item), g in windows.groupby(["model", "item_id"], sort=False):
        mean_actual = float(g["actual"].mean())
        total = float(g["actual"].sum())
        rows.append({
            "model": model, "item_id": item, "horizon": horizon,
            "n_windows": int(len(g)),
            "mean_actual_kl_per_window": mean_actual,
            "mae_kl_per_window": float(g["error"].abs().mean()),
            "err_pct_of_window": (100 * float(g["error"].abs().mean()) / mean_actual
                                  if mean_actual > 0 else np.nan),
            "bias_kl_per_window": float(g["error"].mean()),
            "bias_pct": 100 * float(g["error"].sum()) / total if total > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["model", "mean_actual_kl_per_window"], ascending=[True, False])


def volume_bias(scored: pd.DataFrame, healthy: list[str] | None = None) -> pd.DataFrame:
    """Signed volume error per (model, horizon): 100 * (sum pred - sum actual) / sum actual.

    A point forecast that minimises absolute error on a spiky, right-skewed series sits near the
    conditional median, below the conditional mean, so its totals under-shoot. That is invisible
    in MAE/MASE and decisive for refill sizing, so it is reported separately.
    """
    def _bias(d):
        a = d["actual"].sum()
        return 100 * (d["pred"].sum() - a) / a if a else np.nan

    rows = []
    for (model, horizon), g in scored.groupby(["model", "horizon"], sort=False):
        r = {"model": model, "horizon": int(horizon), "volume_bias_pct_all_tanks": _bias(g)}
        if healthy:
            r["volume_bias_pct_healthy_tanks"] = _bias(g[g["item_id"].isin(healthy)])
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["model", "horizon"])


# ── §4 final holdout ─────────────────────────────────────────────────────────

def final_holdout(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """The last origin of the completed grid: forecasts the final 7 contiguous days.

    No leakage: the AutoGluon baselines were fitted strictly on data before the FIRST origin
    (2026-03-24 22:00), three weeks earlier, and Chronos-2 is zero-shot with context truncated
    at the origin.
    """
    origin = scored["origin"].max()
    hold = scored[scored["origin"] == origin]
    keep = hold["model"].isin(HEADLINE)
    hold = hold[keep]
    pt = per_tank_metrics(hold)
    return hold, pt, origin


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    trust = json.loads(Path("eda/tank_trust.json").read_text())

    scored = load_scored()
    scored = scored[~scored["model"].isin(EXCLUDED_MODELS)]
    print(f"loaded {len(scored):,} scored rows | models: {sorted(scored['model'].unique())}")

    pt = per_tank_metrics(scored)
    mac = macro(pt).merge(coverage(scored), on=["model", "horizon"], how="left")
    pt.to_csv(OUT / "per_tank_metrics_all_models.csv", index=False)
    mac.to_csv(OUT / "macro_metrics_all_models.csv", index=False)

    head = headline_table(mac)
    head.to_csv(OUT / "headline_comparison.csv", index=False)

    cmp3 = per_tank_comparison(pt)
    cmp3.to_csv(OUT / "per_tank_chronos2_vs_npts.csv", index=False)

    prac = {}
    for h in (24, 168):
        p = practical_accuracy(scored, CHRONOS, h, trust)
        p.to_csv(OUT / f"per_tank_practical_accuracy_h{h}.csv", index=False)
        prac[h] = p

    healthy = [k for k, v in trust.items() if v == "healthy"]
    dv = daily_volume_accuracy(scored, horizon=24)
    dv.to_csv(OUT / "per_tank_daily_volume_accuracy.csv", index=False)
    vb = volume_bias(scored, healthy=healthy)
    vb.to_csv(OUT / "volume_bias.csv", index=False)

    hold, hold_pt, origin = final_holdout(scored)
    hold_mac = macro(hold_pt).merge(coverage(hold), on=["model", "horizon"], how="left")
    hold_pt.to_csv(OUT / "final_holdout_per_tank.csv", index=False)
    hold_mac.to_csv(OUT / "final_holdout_macro.csv", index=False)

    manifest = {
        "source": "completed rolling-origin benchmark; nothing re-run",
        "origins": int(scored["origin"].nunique()),
        "origin_stride_hours": 23,
        "horizons": HORIZONS,
        "tanks": int(scored["item_id"].nunique()),
        "models_included": sorted(scored["model"].unique().tolist()),
        "models_excluded_by_scope": EXCLUDED_MODELS + ["PatchTST (never run on this grid)"],
        "final_holdout_origin": str(origin),
        "final_holdout_window": [str(hold["timestamp"].min()), str(hold["timestamp"].max())],
        "rows_per_model_per_horizon": (
            scored.groupby(["horizon", "model"]).size().unstack("model").to_dict()),
    }
    (OUT / "review_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\n=== HEADLINE (macro) ===")
    show = ["horizon_label", "chronos2_mase", "npts_mase", "seasonalnaive_mase",
            "pct_impr_vs_npts_mase", "pct_impr_vs_naive_mase"]
    print(head[show].round(4).to_string(index=False))
    print(f"\n=== FINAL HOLDOUT origin={origin} "
          f"({hold['timestamp'].min()} -> {hold['timestamp'].max()}) ===")
    print(hold_mac[["model", "horizon", "n_tanks", "rows", "mae", "rmse", "mase", "rmsse"]]
          .sort_values(["horizon", "mase"]).round(4).to_string(index=False))
    print("\n=== VOLUME BIAS (100*(sum pred - sum actual)/sum actual) ===")
    print(vb.pivot(index="model", columns="horizon", values="volume_bias_pct_all_tanks")
            [HORIZONS].round(2).to_string())
    print(f"\nwrote {len(list(OUT.glob('*.csv')))} CSVs -> {OUT}")


if __name__ == "__main__":
    main()
