"""One analysis over **every** model on the grid, so the results can be read as a single study.

The Phase III material grew opponent by opponent: Chronos-2 vs NPTS, then the covariate variants,
then PatchTST. Each comparison got its own tables and its own figures, which meant the same fact
was stated three times in three shapes and no single artefact answered "how do all eleven models
compare". This module replaces that with one pass:

* every model, every horizon, every metric, on the identical 188,664 rows;
* Chronos-2 against **each** opponent by the same paired bootstrap + Diebold-Mariano;
* a full model x model win matrix, not a series of pairwise ones;
* per-tank, calibration, cost and operational views that all carry every model.

Nothing is refitted or re-forecast. It reads ``results/chronos2/predictions_*.parquet``.

    python -m src.models.unified_analysis      # ~4 min -> results/chronos2/unified/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import attach_scales
from .significance import boot_ci as _boot_ci, dm_test as _dm_test

RESULTS = Path("results/chronos2")
OUT = RESULTS / "unified"
PLOTS = OUT / "plots"

TARGET = "Outflow in KL"
HORIZONS = [6, 12, 24, 48, 72, 168]
HLABEL = {6: "6 h", 12: "12 h", 24: "1 d", 48: "2 d", 72: "3 d", 168: "7 d"}

CHRONOS = "Chronos2-ZS"
INCUMBENT = "NPTS"
REFERENCE = "SeasonalNaive"

# Display order: the production candidate first, then its variants, then every opponent in
# decreasing order of interest. Fixed here so no table or figure re-orders the models.
ORDER = [
    "Chronos2-ZS", "Chronos2-COV-XL", "Chronos2-COV", "Chronos2-COV-LEAN",
    "NPTS", "PatchTST-Tuned", "PatchTST", "ETS", "Theta", "DynamicOptimizedTheta",
    "SeasonalNaive",
]

LABEL = {
    "Chronos2-ZS": "Chronos-2 (zero-shot)",
    "Chronos2-COV": "Chronos-2 + covariates",
    "Chronos2-COV-LEAN": "Chronos-2 + covariates (lean)",
    "Chronos2-COV-XL": "Chronos-2 + covariates (XL)",
    "NPTS": "NPTS",
    "PatchTST": "PatchTST (defaults)",
    "PatchTST-Tuned": "PatchTST (tuned)",
    "ETS": "ETS",
    "Theta": "Theta",
    "DynamicOptimizedTheta": "DynamicOptimizedTheta",
    "SeasonalNaive": "SeasonalNaive-24",
}

SHORT = {
    "Chronos2-ZS": "Chronos-2 ZS", "Chronos2-COV": "C2 +cov",
    "Chronos2-COV-LEAN": "C2 +cov lean", "Chronos2-COV-XL": "C2 +cov XL",
    "NPTS": "NPTS", "PatchTST": "PatchTST", "PatchTST-Tuned": "PatchTST tuned",
    "ETS": "ETS", "Theta": "Theta", "DynamicOptimizedTheta": "DOTheta",
    "SeasonalNaive": "SeasNaive-24",
}

# Family, used for colour and for the "kind of model" column. Assigned by construction, never
# by outcome, so a model cannot change family by scoring differently.
FAMILY = {
    "Chronos2-ZS": "foundation", "Chronos2-COV": "foundation",
    "Chronos2-COV-LEAN": "foundation", "Chronos2-COV-XL": "foundation",
    "PatchTST": "trained deep", "PatchTST-Tuned": "trained deep",
    "NPTS": "statistical", "ETS": "statistical", "Theta": "statistical",
    "DynamicOptimizedTheta": "statistical", "SeasonalNaive": "reference",
}

ROLE = {
    "Chronos2-ZS": "production candidate",
    "NPTS": "deployed incumbent",
    "SeasonalNaive": "metric reference",
}


def _write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"  [table] {name:38s} {len(df):>6,} rows")


# ────────────────────────────────────────────────────────────── data

def load_long() -> pd.DataFrame:
    """Every model's scored rows in one long frame, with seasonal-naive scales attached.

    Long rather than wide: eleven models x six quantile columns would be 70+ columns, and every
    aggregate below is a groupby anyway. The paired tests re-pivot the two columns they need.
    """
    frames = [pd.read_parquet(p) for p in sorted(RESULTS.glob("predictions_*.parquet"))]
    preds = pd.concat(frames, ignore_index=True)

    from ..data.curate import load_curated_hourly
    panel = load_curated_hourly(with_features=False)
    scored = attach_scales(preds, panel)
    scored = scored.dropna(subset=["actual", "pred"]).copy()

    scored["err"] = scored["actual"] - scored["pred"]
    scored["abs_err"] = scored["err"].abs()
    scored["sq_err"] = scored["err"] ** 2
    # Scale each row by its own (tank, origin) denominator before averaging. Averaging the
    # scales first and dividing afterwards is NOT the same thing — the denominators differ across
    # origins — and it is how RMSSE silently drifted from metrics.py in an earlier draft.
    ok = scored["scale_mae"] > 0
    ok_sq = scored["scale_mse"] > 0
    scored["scaled_abs"] = np.where(ok, scored["abs_err"] / scored["scale_mae"], np.nan)
    scored["scaled_sq"] = np.where(ok_sq, scored["sq_err"] / scored["scale_mse"], np.nan)

    n = scored.groupby(["model", "horizon"]).size().unstack("model")
    if int(n.nunique(axis=1).max()) != 1:
        raise AssertionError(f"models scored on different row counts:\n{n}")
    print(f"  {len(scored):,} scored rows | {scored.model.nunique()} models | "
          f"{int(n.sum(axis=1).sum() / scored.model.nunique()):,} rows per model")
    return scored


def _present(scored: pd.DataFrame) -> list[str]:
    have = set(scored["model"])
    return [m for m in ORDER if m in have] + sorted(have - set(ORDER))


# ────────────────────────────────────────────────────────────── 1. leaderboard

def leaderboard(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, horizon): macro and volume-weighted point metrics, interval
    calibration, and signed volume bias — the whole benchmark in a single table."""
    per_tank = scored.groupby(["model", "horizon", "item_id"], sort=False).agg(
        n=("actual", "size"),
        mae=("abs_err", "mean"),
        mse=("sq_err", "mean"),
        mase=("scaled_abs", "mean"),
        mean_scaled_sq=("scaled_sq", "mean"),
        mean_actual=("actual", "mean"),
        mean_pred=("pred", "mean"),
    ).reset_index()
    per_tank["rmse"] = np.sqrt(per_tank["mse"])
    per_tank["rmsse"] = np.sqrt(per_tank["mean_scaled_sq"])

    rows = []
    for (model, horizon), g in per_tank.groupby(["model", "horizon"], sort=False):
        w = g["mean_actual"].to_numpy()
        s = scored[(scored.model == model) & (scored.horizon == horizon)]
        lo, hi = s["0.1"], s["0.9"]
        cov = s.dropna(subset=["0.1", "0.9"])
        rows.append({
            "model": model, "family": FAMILY.get(model, ""), "horizon": horizon,
            "horizon_label": HLABEL[horizon],
            "n_tanks": len(g), "rows": int(g["n"].sum()),
            "macro_mae": g["mae"].mean(), "macro_rmse": g["rmse"].mean(),
            "macro_mase": g["mase"].mean(), "macro_rmsse": g["rmsse"].mean(),
            "vw_mae": float((g["mae"] * w).sum() / w.sum()),
            "vw_rmse": float((g["rmse"] * w).sum() / w.sum()),
            "coverage": float(((cov["actual"] >= cov["0.1"]) & (cov["actual"] <= cov["0.9"])).mean()),
            "width": float((cov["0.9"] - cov["0.1"]).mean()),
            "neg_p10_pct": float((cov["0.1"] < 0).mean() * 100),
            "volume_bias_pct": float(100 * (s["pred"].sum() - s["actual"].sum()) / s["actual"].sum()),
        })
    lb = pd.DataFrame(rows)
    lb["rank_mase"] = lb.groupby("horizon")["macro_mase"].rank().astype(int)
    return lb.sort_values(["horizon", "macro_mase"]).reset_index(drop=True)


# ────────────────────────────────────────────────────────────── 2. significance

def _macro_per_origin(scored: pd.DataFrame, model: str, horizon: int, metric: str) -> np.ndarray:
    s = scored[(scored.model == model) & (scored.horizon == horizon)]
    col = "abs_err" if metric == "mae" else "scaled_abs"
    per_tank = s.groupby(["origin", "item_id"])[col].mean()
    return per_tank.groupby("origin").mean().sort_index().to_numpy()


def significance_vs_all(scored: pd.DataFrame, base: str = CHRONOS) -> pd.DataFrame:
    """``base`` against every other model, per horizon, on both metrics.

    Same machinery as the pairwise study it replaces: 10,000 paired bootstrap resamples over the
    24 origins, plus Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction.
    Negative ``diff`` means ``base`` is better.
    """
    rows = []
    for opponent in [m for m in _present(scored) if m != base]:
        for horizon in HORIZONS:
            for metric in ("mase", "mae"):
                b = _macro_per_origin(scored, base, horizon, metric)
                o = _macro_per_origin(scored, opponent, horizon, metric)
                d = b - o
                mean, lo, hi, p_boot = _boot_ci(d)
                dm, p_dm = _dm_test(d, horizon)
                ref = float(o.mean())
                rows.append({
                    "base": base, "opponent": opponent,
                    "opponent_family": FAMILY.get(opponent, ""),
                    "horizon": horizon, "horizon_label": HLABEL[horizon],
                    "metric": metric.upper(), "n_origins": len(d),
                    "base_score": float(b.mean()), "opponent_score": ref,
                    "diff": mean, "improvement_pct": -100 * mean / ref,
                    "ci_lo_pct": -100 * hi / ref, "ci_hi_pct": -100 * lo / ref,
                    "p_bootstrap": p_boot, "dm_stat": dm, "p_dm": p_dm,
                    "significant_95": bool(hi < 0 or lo > 0),
                    "base_better": bool(hi < 0),
                    "origins_won": int((d < 0).sum()),
                })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────── 3. head to head

def win_matrix(scored: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Every ordered model pair, counting tank-horizon cells won on per-tank MAE.

    Decided on MAE within a tank because that is the cell an operator reads; MASE would rescale
    by a denominator both models share, which changes nothing about who wins a cell.
    """
    models = _present(scored)
    hs = [horizon] if horizon else HORIZONS
    per = {(m, h): scored[(scored.model == m) & (scored.horizon == h)]
           .groupby("item_id")["abs_err"].mean() for m in models for h in hs}
    rows = []
    for a in models:
        for b in models:
            if a == b:
                continue
            wins = cells = 0
            for h in hs:
                pa, pb = per[(a, h)], per[(b, h)]
                wins += int((pa < pb).sum())
                cells += len(pa)
            rows.append({"model_a": a, "model_b": b, "horizon": horizon or "all",
                         "a_wins": wins, "cells": cells, "a_win_pct": 100 * wins / cells})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────── 4. per tank

def per_tank(scored: pd.DataFrame) -> pd.DataFrame:
    """model x tank x horizon, with the sensor trust tier joined on."""
    g = scored.groupby(["model", "horizon", "item_id"], sort=False).agg(
        n=("actual", "size"), mae=("abs_err", "mean"), mase=("scaled_abs", "mean"),
        mean_actual=("actual", "mean"),
    ).reset_index().rename(columns={"item_id": "tank"})
    tp = Path("eda/tank_trust.json")
    if tp.exists():
        g["trust"] = g["tank"].map(json.loads(tp.read_text()))
    g["horizon_label"] = g["horizon"].map(HLABEL)
    return g


def skill(scored: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Skill = 1 - MAE(model)/MAE(SeasonalNaive-24), per tank, for every model.

    Positive skill means the model beats the naive reference on that tank. Reported for every
    model so "is the average flattered by the dead tanks" can be answered for all of them at once.
    """
    pt = per_tank(scored)
    pt = pt[pt.horizon == horizon]
    ref = pt[pt.model == REFERENCE].set_index("tank")["mae"]
    out = pt[pt.model != REFERENCE].copy()
    out["ref_mae"] = out["tank"].map(ref)
    out["skill"] = 1 - out["mae"] / out["ref_mae"]
    return out[["model", "tank", "trust", "mean_actual", "mae", "ref_mae", "skill"]]


# ────────────────────────────────────────────────────────────── 5. diagnostics

def error_by_leadtime(scored: pd.DataFrame, horizon: int = 168) -> pd.DataFrame:
    s = scored[scored.horizon == horizon]
    return (s.groupby(["model", "step"])["abs_err"].mean()
            .reset_index().rename(columns={"abs_err": "mae"}))


def diurnal(scored: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    s = scored[scored.horizon == horizon].copy()
    s["hour"] = pd.to_datetime(s["timestamp"]).dt.hour
    g = s.groupby(["model", "hour"]).agg(
        mae=("abs_err", "mean"), bias=("err", "mean"), mean_actual=("actual", "mean")
    ).reset_index()
    return g


def zero_inflation(scored: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Why intervals miss, per model. The mechanism behind the calibration numbers."""
    s = scored[scored.horizon == horizon].dropna(subset=["0.1", "0.9"])
    rows = []
    for model, g in s.groupby("model"):
        below = g["actual"] < g["0.1"]
        above = g["actual"] > g["0.9"]
        zero = g["actual"] == 0
        rows.append({
            "model": model, "horizon": horizon, "n": len(g),
            "zero_fraction": float(zero.mean()),
            "miss_below_p10": float(below.mean()),
            "miss_above_p90": float(above.mean()),
            "coverage": float(1 - below.mean() - above.mean()),
            "p10_above_zero_pct": float((g["0.1"] > 0).mean() * 100),
            "p10_below_zero_pct": float((g["0.1"] < 0).mean() * 100),
            "below_misses_that_are_zero": float((below & zero).sum() / max(below.sum(), 1)),
            # "Clamped" means the lower bound is *replaced by* zero, not raised to zero: the
            # question is what the band would cover if it were allowed to say "possibly nothing
            # at all". Since outflow is non-negative, that is just P(actual <= p90).
            "coverage_if_p10_clamped": float((g["actual"] <= g["0.9"]).mean()),
        })
    return pd.DataFrame(rows)


def cost(lb: pd.DataFrame) -> pd.DataFrame:
    """Measured wall clock per model, read from the run manifests — never typed."""
    secs: dict[str, float] = {}
    regime: dict[str, str] = {}
    p = RESULTS / "run_manifest.json"
    if p.exists():
        for k, v in json.loads(p.read_text()).get("variants", {}).items():
            secs[k] = v.get("wall_clock_s")
            regime[k] = "zero-shot — no training"
    for name, f in (("PatchTST", "patchtst_manifest.json"),
                    ("PatchTST-Tuned", "patchtst_tuned_manifest.json")):
        q = RESULTS / f
        if q.exists():
            man = json.loads(q.read_text())
            hp = man.get("hyperparameters", {}).get("PatchTST", {})
            secs[name] = man.get("wall_clock_s")
            regime[name] = (f"trained — {hp.get('max_epochs')} epochs, "
                            f"context {hp.get('context_length')} h")
    q = RESULTS / "baselines_manifest.json"
    if q.exists():
        total = sum(json.loads(q.read_text()).get("seconds_per_horizon", {}).values())
        for m in ("NPTS", "SeasonalNaive", "ETS", "Theta", "DynamicOptimizedTheta"):
            secs[m] = total
            regime[m] = "fitted — five models in one AutoGluon pass"

    m24 = lb[lb.horizon == 24].set_index("model")
    rows = []
    for m in [x for x in ORDER if x in set(lb.model)]:
        rows.append({
            "model": m, "family": FAMILY.get(m, ""), "regime": regime.get(m, ""),
            "wall_clock_s": secs.get(m),
            "wall_clock_min": round(secs[m] / 60, 2) if secs.get(m) else None,
            "macro_mase_h24": float(m24.loc[m, "macro_mase"]),
            "coverage_h24": float(m24.loc[m, "coverage"]),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────── driver

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Unified analysis — every model on the shared grid\n")
    scored = load_long()
    models = _present(scored)

    lb = leaderboard(scored)
    _write(lb, "leaderboard.csv")

    sig = significance_vs_all(scored)
    _write(sig, "significance_vs_all.csv")

    wm = win_matrix(scored)
    _write(wm, "win_matrix_all_horizons.csv")
    wm24 = win_matrix(scored, 24)
    _write(wm24, "win_matrix_h24.csv")

    pt = per_tank(scored)
    _write(pt, "per_tank.csv")

    sk = skill(scored)
    _write(sk, "skill_h24.csv")

    lt = error_by_leadtime(scored)
    _write(lt, "error_by_leadtime.csv")

    dn = diurnal(scored)
    _write(dn, "diurnal_h24.csv")

    zi = zero_inflation(scored)
    _write(zi, "zero_inflation_h24.csv")

    ca = cost(lb)
    _write(ca, "cost.csv")

    s24 = sig[(sig.metric == "MASE") & (sig.horizon == 24)].set_index("opponent")
    best = lb[lb.horizon == 24].nsmallest(1, "macro_mase").iloc[0]
    summary = {
        "models": models, "n_models": len(models),
        "rows_per_model": int(lb[lb.model == CHRONOS]["rows"].sum()),
        "n_origins": int(scored["origin"].nunique()),
        "n_tanks": int(scored["item_id"].nunique()),
        "horizons": HORIZONS,
        "best_mase_h24": {"model": best.model, "macro_mase": float(best.macro_mase)},
        "chronos2_rank_by_horizon": {
            HLABEL[h]: int(lb[(lb.horizon == h) & (lb.model == CHRONOS)]["rank_mase"].iloc[0])
            for h in HORIZONS},
        "chronos2_improvement_pct_h24": {
            o: round(float(s24.loc[o, "improvement_pct"]), 2) for o in s24.index},
        "opponents_beaten_significantly_all_horizons": [
            o for o in s24.index
            if bool(sig[(sig.metric == "MASE") & (sig.opponent == o)]["base_better"].all())],
        "tank_horizon_cells_won_vs": {
            r.model_b: int(r.a_wins) for r in
            wm[wm.model_a == CHRONOS].itertuples() },
        "tank_horizon_cells_total": int(len(HORIZONS) * scored["item_id"].nunique()),
        "skill_positive_by_model": {
            m: int((sk[sk.model == m]["skill"] > 0).sum()) for m in sk.model.unique()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2)[:2600])


if __name__ == "__main__":
    main()
