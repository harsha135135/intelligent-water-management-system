"""Chronos-2+ — a methodology stack built around the zero-shot model, and its ablation.

The benchmark established that Chronos-2 zero-shot wins. It also diagnosed, precisely, *how* it
loses: **~24 % of hourly readings are exactly zero**, and a continuous-density foundation model
cannot place probability mass at exactly zero. That single fact explains the interval
under-coverage and part of the point error, and it is the thing to build against.

This module is that construction. Four layers, each cheap, each **fitted on a validation grid that
ends before the first test origin**, and each measured separately so the ablation shows what every
one of them is worth:

    L1  point estimator     which functional of the predictive distribution to report.
                            Measured first: Chronos-2's published point forecast is **already the
                            median** — `pred` equals `q0.50` on 100 % of rows — so "switch to the
                            median for MAE" is not available, it is already taken. What is
                            available is the choice among lower quantile blends (better MAE) and
                            the quantile-integrated mean (better volume). Costs nothing: every
                            quantile is already in the output.

    L2  zero hurdle         where P(zero | tank, hour-of-week) from pre-origin history exceeds a
                            threshold, emit exactly 0. This is the layer that gives the model the
                            zero atom its density cannot represent.

    L3  volume calibration  a per-tank multiplicative factor, so campus totals stop being
                            systematically short.

    L4  context ensemble    average the forecasts the same model makes from different context
                            lengths. Free variance reduction from one model.

Nothing here retrains or fine-tunes anything, and nothing here sees a test row before it is
scored. The layers are applied in the order above because they interact: the hurdle removes
volume, so the volume factor must be fitted after it.

    python -m src.models.chronos2_plus            # ~15 min -> results/chronos2/plus/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestSpec, TARGET_COL, make_spec
from .metrics import attach_scales

logger = logging.getLogger(__name__)

RESULTS = Path("results/chronos2")
OUT = RESULTS / "plus"
CHRONOS = "Chronos2-ZS"
HORIZONS = [6, 12, 24, 48, 72, 168]
HLABEL = {6: "6 h", 12: "12 h", 24: "1 d", 48: "2 d", 72: "3 d", 168: "7 d"}


def hlabel(h: int) -> str:
    return HLABEL.get(int(h), f"{int(h)} h")
QUANTILES = ["0.1", "0.25", "0.5", "0.75", "0.9"]


# ────────────────────────────────────────────────────────────── grids

def validation_spec(panel: pd.DataFrame, test: BacktestSpec, *, n_origins: int = 24,
                    stride_hours: int = 23) -> BacktestSpec:
    """A grid with the same shape as the test grid, ending before it starts.

    The last validation origin is placed so its longest horizon closes at or before the first test
    origin. Every parameter this module fits is therefore estimated on forecasts of data the test
    scoring never touches.
    """
    last = test.origins[0] - pd.Timedelta(hours=test.max_horizon)
    origins = [last - pd.Timedelta(hours=stride_hours * k) for k in range(n_origins)][::-1]
    return BacktestSpec(origins=origins, horizons=list(test.horizons))


# ────────────────────────────────────────────────────────────── L1 point estimator

QLEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def _integrated_mean(df: pd.DataFrame) -> np.ndarray:
    """E[X] from the predictive quantile function, by trapezoid over the five levels.

    Worth having explicitly because the published point forecast is the *median*: on a
    right-skewed predictive distribution the mean sits higher, and the gap is one component of
    the volume shortfall. Measured here rather than assumed — it turns out to explain about a
    quarter of it, so the rest is genuine under-prediction.
    """
    Q = df[[str(q) for q in QLEVELS]].to_numpy()
    return np.trapezoid(Q, QLEVELS, axis=1) / (QLEVELS[-1] - QLEVELS[0])


POINT_ESTIMATORS = {
    # `pred` is Chronos-2's published point forecast, which is q0.50 exactly.
    "published (q0.50)": lambda df: df["pred"].to_numpy(),
    "trimmed q25-q75": lambda df: df[["0.25", "0.5", "0.75"]].mean(axis=1).to_numpy(),
    "lower q10-q50": lambda df: df[["0.1", "0.25", "0.5"]].mean(axis=1).to_numpy(),
    "q0.25": lambda df: df["0.25"].to_numpy(),
    "integrated mean": _integrated_mean,
}


# ────────────────────────────────────────────────────────────── L2 zero hurdle

def zero_probability(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """P(demand is exactly zero | tank, hour-of-week), from history strictly before ``origin``.

    Hour-of-week rather than hour-of-day: the campus is materially quieter at weekends, and the
    zero pattern follows occupancy. Indexed by (item_id, how) so it joins onto any forecast frame.
    """
    hist = panel[panel["timestamp"] < origin]
    hist = hist.dropna(subset=[TARGET_COL])
    how = hist["timestamp"].dt.dayofweek * 24 + hist["timestamp"].dt.hour
    return (hist.assign(how=how, is_zero=(hist[TARGET_COL] == 0).astype(float))
                .groupby(["item_id", "how"])["is_zero"].mean())


def attach_zero_prob(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for origin, g in preds.groupby("origin", sort=False):
        p = zero_probability(panel, pd.Timestamp(origin))
        g = g.copy()
        how = pd.to_datetime(g["timestamp"]).dt.dayofweek * 24 + pd.to_datetime(g["timestamp"]).dt.hour
        g["p_zero"] = p.reindex(pd.MultiIndex.from_arrays([g["item_id"], how])).to_numpy()
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res["p_zero"] = res["p_zero"].fillna(0.0)
    return res


def apply_hurdle(point: np.ndarray, p_zero: np.ndarray, tau: float) -> np.ndarray:
    return np.where(p_zero >= tau, 0.0, point)


# ────────────────────────────────────────────────────────────── L3 volume calibration

def volume_factors(df: pd.DataFrame, point_col: str, *, clip: float = 2.0,
                   min_mean: float = 1e-3) -> pd.Series:
    """Per-tank total-actual / total-predicted, clipped.

    The clip is not cosmetic: a tank whose sensor was dead through the calibration window produces
    a ratio in the tens, which is a broken sensor rather than a model error. Tanks below
    ``min_mean`` KL/h are left uncorrected for the same reason.
    """
    g = df.groupby("item_id").agg(a=("actual", "sum"), p=(point_col, "sum"),
                                  m=("actual", "mean"))
    f = (g["a"] / g["p"]).replace([np.inf, -np.inf], np.nan)
    f = f.where(g["m"] >= min_mean, 1.0).fillna(1.0).clip(1 / clip, clip)
    return f


# ────────────────────────────────────────────────────────────── scoring

def macro_mase(df: pd.DataFrame, point_col: str) -> float:
    d = df.dropna(subset=["actual", point_col])
    d = d[d["scale_mae"] > 0]
    per_tank = ((d["actual"] - d[point_col]).abs() / d["scale_mae"]).groupby(
        [d["horizon"], d["item_id"]]).mean()
    return float(per_tank.groupby("horizon").mean().mean())


def score_all(df: pd.DataFrame, point_col: str, label: str) -> list[dict]:
    rows = []
    for h, g in df.groupby("horizon", sort=False):
        d = g.dropna(subset=["actual", point_col])
        ok = d[d["scale_mae"] > 0]
        per_tank_mae = (d["actual"] - d[point_col]).abs().groupby(d["item_id"]).mean()
        per_tank_mase = ((ok["actual"] - ok[point_col]).abs() / ok["scale_mae"]).groupby(
            ok["item_id"]).mean()
        rows.append({
            "layer": label, "horizon": int(h), "horizon_label": hlabel(h),
            "macro_mae": float(per_tank_mae.mean()),
            "macro_mase": float(per_tank_mase.mean()),
            "volume_bias_pct": float(100 * (d[point_col].sum() - d["actual"].sum())
                                     / d["actual"].sum()),
            "zero_rows_pct": float(100 * (d["actual"] == 0).mean()),
            "rows": int(len(d)),
        })
    return rows


# ────────────────────────────────────────────────────────────── inference

def run_grid(panel: pd.DataFrame, spec: BacktestSpec, contexts: list[int], device: str,
             cache: Path) -> dict[int, pd.DataFrame]:
    """Chronos-2 zero-shot over ``spec`` at each context length, cached to parquet."""
    from .chronos2_forecasting import load_pipeline, run_variant

    out, pipe = {}, None
    for ctx in contexts:
        path = cache / f"ctx{ctx}.parquet"
        if path.exists():
            logger.info("cached  context %-5d %s", ctx, path.name)
            out[ctx] = pd.read_parquet(path)
            continue
        if pipe is None:
            pipe = load_pipeline(device)
        logger.info("running context %d over %d origins", ctx, len(spec.origins))
        df = run_variant(pipe, panel, spec, CHRONOS, context=ctx)
        cache.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        out[ctx] = df
    return out


def ensemble(frames: dict[int, pd.DataFrame], contexts: list[int]) -> pd.DataFrame:
    """Average the point forecast and every quantile across context lengths."""
    base = frames[contexts[0]].copy()
    keys = ["item_id", "origin", "horizon", "timestamp"]
    cols = ["pred"] + QUANTILES
    stack = np.stack([frames[c].sort_values(keys)[cols].to_numpy() for c in contexts])
    base = base.sort_values(keys).reset_index(drop=True)
    base[cols] = stack.mean(axis=0)
    return base


# ────────────────────────────────────────────────────────────── driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contexts", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99])
    ap.add_argument("--out-dir", default=None, help="Override the output directory (smoke runs).")
    ap.add_argument("--max-tanks", type=int, default=None)
    ap.add_argument("--n-origins", type=int, default=None)
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    args = ap.parse_args()

    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.curate import load_curated_hourly

    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_curated_hourly(with_features=False)
    feats = load_curated_hourly()
    if args.max_tanks:
        keep = sorted(feats["item_id"].unique())[: args.max_tanks]
        feats = feats[feats["item_id"].isin(keep)]
        panel = panel[panel["item_id"].isin(keep)]
    spec_kw = {}
    if args.horizons:
        spec_kw["horizons"] = args.horizons
    if args.n_origins:
        spec_kw["n_origins"] = args.n_origins
    test = make_spec(feats, **spec_kw)
    val = validation_spec(feats, test, n_origins=len(test.origins))
    logger.info("test grid: %s", test.describe())
    logger.info("val  grid: %s", val.describe())
    assert val.origins[-1] + pd.Timedelta(hours=val.max_horizon) <= test.origins[0], \
        "validation window overlaps the test grid"

    val_frames = run_grid(feats, val, args.contexts, args.device, OUT / "_val")
    test_frames = run_grid(feats, test, args.contexts, args.device, OUT / "_test")

    def prep(frames):
        single = frames[max(args.contexts)]
        ens = ensemble(frames, args.contexts)
        out = {}
        for name, df in (("single", single), ("ensemble", ens)):
            d = attach_scales(df, panel)
            d = attach_zero_prob(d, panel)
            out[name] = d
        return out

    V, T = prep(val_frames), prep(test_frames)

    # ── L1: choose the point estimator on validation ────────────────────────
    l1 = {name: macro_mase(V["single"].assign(p=fn(V["single"])), "p")
          for name, fn in POINT_ESTIMATORS.items()}
    best_point = min(l1, key=l1.get)
    l1_adopted = l1[best_point] < l1["published (q0.50)"] - 1e-9
    if not l1_adopted:
        best_point = "published (q0.50)"
    logger.info("L1 point estimator (val macro MASE): %s -> %s",
                {k: round(v, 4) for k, v in l1.items()},
                f"adopt {best_point}" if l1_adopted else "keep the mean")

    # ── L2: choose the hurdle threshold on validation ───────────────────────
    v = V["single"].copy()
    v["p1"] = POINT_ESTIMATORS[best_point](v)
    l2 = {tau: macro_mase(v.assign(p=apply_hurdle(v["p1"].to_numpy(), v["p_zero"].to_numpy(), tau)), "p")
          for tau in args.taus}
    base_l2 = macro_mase(v, "p1")
    best_tau = min(l2, key=l2.get)
    if l2[best_tau] >= base_l2 - 1e-9:
        best_tau = None
    logger.info("L2 hurdle (val macro MASE, no hurdle = %.4f): %s -> %s", base_l2,
                {k: round(x, 4) for k, x in l2.items()},
                f"adopt tau={best_tau}" if best_tau is not None else "reject")
    v["p2"] = (apply_hurdle(v["p1"].to_numpy(), v["p_zero"].to_numpy(), best_tau)
               if best_tau is not None else v["p1"].to_numpy())

    # ── L3: does the context ensemble help on validation? ───────────────────
    ve = V["ensemble"].copy()
    ve["p1"] = POINT_ESTIMATORS[best_point](ve)
    ve["p2"] = (apply_hurdle(ve["p1"].to_numpy(), ve["p_zero"].to_numpy(), best_tau)
                if best_tau is not None else ve["p1"].to_numpy())
    use_ensemble = macro_mase(ve, "p2") < macro_mase(v, "p2") - 1e-9
    logger.info("L3 context ensemble on val: %.4f vs single %.4f -> %s",
                macro_mase(ve, "p2"), macro_mase(v, "p2"),
                "adopt" if use_ensemble else "reject")

    # ── L4: volume calibration is a *different objective* ───────────────────
    # Scaling the point forecast up to match totals moves it away from the conditional median,
    # so it costs MASE while it buys volume accuracy. The two cannot both be optimised by one
    # number, so the stack forks here: an accuracy track that stops at L3, and an operational
    # track that adds the factors and is judged on volume bias and daily campus error.
    acc_val = ve if use_ensemble else v
    factors = volume_factors(acc_val, "p2")
    acc_val["p3"] = acc_val["p2"] * acc_val["item_id"].map(factors).to_numpy()
    logger.info("L4 volume factors: median %.3f, range %.3f-%.3f | val macro MASE %.4f -> %.4f, "
                "val volume bias %+.2f%% -> %+.2f%%",
                factors.median(), factors.min(), factors.max(),
                macro_mase(acc_val, "p2"), macro_mase(acc_val, "p3"),
                100 * (acc_val["p2"].sum() - acc_val["actual"].sum()) / acc_val["actual"].sum(),
                100 * (acc_val["p3"].sum() - acc_val["actual"].sum()) / acc_val["actual"].sum())

    # ── apply the fitted stack to the TEST grid and ablate ──────────────────
    rows = []
    t = T["single"].copy()
    rows += score_all(t.assign(p=t["pred"]), "p", "L0 · published (q0.50)")
    t["p1"] = POINT_ESTIMATORS[best_point](t)
    rows += score_all(t, "p1", f"L1 · {best_point}" if l1_adopted
                      else "L1 · point estimate (published kept)")
    t["p2"] = (apply_hurdle(t["p1"].to_numpy(), t["p_zero"].to_numpy(), best_tau)
               if best_tau is not None else t["p1"].to_numpy())
    rows += score_all(t, "p2", (f"L2 · + zero hurdle (tau={best_tau})" if best_tau is not None
                                else "L2 · zero hurdle (rejected on validation)"))

    te = T["ensemble"].copy()
    te["p1"] = POINT_ESTIMATORS[best_point](te)
    te["p2"] = (apply_hurdle(te["p1"].to_numpy(), te["p_zero"].to_numpy(), best_tau)
                if best_tau is not None else te["p1"].to_numpy())
    rows += score_all(te, "p2", f"L3 · + {len(args.contexts)}-context ensemble"
                      + ("" if use_ensemble else " (rejected on validation)"))

    acc = te if use_ensemble else t
    acc["p3"] = acc["p2"] * acc["item_id"].map(factors).fillna(1.0).to_numpy()
    rows += score_all(acc, "p3", "L4 · + volume calibration (operational track)")

    ablation = pd.DataFrame(rows)
    ablation.to_csv(OUT / "ablation.csv", index=False)

    # where the gain lands: per tank at 1 d, published baseline against the accuracy track
    d24 = acc[acc.horizon == 24]
    if len(d24):
        g = d24.groupby("item_id")
        per_tank = pd.DataFrame({
            "baseline_mae": (d24["actual"] - d24["pred"]).abs().groupby(d24["item_id"]).mean(),
            "plus_mae": (d24["actual"] - d24["p2"]).abs().groupby(d24["item_id"]).mean(),
            "vol_mae": (d24["actual"] - d24["p3"]).abs().groupby(d24["item_id"]).mean(),
            "zero_share": (d24["actual"] == 0).groupby(d24["item_id"]).mean(),
            "gated_share": (d24["p_zero"] >= (best_tau if best_tau is not None else 2)
                            ).groupby(d24["item_id"]).mean(),
            "mean_actual": g["actual"].mean(),
        })
        per_tank["improvement_pct"] = (
            100 * (per_tank.baseline_mae - per_tank.plus_mae) / per_tank.baseline_mae)
        per_tank.sort_values("improvement_pct", ascending=False).to_csv(OUT / "per_tank_h24.csv")

    keep = ["item_id", "timestamp", "step", "actual", "origin", "horizon"] + QUANTILES
    dest = RESULTS if not args.out_dir else OUT
    for col, name in (("p2", "Chronos2-Plus"), ("p3", "Chronos2-Plus-Vol")):
        out = acc[keep].copy()
        out["pred"] = acc[col].to_numpy()
        out["model"] = name
        out.to_parquet(dest / f"predictions_{name}.parquet", index=False)

    base = ablation[ablation.layer.str.startswith("L0")].set_index("horizon")["macro_mase"]
    acc_label = [r for r in ablation.layer.unique() if r.startswith("L3")][0] if use_ensemble \
        else [r for r in ablation.layer.unique() if r.startswith("L2")][0]
    best = ablation[ablation.layer == acc_label].set_index("horizon")
    vol = ablation[ablation.layer.str.startswith("L4")].set_index("horizon")
    summary = {
        "validation_grid": {"origins": len(val.origins), "first": str(val.origins[0]),
                            "last": str(val.origins[-1])},
        "test_grid": {"origins": len(test.origins), "first": str(test.origins[0]),
                      "last": str(test.origins[-1])},
        "contexts": args.contexts,
        "L1_point_estimator": {"chosen": best_point, "adopted": bool(l1_adopted),
                               "val_macro_mase": {k: round(x, 4) for k, x in l1.items()}},
        "L2_hurdle_tau": {"chosen": best_tau, "adopted": best_tau is not None,
                          "val_macro_mase": {str(k): round(x, 4) for k, x in l2.items()}},
        "L3_context_ensemble": {"adopted": bool(use_ensemble)},
        "L4_volume_calibration": {
            "objective": "volume bias, not MASE — reported as a separate operational track",
            "median_factor": round(float(factors.median()), 4),
            "min": round(float(factors.min()), 4), "max": round(float(factors.max()), 4),
            "n_corrected": int((factors != 1.0).sum()),
            "test_volume_bias_pct": {hlabel(h): round(float(vol.loc[h, "volume_bias_pct"]), 2)
                                     for h in sorted(vol.index)},
            "test_macro_mase": {hlabel(h): round(float(vol.loc[h, "macro_mase"]), 4)
                                for h in sorted(vol.index)}},
        "accuracy_track_layer": acc_label,
        "test_macro_mase_by_horizon": {
            hlabel(h): {"L0": round(float(base[h]), 4),
                        "final": round(float(best.loc[h, "macro_mase"]), 4),
                        "improvement_pct": round(100 * (base[h] - best.loc[h, "macro_mase"]) / base[h], 2)}
            for h in sorted(base.index)},
        "overall_improvement_pct": round(
            100 * (base.mean() - best["macro_mase"].mean()) / base.mean(), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    factors.rename("factor").to_frame().to_csv(OUT / "volume_factors.csv")

    print()
    print(ablation.pivot(index="layer", columns="horizon_label", values="macro_mase")
          .reindex(columns=[hlabel(h) for h in sorted(ablation.horizon.unique())]).round(4).to_string())
    print()
    print(json.dumps(summary["test_macro_mase_by_horizon"], indent=2))
    print(f"\noverall: {summary['overall_improvement_pct']:+.2f}% macro MASE vs the published point forecast")


if __name__ == "__main__":
    main()
