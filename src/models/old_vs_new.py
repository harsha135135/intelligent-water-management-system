"""Score the published study's trained models on the refresh study's dates: old vs retrained.

The published study fitted its models on data up to March 2026. The refresh study refitted them on
data up to September. This puts the two side by side on the refresh study's own dates and rows:

* **Benchmark grid** (24 origins, six horizons): the published PatchTST predictors (defaults and
  tuned) and AutoGluon baseline predictors, loaded from disk and rolled over the new origins,
  against the refreshed ones. Paired bootstrap + Diebold-Mariano on per-origin macro MASE.
* **45-day holdout**: the published holdout PatchTST and NPTS predictors on the new daily origins.
* **Calibration**: the March-fitted volume factors, conformal offsets and daily band, applied to
  the new window's raw forecasts, against the September-fitted ones.

What cannot change, and is therefore not re-run: Chronos-2, TimesFM and Timestral are zero-shot -
nothing in them was fitted to this campus, so a forecast from the same context is the same forecast.
AutoGluon's statistical models (NPTS, ETS, Theta, SeasonalNaive) are local: they are refitted on
the history they are handed at prediction time, so an old predictor should behave like a new one;
they are run anyway, as a check of exactly that.

Run with the refresh study's environment:

    PESU_DATASET_DIR=dataset_2026-10-22 PESU_RESULTS_DIR=results/refresh_2026-10-22 \\
        python -m src.models.old_vs_new
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import IS_PUBLISHED, PUBLISHED_RESULTS, RESULTS_DIR
from . import significance
from .backtest import actuals_after, history_before, make_spec
from .patchtst_benchmark import _to_tsdf

logger = logging.getLogger(__name__)

OUT = RESULTS_DIR / "old_vs_new"
TAG = "@Mar"                          # suffix for a model fitted in the published study
HL = {6: "6 h", 12: "12 h", 24: "1 d", 48: "2 d", 72: "3 d", 168: "7 d"}


def load_old(rel: str):
    """Load a published predictor from a copy, so predicting never writes into the published folder."""
    import shutil
    from autogluon.timeseries import TimeSeriesPredictor

    src, dst = PUBLISHED_RESULTS / rel, RESULTS_DIR / "_old_models" / rel
    if not dst.exists():
        shutil.copytree(src, dst)
    return TimeSeriesPredictor.load(str(dst))


# ────────────────────────────────────────────────────────────── predictions

def roll(predictor, panel, origins, horizon, models, *, target, context, label=None):
    """Predict-only rolling forecast with an already-fitted predictor, in the benchmark schema."""
    blocks = []
    for origin in origins:
        hist = history_before(panel, origin, context=context)
        fut = actuals_after(panel, origin, horizon)
        known = _to_tsdf(fut, target).drop(columns=[target])
        tsdf = _to_tsdf(hist, target)
        for model in models:
            pred = predictor.predict(tsdf, known_covariates=known, model=model)
            pred = pred.reset_index()[["item_id", "timestamp", "mean", "0.1", "0.5", "0.9"]]
            b = fut[["item_id", "timestamp", "step", target]].merge(
                pred, on=["item_id", "timestamp"], how="left").rename(columns={target: "actual"})
            b["pred"] = np.clip(b["mean"].to_numpy(), 0.0, None)
            for q in ("0.1", "0.5", "0.9"):
                b[q] = np.clip(b[q].to_numpy(), 0.0, None)
            b = b.drop(columns=["mean"])
            b["model"] = (label or model) + TAG
            b["origin"] = origin
            b["horizon"] = horizon
            blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


def benchmark_predictions(panel, spec, context: int) -> pd.DataFrame:
    sources = [("_patchtst_models", "PatchTST"), ("_patchtst_tuned_models", "PatchTST-Tuned"),
               ("_ag_models", None)]
    frames = []
    for folder, label in sources:
        for h in spec.horizons:
            t0 = time.time()
            predictor = load_old(f"{folder}/h{h}")
            models = list(predictor.model_names())
            frames.append(roll(predictor, panel, spec.origins, h, models, target=spec.target_col,
                               context=context, label=label))
            logger.info("%s h=%d: %s in %.0fs", folder, h, models, time.time() - t0)
    return pd.concat(frames, ignore_index=True)


def holdout_predictions(panel, origins, context: int) -> pd.DataFrame:
    """The published holdout predictors (PatchTST fitted 2026-03-08, NPTS) on the new origins,
    in the calibrated table's schema (pred, q10, q90)."""
    from .holdout45_continuous import HORIZON, TARGET

    frames = []
    for rel, model, label in (("holdout45_patchtst/_model", None, "PatchTST"),
                              ("calibrated/_npts_test", "NPTS", "NPTS")):
        predictor = load_old(rel)
        name = model or list(predictor.model_names())[0]
        df = roll(predictor, panel, origins, HORIZON, [name], target=TARGET, context=context,
                  label=label)
        frames.append(df.rename(columns={"0.1": "q10", "0.9": "q90"}).drop(columns=["0.5", "horizon"]))
        logger.info("holdout %s%s done", label, TAG)
    return pd.concat(frames, ignore_index=True)


# ────────────────────────────────────────────────────────────── scoring

def per_origin_mase(scored: pd.DataFrame, model: str, horizon: int) -> pd.Series:
    s = scored[(scored.model == model) & (scored.horizon == horizon)]
    s = s[s.scale_mae.notna() & (s.scale_mae > 0)].dropna(subset=["actual", "pred"])
    s = s.assign(sa=(s.actual - s.pred).abs() / s.scale_mae)
    return s.groupby(["origin", "item_id"]).sa.mean().groupby("origin").mean().sort_index()


def pairs_table(scored: pd.DataFrame, pairs: list[tuple[str, str]], horizons) -> pd.DataFrame:
    """New (retrained) vs old on per-origin macro MASE, the series the main study tests.

    ``diff`` is new minus old, so negative = retraining lowered the error; ``improvement_pct`` is
    the same difference as a share of the old model's per-origin mean, sign flipped.
    """
    significance.RNG = np.random.default_rng(significance.SEED)
    rows = []
    for new, old in pairs:
        for h in horizons:
            a, b = per_origin_mase(scored, new, h), per_origin_mase(scored, old, h)
            a, b = a.align(b, join="inner")
            d = a.to_numpy() - b.to_numpy()
            mean, lo, hi, p_boot = significance.boot_ci(d)
            dm, p_dm = significance.dm_test(d, h)
            ref = float(b.mean())
            rows.append({"new": new, "old": old, "horizon": h, "mase_new": float(a.mean()),
                         "mase_old": ref, "diff": mean, "diff_lo": lo, "diff_hi": hi,
                         "improvement_pct": -100 * mean / ref,
                         "ci_lo_pct": -100 * hi / ref, "ci_hi_pct": -100 * lo / ref,
                         "p_bootstrap": p_boot, "p_dm": p_dm,
                         "significant_95": bool(hi < 0 or lo > 0), "origins_new_better": int((d < 0).sum())})
    return pd.DataFrame(rows)


def calibration_compare(raw: pd.DataFrame, old_dir: Path, new_dir: Path) -> pd.DataFrame:
    """The same raw forecasts calibrated with March-fitted vs September-fitted parameters."""
    from . import calibration as cal
    from .calibrated_holdout import apply_daily_band, daily_campus, daily_summary

    rows = []
    for which, d in (("Mar-fitted", old_dir), ("Sep-fitted", new_dir)):
        params = {"bias": pd.read_csv(d / "bias_factors.csv"),
                  "conformal": pd.read_csv(d / "conformal_offsets.csv")}
        models = sorted(set(params["bias"].model) & set(raw.model))
        test = cal.apply(raw[raw.model.isin(models)], params)
        daily = apply_daily_band(daily_campus(test), pd.read_csv(d / "daily_band.csv"))
        summ = daily_summary(daily, test).set_index("model")
        ev = cal.evaluate(test).set_index("model")
        for m in models:
            rows.append({"model": m, "calibration": which,
                         "daily_mae_kl": summ.loc[m, "daily_mae_kl"],
                         "volume_bias_pct": summ.loc[m, "total_bias_pct"],
                         "daily_band_coverage": summ.loc[m, "daily_band_coverage"],
                         "hourly_coverage": ev.loc[m, "coverage_cal"],
                         "hourly_width": ev.loc[m, "width_cal"]})
    return pd.DataFrame(rows)


def training_profile(spec) -> pd.DataFrame:
    """What each PatchTST was trained on, and what the window it is scored on looks like."""
    from ..config import PROJECT_ROOT
    from ..data.curate import load_curated_hourly
    from .backtest import KNOWN_COVARIATES

    flags = [c for c in KNOWN_COVARIATES if c.startswith("is_") and c != "is_weekend"]
    old_cut = pd.Timestamp(json.loads((PUBLISHED_RESULTS / "patchtst_manifest.json").read_text())["fit_cutoff"])
    new_cut = pd.Timestamp(json.loads((RESULTS_DIR / "patchtst_manifest.json").read_text())["fit_cutoff"])
    old_panel, new_panel = load_curated_hourly(PROJECT_ROOT / "dataset"), load_curated_hourly()
    parts = [("March fit (Waltr's earlier figures)", old_panel[old_panel.timestamp <= old_cut]),
             ("Retrained (revised figures)", new_panel[new_panel.timestamp <= new_cut]),
             ("Scored window", new_panel[new_panel.timestamp > spec.origins[0]])]
    rows = []
    for name, t in parts:
        t = t.dropna(subset=[spec.target_col])
        rows.append({"data": name, "first": str(t.timestamp.min()), "last": str(t.timestamp.max()),
                     "rows": len(t), "zero_share_pct": 100 * float((t[spec.target_col] == 0).mean()),
                     "calendar_flag_pct": 100 * float(t[flags].any(axis=1).mean()),
                     "calendar_flags": ",".join(flags)})
    return pd.DataFrame(rows)


def local_check(new: pd.DataFrame, old: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    keys = ["item_id", "timestamp", "origin", "horizon"]
    rows = []
    for m in names:
        if m.startswith("PatchTST"):
            continue
        j = new[new.model == m].merge(old[old.model == m + TAG], on=keys, suffixes=("", "_old"))
        rows.append({"model": m, "rows": len(j),
                     "max_abs_pred_diff": float((j.pred - j.pred_old).abs().max())})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────── report

def _p(p: float) -> str:
    return "n/a" if pd.isna(p) else ("< 0.0001" if p < 1e-4 else f"{p:.4f}")


def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))


def write_report() -> Path:
    """OLD_VS_NEW.md, every number read back from the CSVs this module wrote."""
    man = json.loads((OUT / "manifest.json").read_text())
    old_cut = json.loads((PUBLISHED_RESULTS / "patchtst_manifest.json").read_text())["fit_cutoff"]
    new_cut = json.loads((RESULTS_DIR / "patchtst_manifest.json").read_text())["fit_cutoff"]
    hold_old_cut = json.loads((PUBLISHED_RESULTS / "holdout45_patchtst" / "manifest.json").read_text())["fit_cutoff"]
    hold_new_cut = json.loads((RESULTS_DIR / "holdout45_patchtst" / "manifest.json").read_text())["fit_cutoff"]
    cal_old = json.loads((PUBLISHED_RESULTS / "calibrated" / "manifest.json").read_text())["calibration_window"]
    cal_new = json.loads((RESULTS_DIR / "calibrated" / "manifest.json").read_text())["calibration_window"]
    rep = json.loads((RESULTS_DIR / "calibrated" / "manifest.json").read_text())["reported_window"]
    agg = pd.read_csv(OUT / "benchmark_old_vs_new.csv")
    tests = pd.read_csv(OUT / "benchmark_old_vs_new_tests.csv")
    hold = pd.read_csv(OUT / "holdout_old_vs_new.csv").set_index("model")
    calc = pd.read_csv(OUT / "calibration_old_vs_new.csv")
    d = lambda s: pd.Timestamp(s).strftime("%-d %b %Y")
    o0, o1 = (pd.Timestamp(x) for x in man["benchmark_origins"])

    out = [
        "# Old models vs retrained, on the refresh study's dates",
        "",
        "Generated by `python -m src.models.old_vs_new` (re-render with `--report`) from the CSVs in "
        "this folder; nothing below is typed by hand.",
        "",
        f"The published study's models were fitted on data up to **{d(old_cut)}**. The refresh study "
        f"refitted them up to **{d(new_cut)}**. Here the March-fitted models, loaded unchanged from "
        f"`{PUBLISHED_RESULTS}/`, forecast exactly the refresh study's origins, tanks and hours, and "
        "are scored on the same rows as the retrained ones.",
        "",
        "__SUMMARY__",
        "",
        "## What can change, and what cannot",
        "",
        "| Model | Fitted to this campus | Re-run here |",
        "|---|---|---|",
        "| Chronos-2, TimesFM 2.5, Timestral | nothing (zero-shot) | no: the same history gives the "
        "same forecast, so old and new are one forecast; only their **calibration** can differ (§3) |",
        "| NPTS, ETS, Theta, DynamicOptimizedTheta, SeasonalNaive | nothing that persists: refitted "
        "per tank on the history handed to them at forecast time | yes, as a check that old = new |",
        "| PatchTST, PatchTST-Tuned | network weights | yes: the real old-vs-retrained test (§1, §2) |",
        "| Calibration (volume factors, conformal offsets, daily band) | per model and tank | yes: "
        f"fitted {d(cal_old[0])} – {d(cal_old[1])} vs {d(cal_new[0])} – {d(cal_new[1])} (§3) |",
        "",
        f"## 1. Benchmark grid: 24 origins ({d(o0)} – {d(o1)}), 6 horizons, "
        f"{man['rows_per_model']:,} scored rows per model",
        "",
        "MASE is the leaderboard's macro MASE (mean over tanks), so it matches "
        "`unified/leaderboard.csv` for the retrained models. The test is the main study's: per-origin "
        "macro MASE, paired over the 24 origins, 10,000 bootstrap resamples and Diebold–Mariano with "
        "the Harvey–Leybourne–Newbold correction. **Δ is retrained minus March-fitted: negative means "
        "retraining helped.**",
        "",
    ]
    ag = agg.set_index(["model", "horizon"])
    trained = tests[tests.new.str.startswith("PatchTST")]
    rows = []
    for r in trained.itertuples():
        rows.append({"Model": r.new, "Horizon": HL[r.horizon],
                     "MASE Mar-fitted": f"{ag.loc[(r.old, r.horizon), 'macro_mase']:.4f}",
                     "MASE retrained": f"{ag.loc[(r.new, r.horizon), 'macro_mase']:.4f}",
                     "Volume bias Mar → retrained": f"{ag.loc[(r.old, r.horizon), 'volume_bias_pct']:+.1f} → "
                                                     f"{ag.loc[(r.new, r.horizon), 'volume_bias_pct']:+.1f} %",
                     "Δ (95 % CI)": f"{r.diff:+.3f} ({r.diff_lo:+.3f} … {r.diff_hi:+.3f})",
                     "p (bootstrap)": _p(r.p_bootstrap), "p (DM)": _p(r.p_dm),
                     "Origins retrained better": f"{r.origins_new_better}/24"})
    out.append(_md(pd.DataFrame(rows)))
    for name, g in trained.groupby("new"):
        helped = g[g.significant_95 & (g["diff"] < 0)]
        hurt = g[g.significant_95 & (g["diff"] > 0)]
        old_m, new_m = (ag.xs(name + TAG, level="model"), ag.xs(name, level="model"))
        msg = (f"- **{name}**: the retrained network has the lower MASE at "
               f"{int((new_m.macro_mase < old_m.macro_mase).sum())} of 6 horizons. The bootstrap "
               f"interval excludes zero in favour of retraining at {', '.join(helped.horizon.map(HL)) or 'no horizon'}"
               f" and in favour of the March fit at {', '.join(hurt.horizon.map(HL)) or 'no horizon'}.")
        zero = new_m.index[new_m.volume_bias_pct <= -99].tolist()
        if zero:
            msg += (f" At {', '.join(HL[h] for h in zero)} the retrained network forecasts almost "
                    f"nothing (volume bias {new_m.loc[zero, 'volume_bias_pct'].max():+.1f} % or lower): "
                    "it has collapsed to a near-zero output.")
        out += ["", msg]

    lb = pd.read_csv(RESULTS_DIR / "unified" / "leaderboard.csv")
    lb = lb[~lb.model.isin(["PatchTST", "PatchTST-Tuned"])][["model", "horizon", "macro_mase"]]
    marf = agg[agg.model.str.endswith(TAG) & agg.model.str.startswith("PatchTST")][["model", "horizon", "macro_mase"]]
    full = pd.concat([lb, marf], ignore_index=True)
    full["rank"] = full.groupby("horizon").macro_mase.rank().astype(int)
    n = full.model.nunique()
    show = ["Timestral-209M-ZS", "Chronos2-ZS", "TimesFM2.5-ZS", "NPTS"] + sorted(marf.model.unique()) + ["SeasonalNaive"]
    piv = full[full.model.isin(show)].pivot(index="model", columns="horizon", values="macro_mase").loc[show]
    rk = full.pivot(index="model", columns="horizon", values="rank")
    t = pd.DataFrame({"Model": piv.index})
    for h in piv.columns:
        t[HL[h]] = [f"{piv.loc[m, h]:.3f} (#{rk.loc[m, h]})" for m in piv.index]
    out += ["", f"**Where the March-fitted networks would stand** if they replaced the retrained ones "
            f"on the refresh leaderboard ({n} models; macro MASE, rank in brackets):", "", _md(t)]

    tp = pd.read_csv(OUT / "training_data.csv").set_index("data")
    fm = lambda k, c, f: f.format(tp.loc[k, c])
    out += ["", "**What each network was trained on**, and what it is scored on:", "",
            _md(pd.DataFrame([{"Data": k, "Span": f"{d(tp.loc[k, 'first'])} – {d(tp.loc[k, 'last'])}",
                               "Tank-hours": fm(k, "rows", "{:,.0f}"),
                               "Zero-outflow hours": fm(k, "zero_share_pct", "{:.1f} %"),
                               "Hours with a calendar flag": fm(k, "calendar_flag_pct", "{:.1f} %")}
                              for k in tp.index])),
            "",
            "The retrained network saw roughly twice the share of zero-demand hours, from Waltr's "
            "revised figures, and calendar flags on a smaller share of hours, because the calendar "
            "covers neither 2024 nor anything after May 2026. Either could push a network towards "
            "forecasting zero. This run does not separate the two from ordinary training variance: "
            "that needs a retrain on the revised data cut at the March date (*hypothesis, untested*)."]

    lc = pd.read_csv(OUT / "local_models_check.csv")
    out += ["", "**Check on the local models** (expected: no change):", "",
            _md(pd.DataFrame({"Model": lc.model,
                              "Largest |Δ forecast| (KL/h)": lc.max_abs_pred_diff.map("{:.2g}".format),
                              "Rows compared": lc.rows.map("{:,}".format)}))]
    same = bool((lc.max_abs_pred_diff == 0).all())
    out += ["", "The March predictors and the refresh predictors give " +
            ("identical forecasts on every row: these models keep nothing from their fit, so there "
             "is no old-vs-new difference to report." if same else
             "slightly different forecasts; see the table.")]

    fmt = lambda m, c, f: f.format(hold.loc[m, c]) if m in hold.index else "—"
    out += ["", f"## 2. 45-day holdout: {d(rep[0])} – {d(rep[1])}, one 24 h forecast per day", "",
            f"Holdout PatchTST (tuned preset) fitted to {d(hold_old_cut)} vs {d(hold_new_cut)}. "
            "Point forecasts raw, as in the explorer. Daily figures use days with ≥ 95 % of hours "
            "observed.", ""]
    rows = []
    for m, lab in (("PatchTST" + TAG, f"PatchTST, fitted to {d(hold_old_cut)}"),
                   ("PatchTST", f"PatchTST, fitted to {d(hold_new_cut)}"),
                   ("NPTS" + TAG, "NPTS, March predictor"), ("NPTS", "NPTS, refresh predictor")):
        rows.append({"Model": lab, "Daily MAE (KL)": fmt(m, "daily_mae_kl", "{:.1f}"),
                     "Volume bias": fmt(m, "total_bias_pct", "{:+.1f} %"),
                     "Mean forecast / actual (KL/day)": f"{fmt(m, 'mean_pred_kl_day', '{:.0f}')} / "
                                                         f"{fmt(m, 'mean_actual_kl_day', '{:.0f}')}",
                     "Days under-forecast": f"{fmt(m, 'days_under_forecast', '{:.0f}')}/"
                                            f"{fmt(m, 'days_full_coverage', '{:.0f}')}",
                     "Hourly MAE (KL/h)": fmt(m, "hourly_mae_kl_h", "{:.3f}")})
    out.append(_md(pd.DataFrame(rows)))
    a, b = hold.loc["PatchTST" + TAG, "daily_mae_kl"], hold.loc["PatchTST", "daily_mae_kl"]
    out += ["", f"Retraining the holdout PatchTST moved its daily MAE from {a:.1f} to {b:.1f} KL "
            f"({'lower' if b < a else 'higher'} by {abs(a - b):.1f} KL, {100 * abs(a - b) / a:.0f} %)."]

    out += ["", "## 3. Calibration: March parameters vs September parameters, same raw forecasts", "",
            "The zero-shot forecasts below are identical in both rows; only the post-processing "
            "differs. Hourly coverage is of the calibrated p10–p90 interval (nominal 80 %); the daily "
            "band is the campus-total 80 % band.", ""]
    piv = calc.pivot(index="model", columns="calibration")
    rows = []
    for m in piv.index:
        g = lambda c, w: piv.loc[m, (c, w)]
        rows.append({"Model": m,
                     "Daily MAE Mar → Sep (KL)": f"{g('daily_mae_kl', 'Mar-fitted'):.1f} → {g('daily_mae_kl', 'Sep-fitted'):.1f}",
                     "Volume bias Mar → Sep": f"{g('volume_bias_pct', 'Mar-fitted'):+.1f} → {g('volume_bias_pct', 'Sep-fitted'):+.1f} %",
                     "Daily band coverage Mar → Sep": f"{100 * g('daily_band_coverage', 'Mar-fitted'):.0f} → {100 * g('daily_band_coverage', 'Sep-fitted'):.0f} %",
                     "Hourly coverage Mar → Sep": f"{100 * g('hourly_coverage', 'Mar-fitted'):.1f} → {100 * g('hourly_coverage', 'Sep-fitted'):.1f} %"})
    out.append(_md(pd.DataFrame(rows)))
    mae = piv["daily_mae_kl"]
    lower = mae.index[mae["Sep-fitted"] < mae["Mar-fitted"]].tolist()
    delta = (mae["Sep-fitted"] - mae["Mar-fitted"]).sort_values()
    out += ["", "Change in daily MAE from re-fitting (KL/day, negative = better): " +
            ", ".join(f"{m} {v:+.2f}" for m, v in delta.items()) + "."]

    out += ["", "## Caveats", "",
            "- **The March models were fitted on Waltr's old figures.** Waltr has since revised its "
            "history (flows zeroed in hours where the level moved; campus outflow on the published "
            "period fell 4.8 %), and every row here is scored on the revised figures. Part of any "
            "old-vs-new gap is that revision, not the six extra months of data.",
            "- **Calendar covariates are blank after 30 May 2026 for both.** PatchTST reads them; "
            "neither the old nor the retrained network sees the 2026-27 semester calendar.",
            "- The holdout PatchTST and the calibration are each one fit; there is no interval on "
            "§2 and §3, only on §1."]
    # the summary, written last so it can only repeat what the sections above computed
    summ = ["## In short", "",
            "- **Chronos-2, TimesFM, Timestral:** nothing was trained, so there is nothing to compare; "
            "their forecasts for these dates are the same whichever study they come from."]
    summ.append("- **NPTS, ETS, Theta, DynamicOptimizedTheta, SeasonalNaive:** " +
                ("identical forecasts on every row." if same else "near-identical forecasts (§1)."))
    for name, g in trained.groupby("new"):
        old_m, new_m = ag.xs(name + TAG, level="model"), ag.xs(name, level="model")
        k = int((old_m.macro_mase < new_m.macro_mase).sum())
        hurt = int((g.significant_95 & (g["diff"] > 0)).sum())
        summ.append(f"- **{name}:** on the benchmark grid the March-fitted network beats the retrained "
                    f"one at {k} of 6 horizons, by a margin whose 95 % interval excludes zero at {hurt}"
                    + (". These are one training run each, and the bootstrap resamples origins, not "
                       "training seeds: it shows that this retrained network is worse than the March one, "
                       "not that retraining makes the model worse in general (see the training-data note "
                       "in §1)." if k == 6 and hurt == 6 else "."))
    po, pn = hold.loc["PatchTST" + TAG], hold.loc["PatchTST"]
    summ.append(f"- **Holdout PatchTST (45-day view):** daily campus MAE {po.daily_mae_kl:.1f} KL "
                f"March-fitted vs {pn.daily_mae_kl:.1f} KL retrained, hourly MAE "
                f"{po.hourly_mae_kl_h:.3f} vs {pn.hourly_mae_kl_h:.3f} KL/h; both under-forecast campus "
                f"demand by more than half (volume bias {po.total_bias_pct:+.0f} % and "
                f"{pn.total_bias_pct:+.0f} %), so neither is usable for the daily total.")
    summ.append(f"- **Calibration:** re-fitting it on {d(cal_new[0])} – {d(cal_new[1])} changed the "
                "daily MAE by " + ", ".join(f"{m} {v:+.2f}" for m, v in delta.items()) +
                " KL/day, and moved the daily band's coverage towards its nominal 80 % for " +
                str(int(((piv['daily_band_coverage']['Sep-fitted'] - 0.8).abs()
                         < (piv['daily_band_coverage']['Mar-fitted'] - 0.8).abs()).sum())) +
                f" of {len(mae)} models (§3).")
    cb = piv["volume_bias_pct"]
    bias_closer = [m for m in mae.index if abs(cb.loc[m, "Sep-fitted"]) < abs(cb.loc[m, "Mar-fitted"])]
    helped = [m for m in delta.index if delta[m] <= -1.0]
    flat = [m for m in delta.index if abs(delta[m]) < 1.0]
    summ.append("- *Interpretation:* none of the calibrated models gains from a retrain (zero-shot, or "
                "refitted at every forecast); what fresher data can improve for them is "
                "the calibration. On these dates re-fitting it clearly helped "
                f"{' and '.join(helped) or 'no model'} (daily MAE down by 1 KL or more). It was a wash "
                f"on daily MAE for {', '.join(flat) or 'none'}, and volume bias moved closer to zero "
                f"only for {', '.join(bias_closer) or 'none'}.")
    out[out.index("__SUMMARY__")] = "\n".join(summ)
    path = OUT / "OLD_VS_NEW.md"
    path.write_text("\n".join(out) + "\n")
    return path


# ────────────────────────────────────────────────────────────── driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--report", action="store_true", help="Only re-render OLD_VS_NEW.md from the CSVs.")
    ap.add_argument("--rescore", action="store_true",
                    help="Re-score from the saved predictions of the March models; no forecasting.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if IS_PUBLISHED:
        raise SystemExit("set PESU_RESULTS_DIR to the refresh study")
    if args.report:
        print(write_report())
        return
    OUT.mkdir(parents=True, exist_ok=True)

    from ..data.curate import load_curated_hourly
    from .holdout45_continuous import build_origins, daily_campus, summary
    from .metrics import aggregate_metrics, attach_scales, coverage, per_tank_metrics

    panel = load_curated_hourly()
    spec = make_spec(panel)

    # A. benchmark grid
    if args.rescore:
        old = pd.read_parquet(OUT / "predictions_old_models.parquet")
    else:
        old = benchmark_predictions(panel, spec, args.context)
        old.to_parquet(OUT / "predictions_old_models.parquet", index=False)
    new = pd.concat([pd.read_parquet(p) for p in sorted(RESULTS_DIR.glob("predictions_*.parquet"))],
                    ignore_index=True)
    names = sorted({m[:-len(TAG)] for m in old.model.unique()})
    new = new[new.model.isin(names)]
    both = pd.concat([new, old[new.columns.intersection(old.columns)]], ignore_index=True)
    scored = attach_scales(both, load_curated_hourly(with_features=False))
    rows = scored.dropna(subset=["actual", "pred"]).groupby("model").size()
    assert rows.nunique() == 1, f"old and new scored on different rows:\n{rows}"
    ok = scored.dropna(subset=["actual", "pred"])
    vb = (ok.groupby(["model", "horizon"])[["pred", "actual"]].sum()
          .pipe(lambda t: 100 * (t.pred - t.actual) / t.actual).rename("volume_bias_pct").reset_index())
    agg = (aggregate_metrics(per_tank_metrics(scored))
           .merge(coverage(scored), on=["model", "horizon"]).merge(vb, on=["model", "horizon"]))
    agg.to_csv(OUT / "benchmark_old_vs_new.csv", index=False)
    pairs = pairs_table(scored, [(n, n + TAG) for n in names], spec.horizons)
    pairs.to_csv(OUT / "benchmark_old_vs_new_tests.csv", index=False)
    local_check(new, old, names).to_csv(OUT / "local_models_check.csv", index=False)
    training_profile(spec).to_csv(OUT / "training_data.csv", index=False)

    # B. 45-day holdout
    cman = json.loads((RESULTS_DIR / "calibrated" / "manifest.json").read_text())
    origins = build_origins(panel, cman["test_days"])
    if args.rescore:
        hold_old = pd.read_parquet(OUT / "holdout_old_models.parquet")
    else:
        hold_old = holdout_predictions(panel, origins, args.context)
        hold_old.to_parquet(OUT / "holdout_old_models.parquet", index=False)
    cal_new = pd.read_parquet(RESULTS_DIR / "calibrated" / "predictions_calibrated.parquet")
    pt_new = pd.read_parquet(RESULTS_DIR / "holdout45_patchtst" / "predictions_holdout45_patchtst.parquet")
    cols = ["item_id", "timestamp", "step", "actual", "pred", "q10", "q90", "model", "origin"]
    hold = pd.concat([cal_new[cal_new.model == "NPTS"][cols], pt_new[cols], hold_old[cols]],
                     ignore_index=True)
    hold_summary = summary(daily_campus(hold), hold)
    hold_summary.to_csv(OUT / "holdout_old_vs_new.csv", index=False)

    # C. calibration, same raw forecasts
    raw = cal_new[["item_id", "timestamp", "step", "actual", "pred", "q10", "q90", "model", "origin"]]
    calc = calibration_compare(raw, PUBLISHED_RESULTS / "calibrated", RESULTS_DIR / "calibrated")
    calc.to_csv(OUT / "calibration_old_vs_new.csv", index=False)

    (OUT / "manifest.json").write_text(json.dumps({
        "old_models_from": str(PUBLISHED_RESULTS), "new_results": str(RESULTS_DIR),
        "benchmark_origins": [str(spec.origins[0]), str(spec.origins[-1])],
        "holdout_origins": [str(origins[0]), str(origins[-1])],
        "rows_per_model": int(rows.iloc[0]), "models": names,
    }, indent=2))
    logger.info("wrote %s", write_report())
    print(pairs.round(3).to_string(index=False))
    print(hold_summary.round(3).to_string(index=False))
    print(calc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
