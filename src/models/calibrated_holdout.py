"""Calibrated 45-day holdout: fit conformal intervals and volume bias on an earlier window.

Two disjoint windows, both built from consecutive daily origins forecasting 24 h ahead:

    calibration   2026-01-08 .. 2026-03-08   (60 days)  -> fits the correction parameters
    reported      2026-03-09 .. 2026-04-22   (45 days)  -> where coverage and bias are measured

The split is the whole point. Coverage measured on the rows the calibration was fitted on is
circular; ``calibration.assert_disjoint`` fails the run if the windows touch.

Each window gets its own NPTS predictor, fitted strictly on data at or before that window's first
origin, so no window sees its own future. Chronos-2 is zero-shot and SeasonalNaive-24 is a
computation, so neither can leak.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as cal
from .holdout45_continuous import (
    CHRONOS, HORIZON, INCUMBENT, REFERENCE, build_origins, chronos2, npts, seasonal_naive,
)

logger = logging.getLogger(__name__)

OUT = Path("results/chronos2/calibrated")
CAL_DAYS, TEST_DAYS = 60, 45


def run_window(panel, origins, *, context, device, time_limit, model_dir, tag):
    logger.info("[%s] %d origins, %s .. +24h", tag, len(origins),
                origins[0] + pd.Timedelta(hours=1))
    parts = [seasonal_naive(panel, origins)]
    parts.append(npts(panel, origins, context=context, time_limit=time_limit,
                      model_dir=model_dir))
    parts.append(chronos2(panel, origins, context=context, device=device))
    df = pd.concat(parts, ignore_index=True)
    assert (df["timestamp"] > df["origin"]).all(), f"{tag}: leakage"
    return df


def daily_campus(preds: pd.DataFrame) -> pd.DataFrame:
    """Campus totals per day for the point forecast and the calibrated interval."""
    p = preds.copy()
    p["day"] = p["timestamp"].dt.normalize()
    n_expected = p.groupby(["model", "day"]).size().rename("n_expected")
    ok = p.dropna(subset=["actual", "pred_cal"])
    agg = ok.groupby(["model", "day"]).agg(
        actual_kl=("actual", "sum"),
        pred_raw_kl=("pred", "sum"),
        pred_kl=("pred_cal", "sum"),
        lower_kl=("lower_cal", "sum"),
        upper_kl=("upper_cal", "sum"),
        n_scored=("actual", "size"),
    ).join(n_expected)
    agg["coverage_pct"] = 100 * agg["n_scored"] / agg["n_expected"]
    agg["error_kl"] = agg["pred_kl"] - agg["actual_kl"]
    agg["abs_error_kl"] = agg["error_kl"].abs()
    agg["error_raw_kl"] = agg["pred_raw_kl"] - agg["actual_kl"]
    agg["inside_band"] = (agg["actual_kl"] >= agg["lower_kl"]) & (agg["actual_kl"] <= agg["upper_kl"])
    return agg.reset_index()


def fit_daily_band(cal_daily: pd.DataFrame, *, alpha: float = 0.20) -> pd.DataFrame:
    """A conformal band for the *daily campus total*, fitted at that aggregation level.

    Summing 24 hourly p10s and p90s does not give an 80 % interval for the daily total — hourly
    errors partially cancel, so the summed band is far too wide. The honest daily band is
    conformalised directly on daily residuals from the calibration window.
    """
    rows = []
    for model, g in cal_daily.groupby("model"):
        g = g[g["coverage_pct"] >= 95]
        r = (g["actual_kl"] - g["pred_kl"]).to_numpy()
        n = len(r)
        k_lo, k_hi = alpha / 2, 1 - alpha / 2
        rows.append({"model": model, "n_days": int(n),
                     "d_lo": float(np.quantile(r, k_lo)) if n else 0.0,
                     "d_hi": float(np.quantile(r, k_hi)) if n else 0.0})
    return pd.DataFrame(rows)


def apply_daily_band(daily: pd.DataFrame, band: pd.DataFrame) -> pd.DataFrame:
    out = daily.merge(band[["model", "d_lo", "d_hi"]], on="model", how="left")
    out["band_lo"] = np.clip(out["pred_kl"] + out["d_lo"], 0.0, None)
    out["band_hi"] = out["pred_kl"] + out["d_hi"]
    out["inside_daily_band"] = ((out["actual_kl"] >= out["band_lo"]) &
                                (out["actual_kl"] <= out["band_hi"]))
    return out


def daily_summary(daily: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in daily.groupby("model"):
        full = g[g["coverage_pct"] >= 95]
        h = preds[preds.model == model].dropna(subset=["actual"])
        a = full["actual_kl"].to_numpy(); p = full["pred_kl"].to_numpy()
        rows.append({
            "model": model, "days": int(len(g)), "days_full_coverage": int(len(full)),
            "mean_actual_kl_day": float(a.mean()), "mean_pred_kl_day": float(p.mean()),
            "daily_mae_kl": float(full["abs_error_kl"].mean()),
            "daily_mape_pct": float((100 * full["abs_error_kl"] / full["actual_kl"]).mean()),
            "daily_mae_raw_kl": float(full["error_raw_kl"].abs().mean()),
            "total_bias_pct": float(100 * (p.sum() - a.sum()) / a.sum()),
            "total_bias_raw_pct": float(100 * (full["pred_raw_kl"].sum() - a.sum()) / a.sum()),
            "days_under_forecast": int((full["error_kl"] < 0).sum()),
            "daily_band_coverage": float(full["inside_daily_band"].mean()),
            "daily_band_width_kl": float((full["band_hi"] - full["band_lo"]).mean()),
            "sd_ratio": float(p.std(ddof=1) / a.std(ddof=1)),
            "corr_with_actual": float(np.corrcoef(a, p)[0, 1]),
            "hourly_mae_kl_h": float((h["actual"] - h["pred_cal"]).abs().mean()),
        })
    order = {CHRONOS: 0, INCUMBENT: 1, REFERENCE: 2}
    return pd.DataFrame(rows).sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cal-days", type=int, default=CAL_DAYS)
    ap.add_argument("--test-days", type=int, default=TEST_DAYS)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--time-limit", type=int, default=600)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.curate import load_curated_hourly

    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_curated_hourly()

    test_origins = build_origins(panel, args.test_days)
    cal_end = test_origins[0]                      # 23:00 the night the test window opens
    cal_origins = build_origins(panel, args.cal_days, end=cal_end)

    logger.info("calibration: %s .. %s (%d days)",
                cal_origins[0] + pd.Timedelta(hours=1), cal_end, len(cal_origins))
    logger.info("reported:    %s .. %s (%d days)",
                test_origins[0] + pd.Timedelta(hours=1), panel["timestamp"].max(),
                len(test_origins))

    cal_df = run_window(panel, cal_origins, context=args.context, device=args.device,
                        time_limit=args.time_limit, model_dir=OUT / "_npts_cal", tag="calibration")
    test_df = run_window(panel, test_origins, context=args.context, device=args.device,
                         time_limit=args.time_limit, model_dir=OUT / "_npts_test", tag="reported")

    cal.assert_disjoint(cal_df, test_df)
    logger.info("windows are disjoint — calibrated coverage is measured out of sample")

    params = cal.fit(cal_df)
    params["bias"].to_csv(OUT / "bias_factors.csv", index=False)
    params["conformal"].to_csv(OUT / "conformal_offsets.csv", index=False)

    test_cal = cal.apply(test_df, params)
    ev = cal.evaluate(test_cal)
    ev.to_csv(OUT / "calibration_effect.csv", index=False)

    # in-sample check: the calibration window itself should land on nominal by construction
    ev_cal = cal.evaluate(cal.apply(cal_df, params))
    ev_cal.to_csv(OUT / "calibration_effect_insample.csv", index=False)

    test_cal.to_parquet(OUT / "predictions_calibrated.parquet", index=False)

    cal_daily = daily_campus(cal.apply(cal_df, params))
    band = fit_daily_band(cal_daily); band.to_csv(OUT / "daily_band.csv", index=False)
    daily = apply_daily_band(daily_campus(test_cal), band)
    daily.to_csv(OUT / "daily_campus.csv", index=False)

    cal_tank = daily_per_tank(cal.apply(cal_df, params))
    tank_band = fit_tank_band(cal_tank); tank_band.to_csv(OUT / "daily_band_per_tank.csv", index=False)
    daily_per_tank(test_cal).to_csv(OUT / "daily_per_tank.csv", index=False)
    summ = daily_summary(daily, test_cal); summ.to_csv(OUT / "summary.csv", index=False)

    (OUT / "manifest.json").write_text(json.dumps({
        "calibration_window": [str(cal_origins[0] + pd.Timedelta(hours=1)), str(cal_end)],
        "reported_window": [str(test_origins[0] + pd.Timedelta(hours=1)),
                            str(panel["timestamp"].max())],
        "cal_days": len(cal_origins), "test_days": len(test_origins),
        "horizon_h": HORIZON, "alpha": params["alpha"], "nominal_coverage": 1 - params["alpha"],
        "method": "CQR (Romano et al. 2019) with independent tail offsets, per (model, tank); "
                  "multiplicative per-tank volume bias correction applied first",
        "windows_disjoint": True,
        "models": sorted(test_df["model"].unique().tolist()),
    }, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== hourly interval calibration, measured OUT OF SAMPLE ===")
    print(ev[["model", "coverage_raw", "coverage_cal", "width_raw", "width_cal",
              "miss_below_raw", "miss_below_cal", "miss_above_raw", "miss_above_cal"]]
          .to_string(index=False))
    print("\n=== volume bias and point error ===")
    print(ev[["model", "volume_bias_raw_pct", "volume_bias_cal_pct", "mae_raw", "mae_cal"]]
          .to_string(index=False))
    print("\n=== daily campus band (fitted on calibration days, measured on reported days) ===")
    print(band.merge(summ[["model", "daily_band_coverage", "daily_band_width_kl"]], on="model")
          .to_string(index=False))
    print("\n=== campus daily, calibrated ===")
    print(summ.to_string(index=False))
    print(f"\n-> {OUT}")



# ---------------------------------------------------------------- figures

def plot_all() -> None:
    """Actual vs calibrated forecast over the reported window, one figure per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from .phase3_analysis import _caption
    from .review_plots import COLOR, GRID, INK, INK2, SURFACE

    plots = OUT / "plots"; plots.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(OUT / "daily_campus.csv", parse_dates=["day"])
    summ = pd.read_csv(OUT / "summary.csv").set_index("model")
    eff = pd.read_csv(OUT / "calibration_effect.csv").set_index("model")

    col = {CHRONOS: COLOR["Chronos2-ZS"], INCUMBENT: COLOR["NPTS"],
           REFERENCE: COLOR["SeasonalNaive"]}
    label = {CHRONOS: "Chronos-2 (zero-shot)", INCUMBENT: "NPTS (incumbent)",
             REFERENCE: "SeasonalNaive-24 (reference)"}
    fname = {CHRONOS: "X_holdout45_chronos2_calibrated",
             INCUMBENT: "Y_holdout45_npts_calibrated",
             REFERENCE: "Z_holdout45_seasonal_naive_calibrated"}

    ymax = max(daily["actual_kl"].max(), daily["band_hi"].max()) * 1.16
    rmax = daily["error_kl"].abs().max() * 1.18
    low = daily[daily["coverage_pct"] < 95]["day"].unique()

    for model in (CHRONOS, INCUMBENT, REFERENCE):
        g = daily[daily.model == model].sort_values("day").reset_index(drop=True)
        s, e = summ.loc[model], eff.loc[model]
        c = col[model]

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(11.5, 5.9), sharex=True,
            gridspec_kw={"height_ratios": [2.7, 1], "hspace": 0.12})

        for d in low:
            for a in (ax, axr):
                a.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                          color="#f0efec", zorder=0)

        ax.fill_between(g["day"], g["band_lo"], g["band_hi"], color=c, alpha=0.17, zorder=1,
                        label=f"conformal band ({100*s.daily_band_coverage:.0f}% measured)")
        ax.plot(g["day"], g["actual_kl"], color=INK, lw=2.3, zorder=4, label="Actual")
        ax.plot(g["day"], g["pred_raw_kl"], color=c, lw=1.3, ls=":", alpha=0.75, zorder=2,
                label="Predicted — uncalibrated")
        ax.plot(g["day"], g["pred_kl"], color=c, lw=2.1, ls="--", zorder=3,
                label="Predicted — calibrated")
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, 100, 200, 300])
        ax.set_ylabel("Campus demand per day (KL)")
        ax.legend(frameon=False, fontsize=8.3, loc="lower left", ncol=2)
        ax.set_title(f"{label[model]} — calibrated, 45 consecutive days",
                     fontsize=11.5, color=INK, loc="left")

        box = (f"                raw     calibrated\n"
               f"daily MAE   {s.daily_mae_raw_kl:6.1f}      {s.daily_mae_kl:6.1f} KL\n"
               f"volume bias {s.total_bias_raw_pct:+6.1f}%     {s.total_bias_pct:+6.1f}%\n"
               f"hourly cov. {e.coverage_raw:6.3f}      {e.coverage_cal:6.3f}\n"
               f"band width  {e.width_raw:6.3f}      {e.width_cal:6.3f} KL/h")
        ax.text(0.988, 0.965, box, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.2, family="monospace", color=INK2,
                bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=GRID, lw=1))

        axr.bar(g["day"], g["error_raw_kl"], color=c, alpha=0.28, width=0.78, zorder=1,
                label="uncalibrated")
        axr.bar(g["day"], g["error_kl"], color=c, alpha=0.92, width=0.5, zorder=2,
                label="calibrated")
        axr.axhline(0, color=INK2, lw=1.1, zorder=3)
        axr.set_ylim(-rmax, rmax)
        axr.set_ylabel("Error (KL)\npred − actual")
        axr.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
        axr.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        axr.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axr.set_xlim(g["day"].min() - pd.Timedelta(hours=14),
                     g["day"].max() + pd.Timedelta(hours=14))
        for lab in axr.get_xticklabels():
            lab.set_fontsize(8.5)

        _caption(fig, f"Corrections are fitted on 8 Jan – 8 Mar 2026 and reported here on "
                      f"9 Mar – 22 Apr, a strictly later and disjoint window, so nothing on this "
                      f"chart is measured on data the calibration saw. Volume bias falls from "
                      f"{s.total_bias_raw_pct:+.1f}% to {s.total_bias_pct:+.1f}% and daily MAE from "
                      f"{s.daily_mae_raw_kl:.1f} to {s.daily_mae_kl:.1f} KL. The shaded band is a "
                      f"conformal 80% interval for the daily total, also fitted on the earlier "
                      f"window; it covers {100*s.daily_band_coverage:.0f}% of these 45 days, short "
                      f"of nominal — 56 calibration days cannot pin down a daily-total percentile "
                      f"across a regime change. The hourly intervals, fitted on ~1,400 residuals "
                      f"per tank, transfer far better ({e.coverage_raw:.3f} to {e.coverage_cal:.3f}).",
                 y=-0.015)
        for ext in ("png", "svg"):
            fig.savefig(plots / f"{fname[model]}.{ext}")
        plt.close(fig)
        print(f"  [plot] {fname[model]}")


# ---------------------------------------------------------------- per-tank view

REPRESENTATIVE = Path("results/chronos2/review/representative_tanks.json")


def daily_per_tank(preds: pd.DataFrame) -> pd.DataFrame:
    """Daily totals per (model, tank). Same pairing rule as the campus view: rows with a missing
    actual are dropped from both series so the pair stays comparable."""
    p = preds.copy()
    p["day"] = p["timestamp"].dt.normalize()
    n_expected = p.groupby(["model", "item_id", "day"]).size().rename("n_expected")
    ok = p.dropna(subset=["actual", "pred_cal"])
    agg = ok.groupby(["model", "item_id", "day"]).agg(
        actual_kl=("actual", "sum"), pred_raw_kl=("pred", "sum"), pred_kl=("pred_cal", "sum"),
        n_scored=("actual", "size"),
    ).join(n_expected)
    agg["coverage_pct"] = 100 * agg["n_scored"] / agg["n_expected"]
    agg["error_kl"] = agg["pred_kl"] - agg["actual_kl"]
    return agg.reset_index()


def fit_tank_band(cal_tank: pd.DataFrame, *, alpha: float = 0.20) -> pd.DataFrame:
    rows = []
    for (model, tank), g in cal_tank.groupby(["model", "item_id"]):
        g = g[g["coverage_pct"] >= 95]
        r = (g["actual_kl"] - g["pred_kl"]).to_numpy()
        rows.append({"model": model, "item_id": tank, "n_days": int(len(r)),
                     "d_lo": float(np.quantile(r, alpha / 2)) if len(r) else 0.0,
                     "d_hi": float(np.quantile(r, 1 - alpha / 2)) if len(r) else 0.0})
    return pd.DataFrame(rows)


def plot_per_tank() -> None:
    """One figure per model, four tanks each.

    The four tanks are the repository's existing representative set — highest, median and lowest
    demand among live tanks plus the highest-MASE tank, chosen by measured role in
    ``review_plots.pick_representatives`` and recorded in ``representative_tanks.json``. Reusing
    it means this panel cannot be cherry-picked and lines up with figures I and J.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from .phase3_analysis import _caption
    from .review_plots import COLOR, GRID, INK, INK2, SURFACE

    plots = OUT / "plots"; plots.mkdir(parents=True, exist_ok=True)
    reps = json.loads(REPRESENTATIVE.read_text())
    tanks = list(reps.values()); roles = {v: k for k, v in reps.items()}

    tank_daily = pd.read_csv(OUT / "daily_per_tank.csv", parse_dates=["day"])
    band = pd.read_csv(OUT / "daily_band_per_tank.csv")
    trust = json.loads(Path("eda/tank_trust.json").read_text())

    d = tank_daily.merge(band, on=["model", "item_id"], how="left")
    d["band_lo"] = np.clip(d["pred_kl"] + d["d_lo"], 0.0, None)
    d["band_hi"] = d["pred_kl"] + d["d_hi"]

    col = {CHRONOS: COLOR["Chronos2-ZS"], INCUMBENT: COLOR["NPTS"],
           REFERENCE: COLOR["SeasonalNaive"]}
    label = {CHRONOS: "Chronos-2 (zero-shot)", INCUMBENT: "NPTS (incumbent)",
             REFERENCE: "SeasonalNaive-24 (reference)"}
    fname = {CHRONOS: "X_tanks45_chronos2", INCUMBENT: "Y_tanks45_npts",
             REFERENCE: "Z_tanks45_seasonal_naive"}

    for model in (CHRONOS, INCUMBENT, REFERENCE):
        c = col[model]
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.8))
        fig.subplots_adjust(hspace=0.44, wspace=0.20, top=0.83, bottom=0.10)

        for ax, tank in zip(axes.ravel(), tanks):
            g = d[(d.model == model) & (d.item_id == tank)].sort_values("day").reset_index(drop=True)
            full = g[g["coverage_pct"] >= 95]
            a, pr, pc = full["actual_kl"], full["pred_raw_kl"], full["pred_kl"]
            mae = (pc - a).abs().mean()
            mae_raw = (pr - a).abs().mean()
            bias = 100 * (pc.sum() - a.sum()) / a.sum() if a.sum() else np.nan
            cov = float(((full["actual_kl"] >= full["band_lo"]) &
                         (full["actual_kl"] <= full["band_hi"])).mean())

            for dd in g[g["coverage_pct"] < 95]["day"]:
                ax.axvspan(dd - pd.Timedelta(hours=12), dd + pd.Timedelta(hours=12),
                           color="#f0efec", zorder=0)
            ax.fill_between(g["day"], g["band_lo"], g["band_hi"], color=c, alpha=0.16, zorder=1)
            ax.plot(g["day"], g["actual_kl"], color=INK, lw=1.9, zorder=4)
            ax.plot(g["day"], g["pred_raw_kl"], color=c, lw=1.0, ls=":", alpha=0.75, zorder=2)
            ax.plot(g["day"], g["pred_kl"], color=c, lw=1.7, ls="--", zorder=3)

            ax.set_title(f"{tank}", fontsize=9.5, color=INK, loc="left", pad=13)
            ax.text(0, 1.015, f"{roles[tank]} · {trust.get(tank, '?')} · "
                              f"{a.mean():.1f} KL/day", transform=ax.transAxes,
                    fontsize=7.8, color=INK2, va="bottom")
            ax.text(0.985, 0.955,
                    f"MAE {mae_raw:.2f}→{mae:.2f} KL\nbias {bias:+.1f}%\nband {100*cov:.0f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
                    family="monospace", color=INK2,
                    bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID, lw=0.8))
            ax.set_ylim(0, max(g["actual_kl"].max(), g["band_hi"].max()) * 1.42)
            ax.set_ylabel("KL/day", fontsize=8.5)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.tick_params(labelsize=8)

        handles = [plt.Line2D([], [], color=INK, lw=1.9, label="Actual"),
                   plt.Line2D([], [], color=c, lw=1.7, ls="--", label="Predicted — calibrated"),
                   plt.Line2D([], [], color=c, lw=1.0, ls=":", label="Predicted — uncalibrated"),
                   plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.16, label="conformal band")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.925),
                   frameon=False, fontsize=8.4, ncol=4)
        fig.suptitle(f"{label[model]} — four representative tanks, 45 consecutive days",
                     fontsize=11.5, color=INK, x=0.012, ha="left", y=0.97)

        _caption(fig, "Daily demand per tank, 9 Mar – 22 Apr 2026. Tanks are the repository's "
                      "existing representative set — highest, median and lowest demand among live "
                      "tanks plus the highest-MASE tank — chosen by measured role, not by hand, so "
                      "the panel cannot be cherry-picked. Each panel has its own y-scale: these "
                      "tanks span two orders of magnitude of demand. Corrections are fitted on the "
                      "earlier, disjoint 8 Jan – 8 Mar window. Shaded columns are days below 95% "
                      "sensor coverage.", y=-0.015)
        for ext in ("png", "svg"):
            fig.savefig(plots / f"{fname[model]}.{ext}")
        plt.close(fig)
        print(f"  [plot] {fname[model]}")

if __name__ == "__main__":
    main()
