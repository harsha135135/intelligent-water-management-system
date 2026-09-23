"""Continuous 45-day rolling 1-day-ahead holdout: actual vs predicted, per model.

Distinct from the published benchmark and written to its own directory. The benchmark uses 24
rolling origins at a 23-hour stride so that every hour-of-day is sampled once — excellent for
unbiased scoring, but the resulting origins overlap and cannot be drawn as one continuous
timeline. This module instead uses **45 consecutive daily origins**, each forecasting exactly the
following 24 hours, so the predictions tile the holdout window with no gaps and no overlap and
can be plotted against reality as a single continuous series.

Window: 2026-03-09 00:00 .. 2026-04-22 23:00 (45 days, the tail of the panel).
Origins: 2026-03-08 23:00 .. 2026-04-21 23:00, one per day, horizon 24 h.

Models: Chronos-2 zero-shot, NPTS (the incumbent), SeasonalNaive-24 (the reference).
NPTS is fitted **once**, strictly on data at or before the first origin, then rolled forward.
Nothing under ``results/chronos2/`` outside ``holdout45/`` is read or written.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import KNOWN_COVARIATES, actuals_after, history_before

logger = logging.getLogger(__name__)

OUT = Path("results/chronos2/holdout45")
TARGET = "Outflow in KL"
HOLDOUT_DAYS = 45
HORIZON = 24
CHRONOS, INCUMBENT, REFERENCE = "Chronos2-ZS", "NPTS", "SeasonalNaive-24"


def build_origins(panel: pd.DataFrame, days: int = HOLDOUT_DAYS,
                  end: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """One origin per day at 23:00, each followed by exactly 24 h of actuals.

    ``end`` is the last hour the window covers; pass it to build an earlier, disjoint window
    (the calibration set must not overlap the window it is reported on).
    """
    end = pd.Timestamp(end) if end is not None else panel["timestamp"].max()
    first_day = end.normalize() - pd.Timedelta(days=days - 1)
    first_origin = first_day - pd.Timedelta(hours=1)          # 23:00 the night before
    return [first_origin + pd.Timedelta(days=k) for k in range(days)]


def seasonal_naive(panel: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    """Repeat the last 24 observed hours forward. No fitting involved."""
    blocks = []
    for origin in origins:
        hist = history_before(panel, origin, context=24 * 4)
        last = {i: g.sort_values("timestamp")[TARGET].to_numpy()[-24:]
                for i, g in hist.groupby("item_id", sort=False)}
        fut = actuals_after(panel, origin, HORIZON)
        preds = []
        for item_id, g in fut.groupby("item_id", sort=False):
            w = last[item_id]
            if np.all(np.isnan(w)):
                w = np.zeros(24)
            w = np.where(np.isnan(w), np.nanmean(w), w)
            preds.append(np.tile(w, int(np.ceil(len(g) / 24)))[: len(g)])
        b = fut[["item_id", "timestamp", "step"]].copy()
        b["pred"] = np.concatenate(preds)
        # No native predictive distribution: p10 = p90 = the point forecast, so the conformal
        # step below builds the whole interval from residuals rather than widening one.
        b["q10"] = b["pred"]
        b["q90"] = b["pred"]
        b["actual"] = fut[TARGET].to_numpy()
        b["model"] = REFERENCE
        b["origin"] = origin
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


def npts(panel: pd.DataFrame, origins: list[pd.Timestamp], *, context: int,
         time_limit: int, model_dir: Path) -> pd.DataFrame:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    def tsdf(df):
        cols = ["item_id", "timestamp", TARGET] + [c for c in KNOWN_COVARIATES if c in df.columns]
        return TimeSeriesDataFrame.from_data_frame(
            df[cols], id_column="item_id", timestamp_column="timestamp")

    fit_cutoff = origins[0]
    train = panel[panel["timestamp"] <= fit_cutoff]
    logger.info("NPTS: fitting on %d rows up to %s", len(train), fit_cutoff)

    predictor = TimeSeriesPredictor(
        path=str(model_dir), target=TARGET, prediction_length=HORIZON, freq="h",
        eval_metric="MASE",
        known_covariates_names=[c for c in KNOWN_COVARIATES if c in panel.columns],
        verbosity=0,
    )
    predictor.fit(train_data=tsdf(train), hyperparameters={"NPTS": {}},
                  time_limit=time_limit, num_val_windows=1, enable_ensemble=False)

    blocks = []
    for k, origin in enumerate(origins, 1):
        hist = history_before(panel, origin, context=context)
        fut = actuals_after(panel, origin, HORIZON)
        known = tsdf(fut).drop(columns=[TARGET])
        pred = predictor.predict(tsdf(hist), known_covariates=known, model="NPTS")
        pred = pred.reset_index()[["item_id", "timestamp", "mean", "0.1", "0.9"]]
        b = fut[["item_id", "timestamp", "step", TARGET]].merge(
            pred, on=["item_id", "timestamp"], how="left").rename(columns={TARGET: "actual"})
        b["pred"] = np.clip(b["mean"].to_numpy(), 0.0, None)
        b["q10"] = np.clip(b["0.1"].to_numpy(), 0.0, None)
        b["q90"] = np.clip(b["0.9"].to_numpy(), 0.0, None)
        b = b.drop(columns=["mean", "0.1", "0.9"])
        b["model"] = INCUMBENT
        b["origin"] = origin
        blocks.append(b)
        if k % 15 == 0 or k == len(origins):
            logger.info("NPTS: %d/%d origins", k, len(origins))
    return pd.concat(blocks, ignore_index=True)


def chronos2(panel: pd.DataFrame, origins: list[pd.Timestamp], *, context: int,
             device: str) -> pd.DataFrame:
    from chronos import Chronos2Pipeline

    logger.info("Chronos-2: loading amazon/chronos-2 on %s", device)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    blocks, t0 = [], time.time()
    for k, origin in enumerate(origins, 1):
        hist = history_before(panel, origin, context=context)
        fut = actuals_after(panel, origin, HORIZON)
        out = pipe.predict_df(
            hist[["item_id", "timestamp", TARGET]], future_df=None,
            id_column="item_id", timestamp_column="timestamp", target=TARGET,
            prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9], batch_size=32,
        )
        out["pred"] = np.clip(out["predictions"].to_numpy(), 0.0, None)
        out["q10"] = np.clip(out["0.1"].to_numpy(), 0.0, None)
        out["q90"] = np.clip(out["0.9"].to_numpy(), 0.0, None)
        b = fut[["item_id", "timestamp", "step", TARGET]].merge(
            out[["item_id", "timestamp", "pred", "q10", "q90"]],
            on=["item_id", "timestamp"], how="left"
        ).rename(columns={TARGET: "actual"})
        b["model"] = CHRONOS
        b["origin"] = origin
        blocks.append(b)
        if k % 15 == 0 or k == len(origins):
            r = (time.time() - t0) / k
            logger.info("Chronos-2: %d/%d origins  %.2fs/origin", k, len(origins), r)
    return pd.concat(blocks, ignore_index=True)


def daily_campus(preds: pd.DataFrame) -> pd.DataFrame:
    """Campus-total demand per calendar day, actual and predicted.

    Rows with a missing actual are dropped from **both** series, so the pair stays comparable;
    ``coverage_pct`` records how much of the day survived, which matters on 2026-03-16 and the
    campus-wide 11-hour outage on 2026-03-25.
    """
    p = preds.copy()
    p["day"] = p["timestamp"].dt.normalize()
    full = p.groupby(["model", "day"]).size().rename("n_expected")
    ok = p.dropna(subset=["actual", "pred"])
    agg = ok.groupby(["model", "day"]).agg(
        actual_kl=("actual", "sum"), pred_kl=("pred", "sum"), n_scored=("actual", "size")
    ).join(full)
    agg["coverage_pct"] = 100 * agg["n_scored"] / agg["n_expected"]
    agg["error_kl"] = agg["pred_kl"] - agg["actual_kl"]
    agg["abs_error_kl"] = agg["error_kl"].abs()
    agg["error_pct"] = 100 * agg["error_kl"] / agg["actual_kl"]
    return agg.reset_index()


def summary(daily: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in daily.groupby("model"):
        full = g[g["coverage_pct"] >= 95]
        h = preds[preds.model == model].dropna(subset=["actual", "pred"])
        rows.append({
            "model": model,
            "days": int(len(g)), "days_full_coverage": int(len(full)),
            "mean_actual_kl_day": float(full["actual_kl"].mean()),
            "mean_pred_kl_day": float(full["pred_kl"].mean()),
            "daily_mae_kl": float(full["abs_error_kl"].mean()),
            "daily_mape_pct": float((100 * full["abs_error_kl"] / full["actual_kl"]).mean()),
            "total_bias_pct": float(100 * (full["pred_kl"].sum() - full["actual_kl"].sum())
                                    / full["actual_kl"].sum()),
            "days_under_forecast": int((full["error_kl"] < 0).sum()),
            "hourly_mae_kl_h": float((h["actual"] - h["pred"]).abs().mean()),
        })
    order = {CHRONOS: 0, INCUMBENT: 1, REFERENCE: 2}
    return pd.DataFrame(rows).sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=HOLDOUT_DAYS)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--models", nargs="+",
                    default=["chronos2", "npts", "seasonal_naive"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.curate import load_curated_hourly

    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_curated_hourly()
    origins = build_origins(panel, args.days)
    logger.info("Holdout: %s .. %s  (%d daily origins, horizon %d h)",
                origins[0] + pd.Timedelta(hours=1), panel["timestamp"].max(),
                len(origins), HORIZON)

    parts = []
    if "seasonal_naive" in args.models:
        parts.append(seasonal_naive(panel, origins))
        logger.info("SeasonalNaive-24 done")
    if "npts" in args.models:
        parts.append(npts(panel, origins, context=args.context,
                          time_limit=args.time_limit, model_dir=OUT / "_npts_model"))
    if "chronos2" in args.models:
        parts.append(chronos2(panel, origins, context=args.context, device=args.device))

    preds = pd.concat(parts, ignore_index=True)

    # Leakage guard: nothing predicted may precede its own origin.
    assert (preds["timestamp"] > preds["origin"]).all(), "leakage: timestamp <= origin"
    # Continuity guard: the 45 days must tile exactly once, no gaps, no overlap.
    for model, g in preds.groupby("model"):
        ts = pd.Series(sorted(g["timestamp"].unique()))
        assert ts.is_unique, f"{model}: overlapping predictions"
        gaps = ts.diff().dropna().unique()
        assert len(gaps) == 1 and gaps[0] == pd.Timedelta(hours=1), f"{model}: gaps {gaps}"
    logger.info("Guards passed: no leakage, no gaps, no overlap")

    preds.to_parquet(OUT / "predictions_holdout45.parquet", index=False)
    daily = daily_campus(preds); daily.to_csv(OUT / "daily_campus.csv", index=False)
    summ = summary(daily, preds);  summ.to_csv(OUT / "summary.csv", index=False)

    (OUT / "manifest.json").write_text(json.dumps({
        "window_start": str(origins[0] + pd.Timedelta(hours=1)),
        "window_end": str(panel["timestamp"].max()),
        "days": len(origins), "horizon_h": HORIZON,
        "origins": [str(o) for o in origins],
        "context": args.context, "device": args.device,
        "models": sorted(preds["model"].unique().tolist()),
        "rows": int(len(preds)),
        "note": "Separate from the published 24-origin benchmark. Consecutive daily origins so "
                "the forecasts tile the window continuously and can be drawn as one series.",
    }, indent=2))

    print("\n" + summ.to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- figures

def plot_all() -> None:
    """One actual-vs-predicted figure per model, on a shared y-axis so they compare directly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from .style import caption as _caption
    from .style import COLOR, GRID, INK, INK2, SURFACE

    plots = OUT / "plots"; plots.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(OUT / "daily_campus.csv", parse_dates=["day"])
    summ = pd.read_csv(OUT / "summary.csv").set_index("model")

    col = {CHRONOS: COLOR["Chronos2-ZS"], INCUMBENT: COLOR["NPTS"],
           REFERENCE: COLOR["SeasonalNaive"]}
    label = {CHRONOS: "Chronos-2 (zero-shot)", INCUMBENT: "NPTS (incumbent)",
             REFERENCE: "SeasonalNaive-24 (reference)"}
    fid = {CHRONOS: "X", INCUMBENT: "Y", REFERENCE: "Z"}
    fname = {CHRONOS: "X_holdout45_chronos2", INCUMBENT: "Y_holdout45_npts",
             REFERENCE: "Z_holdout45_seasonal_naive"}

    ymax = max(daily["actual_kl"].max(), daily["pred_kl"].max()) * 1.30
    rmax = daily["error_kl"].abs().max() * 1.18
    low = daily[daily["coverage_pct"] < 95]["day"].unique()

    for model in (CHRONOS, INCUMBENT, REFERENCE):
        g = daily[daily.model == model].sort_values("day").reset_index(drop=True)
        s = summ.loc[model]
        c = col[model]

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(11.5, 5.6), sharex=True,
            gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.12})

        # shade the reduced-coverage days so a dip is never read as a demand drop
        for d in low:
            for a in (ax, axr):
                a.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                          color="#f0efec", zorder=0)

        ax.fill_between(g["day"], g["actual_kl"], g["pred_kl"], color=c, alpha=0.16,
                        zorder=1, label="error")
        ax.plot(g["day"], g["actual_kl"], color=INK, lw=2.3, zorder=3, label="Actual")
        ax.plot(g["day"], g["pred_kl"], color=c, lw=2.0, ls="--", zorder=4,
                label=f"Predicted — {label[model]}")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Campus demand per day (KL)")
        ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=3)
        ax.set_yticks([0, 100, 200, 300])
        ax.set_title(f"{label[model]} — 45 consecutive days, 1-day-ahead forecast",
                     fontsize=11.5, color=INK, loc="left")

        box = (f"daily MAE  {s.daily_mae_kl:.1f} KL\n"
               f"MAPE       {s.daily_mape_pct:.1f} %\n"
               f"total bias {s.total_bias_pct:+.1f} %\n"
               f"hourly MAE {s.hourly_mae_kl_h:.3f} KL/h")
        ax.text(0.988, 0.955, box, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, family="monospace", color=INK2,
                bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=GRID, lw=1))

        axr.bar(g["day"], g["error_kl"], color=c, alpha=0.85, width=0.78, zorder=2)
        axr.axhline(0, color=INK2, lw=1.1, zorder=3)
        axr.set_ylim(-rmax, rmax)
        axr.set_ylabel("Error (KL)\npred − actual")
        axr.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        axr.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axr.set_xlim(g["day"].min() - pd.Timedelta(hours=14),
                     g["day"].max() + pd.Timedelta(hours=14))
        for lab in axr.get_xticklabels():
            lab.set_fontsize(8.5)

        under = int(s.days_under_forecast)
        _caption(fig, f"Campus-total demand over {int(s.days)} consecutive days "
                      f"({g['day'].min():%d %b} – {g['day'].max():%d %b %Y}). Each day is forecast "
                      f"from an origin at 23:00 the previous night, so the 45 forecasts tile the "
                      f"window exactly once — no gaps, no overlap, no leakage. Shaded columns are "
                      f"the two days with under 95 % sensor coverage (16 Mar, and the campus-wide "
                      f"11-hour outage on 25 Mar); both series are computed on the same surviving "
                      f"hours, so the pair stays comparable. This model under-forecasts on "
                      f"{under} of {int(s.days)} days.", y=-0.06)
        for ext in ("png", "svg"):
            fig.savefig(plots / f"{fname[model]}.{ext}")
        plt.close(fig)
        print(f"  [plot] {fid[model]} — {fname[model]}")


if __name__ == "__main__" and False:
    pass
