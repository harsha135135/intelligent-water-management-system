"""Build ``forecast_bundle.json`` for the Waltr right-dock extension.

The dock ships with a precomputed bundle so the demo runs with **no backend and no network** -
which is what makes it dependable in a review room. The bundle carries, per tank:

* the trailing observed history the chart draws to the left of the "now" line,
* a genuine forward forecast from the end of the dataset at all six horizons, with p10/p90,
* the tank's current level and capacity, so the dock can compute time-to-empty,
* the tank's measured MASE at each horizon, taken from the benchmark - so the UI shows how much
  the number can be trusted rather than presenting a bare point forecast.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EDA_DIR, RESULTS_DIR
from ..data.curate import TARGET_COL, load_curated_hourly, tank_metadata
from .backtest import KNOWN_COVARIATES, PAST_COVARIATES, HORIZONS

logger = logging.getLogger(__name__)

HISTORY_HOURS = 168


def capacity_kl(dim: str | float) -> float | None:
    """Parse the ``Tank Dimensions`` string into a capacity in KL.

    Mirrors ``compute_capacity_kl`` in ``eda/eda_deep.py``: three values are a cuboid
    (H x L x B), two are a cylinder (H x R). Source units are cm, so cm^3 / 1e6 -> KL.
    """
    if not isinstance(dim, str) or not dim.strip():
        return None
    vals = [float(m) for m in re.findall(r"([\d.]+)\s*cm", dim)]
    if len(vals) == 3:
        vol = vals[0] * vals[1] * vals[2]
    elif len(vals) == 2:
        vol = math.pi * vals[1] ** 2 * vals[0]
    else:
        return None
    return vol / 1_000_000


def forward_forecast(pipe, panel: pd.DataFrame, horizons: list[int], context: int, *,
                     covariates: bool = True) -> dict:
    """Forecast forward from the true end of the dataset.

    With ``covariates`` the model also sees the past-only and known-future covariates, as the
    Chronos2-COV variants do; without, it sees the target alone, which is the Chronos2-ZS model
    the benchmark scores."""
    from ..data.calendar_pesu import add_academic_features
    from ..data.curate import add_time_features

    end = panel["timestamp"].max()
    hist = panel.groupby("item_id", sort=False).tail(context)
    out: dict[int, pd.DataFrame] = {}

    for h in horizons:
        future_index = pd.date_range(end + pd.Timedelta(hours=1), periods=h, freq="h")
        fut = pd.concat([
            pd.DataFrame({"item_id": item_id, "timestamp": future_index})
            for item_id in sorted(panel["item_id"].unique())
        ], ignore_index=True)
        fut = add_academic_features(add_time_features(fut))

        if covariates:
            cols = (["item_id", "timestamp", TARGET_COL]
                    + [c for c in PAST_COVARIATES if c in hist.columns]
                    + [c for c in KNOWN_COVARIATES if c in hist.columns])
            future = fut[["item_id", "timestamp"] + KNOWN_COVARIATES]
        else:
            cols, future = ["item_id", "timestamp", TARGET_COL], None
        preds = pipe.predict_df(
            hist[cols], future_df=future,
            id_column="item_id", timestamp_column="timestamp", target=TARGET_COL,
            prediction_length=h, quantile_levels=[0.1, 0.5, 0.9], batch_size=32,
        )
        for c in ("predictions", "0.1", "0.5", "0.9"):
            preds[c] = np.clip(preds[c].to_numpy(), 0.0, None)
        out[h] = preds
        logger.info("forward forecast h=%d done", h)
    return out


def forward_forecast_timestral(panel: pd.DataFrame, horizons: list[int], context: int) -> dict:
    """The same forward forecast from Timestral 209M, through the benchmark's own adapter, so the
    median and the 0.1 / 0.9 quantiles are exactly the scored configuration's. Target only."""
    from .timestral_forecasting import TimestralForecaster

    fc = TimestralForecaster.load()
    end = panel["timestamp"].max()
    items = sorted(panel["item_id"].unique())
    hist = {i: g.sort_values("timestamp")[TARGET_COL].tail(context).to_numpy(dtype=float)
            for i, g in panel.groupby("item_id", sort=False)}
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        point, q = fc.forecast(h, [hist[i] for i in items])
        ts = pd.date_range(end + pd.Timedelta(hours=1), periods=h, freq="h")
        out[h] = pd.concat([pd.DataFrame({
            "item_id": item, "timestamp": ts,
            "predictions": np.clip(point[k], 0.0, None), "0.1": np.clip(q[k, :, 1], 0.0, None),
            "0.5": np.clip(q[k, :, 5], 0.0, None), "0.9": np.clip(q[k, :, 9], 0.0, None)})
            for k, item in enumerate(items)], ignore_index=True)
        logger.info("forward forecast h=%d done", h)
    return out


def calibrate(fc: dict, cal_dir: Path, model: str) -> dict:
    """Apply the per-tank volume factor and conformal tail offsets fitted for ``model`` in the
    calibrated 45-day view (``calibration.apply``). They were fitted on 1-day-ahead forecasts and
    are applied unchanged at every horizon."""
    from . import calibration as cal

    params = {"bias": pd.read_csv(cal_dir / "bias_factors.csv"),
              "conformal": pd.read_csv(cal_dir / "conformal_offsets.csv")}
    if model not in set(params["bias"]["model"]):
        raise SystemExit(f"{cal_dir} has no calibration for {model}")
    out, widened, total = {}, 0, 0
    for h, df in fc.items():
        d = df.rename(columns={"predictions": "pred", "0.1": "q10", "0.9": "q90"}).assign(model=model)
        c = cal.apply(d, params)
        pred = c["pred_cal"].to_numpy()
        # The tails are corrected independently of the point, so a tank whose fitted offset
        # tightens a tail can end with its band on one side of its own forecast. The dock draws
        # the band around the line, so the band is widened to contain the point there.
        lo = np.minimum(c["lower_cal"].to_numpy(), pred)
        hi = np.maximum(c["upper_cal"].to_numpy(), pred)
        widened += int(((lo != c["lower_cal"].to_numpy()) | (hi != c["upper_cal"].to_numpy())).sum())
        total += len(pred)
        out[h] = df.assign(predictions=pred, **{"0.1": lo, "0.9": hi})
    logger.info("calibrated with %s; band widened to contain the point on %d of %d points",
                model, widened, total)
    return out


def check_deployable(model: str) -> None:
    """The extension shows forecasts to end users. A model whose weights' licence forbids that
    (TimesFM 3.0: non-commercial, non-production) must never be packaged into it."""
    from .timesfm3_forecasting import DEPLOYABLE, LICENSE, MODEL_NAME as TIMESFM3
    if model == TIMESFM3 and not DEPLOYABLE:
        raise SystemExit(f"{model} cannot be bundled into the extension: its weights' licence "
                         f"({LICENSE['weights']}) rules out use with end users or in production.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--trust-json", default=str(EDA_DIR / "tank_trust.json"))
    ap.add_argument("--out", default="extension/waltr-dock/forecast_bundle.json")
    ap.add_argument("--model-label", default="Chronos-2 · zero-shot + covariates")
    ap.add_argument("--metric-model", default="Chronos2-COV")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", choices=["chronos2", "timestral"], default="chronos2",
                    help="Chronos-2 (with or without covariates) or Timestral 209M (target only)")
    ap.add_argument("--calibration-dir", default=None,
                    help="apply the per-tank volume factor and conformal offsets fitted for "
                         "--metric-model in this calibrated run (e.g. <results>/calibrated)")
    ap.add_argument("--covariates", action=argparse.BooleanOptionalAction, default=True,
                    help="Condition on the covariates (default) or forecast from the target "
                         "alone, as Chronos2-ZS does. Keep --model-label and --metric-model "
                         "consistent with this choice.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    check_deployable(args.metric_model)

    panel = load_curated_hourly()
    end = panel["timestamp"].max()
    meta = tank_metadata().set_index("item_id")

    trust = {}
    tp = Path(args.trust_json)
    if tp.exists():
        trust = json.loads(tp.read_text())

    # Per-tank MASE from the benchmark, if it has been scored.
    accuracy: dict[str, dict[str, dict]] = {}
    mp = Path(args.results_dir) / "metrics_per_tank.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        m = m[m["model"] == args.metric_model]
        for r in m.itertuples(index=False):
            accuracy.setdefault(r.item_id, {})[str(int(r.horizon))] = {
                "mase": None if pd.isna(r.mase) else round(float(r.mase), 3),
                "mae": round(float(r.mae), 4),
            }
        logger.info("Loaded per-tank accuracy for %s", args.metric_model)
    else:
        logger.warning("No metrics_per_tank.csv yet - bundle will omit accuracy")

    if args.model == "timestral":
        fc = forward_forecast_timestral(panel, HORIZONS, args.context)
    else:
        from .chronos2_forecasting import load_pipeline
        pipe = load_pipeline(args.device)
        fc = forward_forecast(pipe, panel, HORIZONS, args.context, covariates=args.covariates)
    if args.calibration_dir:
        fc = calibrate(fc, Path(args.calibration_dir), args.metric_model)

    tanks = {}
    for item_id, g in panel.groupby("item_id", sort=True):
        g = g.sort_values("timestamp")
        hist = g[TARGET_COL].tail(HISTORY_HOURS).fillna(0.0).round(3).tolist()
        geom_cap = capacity_kl(meta["tank_dimensions"].get(item_id))
        level = g["Closing Value in KL"].dropna()
        level_kl = float(level.iloc[-1]) if len(level) else None

        # The level probe's usable range is shorter than the tank's full height, so geometric
        # capacity overstates what the sensor can ever report. Waltr's own UI shows the sensor
        # range (BE BLOCK OHT: "24.59 out of 60.47 KL" = 255cm of a 280cm tank). Use the observed
        # ceiling so the dock's percentage agrees with the number next to it on screen.
        observed_cap = float(level.quantile(0.999)) if len(level) else None
        cap = observed_cap if observed_cap and observed_cap > 0 else geom_cap

        forecasts = {}
        for h, df in fc.items():
            sub = df[df["item_id"] == item_id].sort_values("timestamp")
            forecasts[str(h)] = [
                {"t": ts.strftime("%H:%M"), "pred": round(float(p), 3),
                 "p10": round(float(lo), 3), "p90": round(float(hi), 3)}
                for ts, p, lo, hi in zip(sub["timestamp"], sub["predictions"],
                                         sub["0.1"], sub["0.9"])
            ]

        tanks[item_id] = {
            "name": str(meta["tank_name"].get(item_id, item_id)),
            "mean_kl": round(float(g[TARGET_COL].mean()), 4),
            "trust": trust.get(item_id, "unknown"),
            "capacity_kl": None if cap is None else round(cap, 2),
            "geometric_capacity_kl": None if geom_cap is None else round(geom_cap, 2),
            "level_kl": None if level_kl is None else round(level_kl, 2),
            "level_pct": (None if (cap in (None, 0) or level_kl is None)
                          else round(100 * min(level_kl / cap, 1.2), 1)),
            "history": hist,
            "forecasts": forecasts,
            "accuracy": accuracy.get(item_id, {}),
        }

    bundle = {
        "model": args.model_label,
        "metric_model": args.metric_model,
        "calibrated": bool(args.calibration_dir),
        "generated_at": end.strftime("%Y-%m-%d %H:%M"),
        "location": "PES University RR",
        "horizons": HORIZONS,
        "tanks": tanks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, separators=(",", ":")))
    logger.info("Wrote %s (%.0f KB, %d tanks)", out, out.stat().st_size / 1024, len(tanks))


if __name__ == "__main__":
    main()
