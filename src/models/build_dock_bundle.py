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


def forward_forecast(pipe, panel: pd.DataFrame, horizons: list[int], context: int) -> dict:
    """Forecast forward from the true end of the dataset, using known-future covariates."""
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

        cols = (["item_id", "timestamp", TARGET_COL]
                + [c for c in PAST_COVARIATES if c in hist.columns]
                + [c for c in KNOWN_COVARIATES if c in hist.columns])
        preds = pipe.predict_df(
            hist[cols], future_df=fut[["item_id", "timestamp"] + KNOWN_COVARIATES],
            id_column="item_id", timestamp_column="timestamp", target=TARGET_COL,
            prediction_length=h, quantile_levels=[0.1, 0.5, 0.9], batch_size=32,
        )
        for c in ("predictions", "0.1", "0.5", "0.9"):
            preds[c] = np.clip(preds[c].to_numpy(), 0.0, None)
        out[h] = preds
        logger.info("forward forecast h=%d done", h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/chronos2")
    ap.add_argument("--trust-json", default="eda/tank_trust.json")
    ap.add_argument("--out", default="extension/waltr-dock/forecast_bundle.json")
    ap.add_argument("--model-label", default="Chronos-2 · zero-shot + covariates")
    ap.add_argument("--metric-model", default="Chronos2-COV")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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

    from .chronos2_forecasting import load_pipeline
    pipe = load_pipeline(args.device)
    fc = forward_forecast(pipe, panel, HORIZONS, args.context)

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
