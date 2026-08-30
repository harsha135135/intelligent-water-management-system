"""Chronos-2 zero-shot forecasting for PESU campus water demand.

Chronos-2 (``amazon/chronos-2``, 119.5M params) is a pretrained time-series foundation model.
Unlike Chronos-1 - the only Chronos variant previously tried in this repo, inside four old
*daily-frequency* AutoGluon runs where it lost to Theta - Chronos-2 accepts **covariates**:

* *known-future* covariates, supplied via ``future_df``: the clock and academic-calendar terms,
  all of which are known in advance for any forecast origin.
* *past-only* covariates, any column present in ``df`` but absent from ``future_df``: inflow and
  the tank level readings, which are observed but unknown over the forecast window.

That capability is the main reason to prefer Chronos-2 here, and the ``zs`` vs ``cov`` variants
below are what quantify it.

The model's native prediction length is 1024 steps, so every horizon we evaluate (up to 168h)
is produced in a single forward pass with no autoregressive unrolling.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import (
    KNOWN_COVARIATES, PAST_COVARIATES, BacktestSpec, actuals_after, history_before,
)

logger = logging.getLogger(__name__)

MODEL_ID = "amazon/chronos-2"
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Covariates that the horizon-specific MI study (eda/eda_hourly.py section C) found to carry
# real signal. Every calendar flag except exam proximity scored MI < 0.01 at every horizon, so
# the LEAN set drops them: they add variates to each task without adding information.
LEAN_KNOWN = ["hour_sin", "hour_cos", "exam_proximity"]

VARIANTS = {
    # name                known covariates   use_past  cross_learning
    "Chronos2-ZS":       ([],                False,    False),
    "Chronos2-COV":      (KNOWN_COVARIATES,  True,     False),
    "Chronos2-COV-LEAN": (LEAN_KNOWN,        True,     False),
    "Chronos2-COV-XL":   (KNOWN_COVARIATES,  True,     True),
}


def load_pipeline(device: str = "mps"):
    from chronos import Chronos2Pipeline

    logger.info("Loading %s on %s", MODEL_ID, device)
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)


def forecast_one(
    pipe,
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    *,
    target_col: str,
    known_cols: list[str],
    use_past: bool,
    cross_learning: bool,
    context: int,
) -> pd.DataFrame:
    """One (origin, horizon) forecast for all tanks in a single batched call."""
    hist = history_before(panel, origin, context=context)
    fut = actuals_after(panel, origin, horizon)

    cols = ["item_id", "timestamp", target_col]
    if use_past:
        cols += [c for c in PAST_COVARIATES if c in hist.columns]
    cols += [c for c in known_cols if c in hist.columns]

    future_df = fut[["item_id", "timestamp"] + list(known_cols)] if known_cols else None

    out = pipe.predict_df(
        hist[cols],
        future_df=future_df,
        id_column="item_id",
        timestamp_column="timestamp",
        target=target_col,
        prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
        batch_size=32,
        cross_learning=cross_learning,
    )

    # Outflow is physically non-negative; a foundation model has no such prior.
    out["pred"] = np.clip(out["predictions"].to_numpy(), 0.0, None)
    for q in QUANTILE_LEVELS:
        out[str(q)] = np.clip(out[str(q)].to_numpy(), 0.0, None)

    merged = fut[["item_id", "timestamp", "step", target_col]].merge(
        out[["item_id", "timestamp", "pred"] + [str(q) for q in QUANTILE_LEVELS]],
        on=["item_id", "timestamp"], how="left",
    )
    merged = merged.rename(columns={target_col: "actual"})
    merged["origin"] = origin
    merged["horizon"] = horizon
    return merged


def run_variant(
    pipe,
    panel: pd.DataFrame,
    spec: BacktestSpec,
    variant: str,
    *,
    context: int = 2048,
) -> pd.DataFrame:
    known_cols, use_past, cross_learning = VARIANTS[variant]
    blocks, t0 = [], time.time()
    total = len(spec.origins) * len(spec.horizons)
    done = 0
    for origin in spec.origins:
        for horizon in spec.horizons:
            block = forecast_one(
                pipe, panel, origin, horizon,
                target_col=spec.target_col, known_cols=known_cols, use_past=use_past,
                cross_learning=cross_learning, context=context,
            )
            block["model"] = variant
            blocks.append(block)
            done += 1
            if done % 12 == 0 or done == total:
                rate = (time.time() - t0) / done
                logger.info(
                    "%s  %d/%d  %.1fs/call  eta %.1f min",
                    variant, done, total, rate, rate * (total - done) / 60,
                )
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["Chronos2-ZS", "Chronos2-COV"],
                    choices=list(VARIANTS))
    ap.add_argument("--n-origins", type=int, default=None,
                    help="Number of backtest origins (default: the grid default, 24).")
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-dir", default="results/chronos2")
    ap.add_argument("--max-tanks", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.curate import load_curated_hourly
    from .backtest import make_spec

    panel = load_curated_hourly()
    if args.max_tanks:
        keep = sorted(panel["item_id"].unique())[: args.max_tanks]
        panel = panel[panel["item_id"].isin(keep)]
    spec_kw = {"horizons": args.horizons}
    if args.n_origins:
        spec_kw["n_origins"] = args.n_origins
    spec = make_spec(panel, **spec_kw)
    logger.info("Panel: %d rows, %d tanks", len(panel), panel["item_id"].nunique())
    logger.info("Backtest: %s", spec.describe())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.device)
    manifest = {
        "model_id": MODEL_ID, "device": args.device, "context": args.context,
        "n_origins": len(spec.origins), "horizons": spec.horizons,
        "origins": [str(o) for o in spec.origins],
        "tanks": sorted(panel["item_id"].unique().tolist()),
        "panel_rows": int(len(panel)),
        "panel_range": [str(panel["timestamp"].min()), str(panel["timestamp"].max())],
        "variants": {},
    }

    for variant in args.variants:
        t0 = time.time()
        preds = run_variant(pipe, panel, spec, variant, context=args.context)
        elapsed = time.time() - t0
        path = out_dir / f"predictions_{variant}.parquet"
        preds.to_parquet(path, index=False)
        manifest["variants"][variant] = {
            "wall_clock_s": round(elapsed, 1), "rows": int(len(preds)), "path": str(path),
        }
        logger.info("%s done in %.1f min -> %s", variant, elapsed / 60, path)

    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest -> %s", out_dir / "run_manifest.json")


if __name__ == "__main__":
    main()
