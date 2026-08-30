"""AutoGluon baselines evaluated on the *identical* backtest grid as Chronos-2.

Each horizon needs its own predictor because ``prediction_length`` is fixed at fit time. Every
predictor is fitted strictly on data **before the first backtest origin**, so no holdout
information leaks into any baseline.

``NPTS`` matters most here: it is the model that currently ships as ``results/autogluon`` (the
saved WeightedEnsemble has weight 1.0 on NPTS), so it is the incumbent Chronos-2 has to beat.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import KNOWN_COVARIATES, BacktestSpec, actuals_after, history_before

logger = logging.getLogger(__name__)

STATISTICAL = {"SeasonalNaive": {}, "NPTS": {}, "ETS": {}, "Theta": {},
               "DynamicOptimizedTheta": {}}
NEURAL = {"PatchTST": {"max_epochs": 30}, "TiDE": {"max_epochs": 30}}


def _to_tsdf(df: pd.DataFrame, target_col: str):
    from autogluon.timeseries import TimeSeriesDataFrame

    cols = ["item_id", "timestamp", target_col] + [c for c in KNOWN_COVARIATES if c in df.columns]
    return TimeSeriesDataFrame.from_data_frame(
        df[cols], id_column="item_id", timestamp_column="timestamp"
    )


def run_horizon(
    panel: pd.DataFrame,
    spec: BacktestSpec,
    horizon: int,
    *,
    model_dir: Path,
    hyperparameters: dict,
    time_limit: int,
    context: int,
) -> pd.DataFrame:
    from autogluon.timeseries import TimeSeriesPredictor

    fit_cutoff = spec.origins[0]
    train = panel[panel["timestamp"] <= fit_cutoff]
    logger.info("h=%d  fitting on %d rows up to %s", horizon, len(train), fit_cutoff)

    predictor = TimeSeriesPredictor(
        path=str(model_dir / f"h{horizon}"),
        target=spec.target_col,
        prediction_length=horizon,
        freq="h",
        eval_metric="MASE",
        known_covariates_names=[c for c in KNOWN_COVARIATES if c in panel.columns],
        verbosity=0,
    )
    predictor.fit(
        train_data=_to_tsdf(train, spec.target_col),
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        num_val_windows=1,
        enable_ensemble=False,
    )
    trained = list(predictor.model_names())
    logger.info("h=%d  trained: %s", horizon, trained)

    blocks = []
    for origin in spec.origins:
        hist = history_before(panel, origin, context=context)
        fut = actuals_after(panel, origin, horizon)
        known = _to_tsdf(fut, spec.target_col).drop(columns=[spec.target_col])
        tsdf = _to_tsdf(hist, spec.target_col)

        for model in trained:
            pred = predictor.predict(tsdf, known_covariates=known, model=model)
            pred = pred.reset_index()[["item_id", "timestamp", "mean", "0.1", "0.5", "0.9"]]
            merged = fut[["item_id", "timestamp", "step", spec.target_col]].merge(
                pred, on=["item_id", "timestamp"], how="left"
            ).rename(columns={spec.target_col: "actual"})
            merged["pred"] = np.clip(merged["mean"].to_numpy(), 0.0, None)
            merged = merged.drop(columns=["mean"])
            merged["model"] = model
            merged["origin"] = origin
            merged["horizon"] = horizon
            blocks.append(merged)
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--n-origins", type=int, default=None,
                    help="Number of backtest origins (default: the grid default, 24).")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--time-limit", type=int, default=900)
    ap.add_argument("--include-neural", action="store_true")
    ap.add_argument("--out-dir", default="results/chronos2")
    ap.add_argument("--model-dir", default="results/chronos2/_ag_models")
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
    logger.info("Backtest: %s", spec.describe())

    hparams = dict(STATISTICAL)
    if args.include_neural:
        hparams.update(NEURAL)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    frames, timings = [], {}
    for horizon in spec.horizons:
        t0 = time.time()
        frames.append(run_horizon(
            panel, spec, horizon, model_dir=model_dir, hyperparameters=hparams,
            time_limit=args.time_limit, context=args.context,
        ))
        timings[horizon] = round(time.time() - t0, 1)
        logger.info("h=%d done in %.1f min", horizon, timings[horizon] / 60)

    preds = pd.concat(frames, ignore_index=True)
    path = out_dir / "predictions_autogluon_baselines.parquet"
    preds.to_parquet(path, index=False)
    (out_dir / "baselines_manifest.json").write_text(json.dumps({
        "hyperparameters": {k: v for k, v in hparams.items()},
        "time_limit": args.time_limit, "context": args.context,
        "fit_cutoff": str(spec.origins[0]), "seconds_per_horizon": timings,
        "models": sorted(preds["model"].unique().tolist()),
    }, indent=2))
    logger.info("Wrote %d rows -> %s", len(preds), path)
    logger.info("Models: %s", sorted(preds["model"].unique()))


if __name__ == "__main__":
    main()
