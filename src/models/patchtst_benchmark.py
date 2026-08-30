"""PatchTST evaluated on the *identical* backtest grid as Chronos-2 and the AutoGluon baselines.

PatchTST (Nie et al., ICLR 2023) is a patched-channel-independent Transformer for long-horizon
forecasting. Unlike Chronos-2 it is **trained on this dataset**: one predictor per horizon,
fitted strictly on data at or before the first backtest origin, then rolled forward over all 24
origins. That is the same protocol ``baselines_autogluon.py`` uses, so nothing leaks and the row
counts stay identical to every other model.

Why it gets its own module rather than ``--include-neural``:
    the statistical baselines and PatchTST have very different fit costs, and re-running the
    26-minute statistical sweep just to add one neural model wastes an hour. This writes its own
    ``predictions_PatchTST.parquet``, which ``score_benchmark`` picks up by glob.

Run:
    python -m src.models.patchtst_benchmark --preset default   # AutoGluon defaults,  ~3 min
    python -m src.models.patchtst_benchmark --preset tuned \
        --out-name predictions_PatchTST-Tuned.parquet \
        --manifest-name patchtst_tuned_manifest.json \
        --model-dir results/chronos2/_patchtst_tuned_models       # paper settings, ~40 min
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

# Two configurations are run, and both are reported.
#
#   PatchTST        AutoGluon's shipped defaults - context_length 96 (4 days), 30 epochs,
#                   50 batches per epoch. This is what `baselines_autogluon --include-neural`
#                   would have produced, so it is the configuration the earlier scope note
#                   referred to.
#   PatchTST-Tuned  context_length 512 (3 weeks, the setting the PatchTST paper uses for
#                   hourly data), 100 epochs, 200 batches per epoch. Chronos-2 sees 2048 hours
#                   of context; leaving PatchTST at 96 would make the comparison unfair, and a
#                   result that only holds against a handicapped opponent is not a result.
#
# Both are fitted on the same pre-origin window and scored on the same rows.
PRESETS = {
    "default": {"max_epochs": 30, "context_length": 96},
    "tuned": {"max_epochs": 100, "context_length": 512, "num_batches_per_epoch": 200,
              "lr": 1e-3, "batch_size": 64},
}


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
    label: str | None = None,
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
        verbosity=1,
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
            merged["model"] = label or model
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
    ap.add_argument("--time-limit", type=int, default=3600)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default")
    ap.add_argument("--label", default=None,
                    help="Name written into the model column (default: the preset's name).")
    ap.add_argument("--out-dir", default="results/chronos2")
    ap.add_argument("--model-dir", default="results/chronos2/_patchtst_models")
    ap.add_argument("--out-name", default="predictions_PatchTST.parquet")
    ap.add_argument("--manifest-name", default="patchtst_manifest.json")
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
    logger.info("Panel: %d rows, %d tanks, %s .. %s", len(panel),
                panel["item_id"].nunique(), panel["timestamp"].min(), panel["timestamp"].max())

    hparams = {"PatchTST": dict(PRESETS[args.preset])}
    label = args.label or ("PatchTST" if args.preset == "default" else "PatchTST-Tuned")
    logger.info("Preset %r -> %s  hyperparameters=%s", args.preset, label, hparams["PatchTST"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    frames, timings = [], {}
    for horizon in spec.horizons:
        t0 = time.time()
        frames.append(run_horizon(
            panel, spec, horizon, model_dir=model_dir, hyperparameters=hparams,
            time_limit=args.time_limit, context=args.context, label=label,
        ))
        timings[horizon] = round(time.time() - t0, 1)
        logger.info("h=%d done in %.1f min", horizon, timings[horizon] / 60)

    preds = pd.concat(frames, ignore_index=True)
    path = out_dir / args.out_name
    preds.to_parquet(path, index=False)
    (out_dir / args.manifest_name).write_text(json.dumps({
        "preset": args.preset, "label": label,
        "hyperparameters": hparams,
        "time_limit": args.time_limit, "context": args.context,
        "panel_start": str(panel["timestamp"].min()),
        "panel_end": str(panel["timestamp"].max()),
        "n_tanks": int(panel["item_id"].nunique()),
        "fit_cutoff": str(spec.origins[0]),
        "n_origins": len(spec.origins),
        "origin_stride_hours": 23,
        "horizons": list(spec.horizons),
        "seconds_per_horizon": timings,
        "wall_clock_s": round(sum(timings.values()), 1),
        "models": sorted(preds["model"].unique().tolist()),
        "rows": int(len(preds)),
    }, indent=2))
    logger.info("Wrote %d rows -> %s", len(preds), path)
    logger.info("Models: %s", sorted(preds["model"].unique()))
    logger.info("Total: %.1f min", sum(timings.values()) / 60)


if __name__ == "__main__":
    main()
