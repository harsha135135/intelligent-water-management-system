"""PatchTST on the same continuous 45-day daily-origin grid as ``holdout45_continuous``.

``calibrated_holdout`` covers Chronos2-ZS, NPTS and SeasonalNaive-24 over
2026-03-09 .. 2026-04-22 with 45 consecutive daily origins at horizon 24 h. PatchTST was only
ever run on the 23-hour-stride benchmark grid, whose origins overlap and start 2026-03-24, so it
cannot be drawn on the same continuous timeline. This module fills that gap: one predictor fitted
strictly on data at or before the first origin (2026-03-08 23:00), then rolled forward over all
45 origins.

The ``tuned`` preset is the default rather than ``searched``: the searched configuration was
selected with a fit cutoff of 2026-03-24, whose validation windows sit *inside* this reported
window, so using it here would leak. The tuned preset's settings come from the PatchTST paper.

Run:
    python -m src.models.holdout45_patchtst
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR
from .backtest import KNOWN_COVARIATES, actuals_after, history_before
from .holdout45_continuous import HORIZON, TARGET, build_origins
from .patchtst_benchmark import PRESETS, _to_tsdf

logger = logging.getLogger(__name__)

OUT = RESULTS_DIR / "holdout45_patchtst"


def patchtst(panel, origins, *, hyperparameters, context, time_limit, model_dir, label):
    from autogluon.timeseries import TimeSeriesPredictor

    fit_cutoff = origins[0]
    train = panel[panel["timestamp"] <= fit_cutoff]
    logger.info("PatchTST: fitting on %d rows up to %s", len(train), fit_cutoff)

    predictor = TimeSeriesPredictor(
        path=str(model_dir), target=TARGET, prediction_length=HORIZON, freq="h",
        eval_metric="MASE",
        known_covariates_names=[c for c in KNOWN_COVARIATES if c in panel.columns],
        verbosity=1,
    )
    predictor.fit(train_data=_to_tsdf(train, TARGET), hyperparameters=hyperparameters,
                  time_limit=time_limit, num_val_windows=1, enable_ensemble=False)
    trained = list(predictor.model_names())
    logger.info("PatchTST: trained %s", trained)

    blocks, t0 = [], time.time()
    for k, origin in enumerate(origins, 1):
        hist = history_before(panel, origin, context=context)
        fut = actuals_after(panel, origin, HORIZON)
        known = _to_tsdf(fut, TARGET).drop(columns=[TARGET])
        pred = predictor.predict(_to_tsdf(hist, TARGET), known_covariates=known,
                                 model=trained[0])
        pred = pred.reset_index()[["item_id", "timestamp", "mean", "0.1", "0.9"]]
        b = fut[["item_id", "timestamp", "step", TARGET]].merge(
            pred, on=["item_id", "timestamp"], how="left").rename(columns={TARGET: "actual"})
        b["pred"] = np.clip(b["mean"].to_numpy(), 0.0, None)
        b["q10"] = np.clip(b["0.1"].to_numpy(), 0.0, None)
        b["q90"] = np.clip(b["0.9"].to_numpy(), 0.0, None)
        b = b.drop(columns=["mean", "0.1", "0.9"])
        b["model"] = label
        b["origin"] = origin
        blocks.append(b)
        if k % 15 == 0 or k == len(origins):
            logger.info("PatchTST: %d/%d origins  %.2fs/origin", k, len(origins),
                        (time.time() - t0) / k)
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--preset", choices=sorted(PRESETS) + ["searched"], default="tuned")
    ap.add_argument("--config", default=str(RESULTS_DIR / "patchtst_search" / "best_config.json"))
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--time-limit", type=int, default=1800)
    ap.add_argument("--label", default="PatchTST")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.curate import load_curated_hourly

    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_curated_hourly()
    origins = build_origins(panel, args.days)
    logger.info("Holdout: %s .. %s  (%d daily origins, horizon %d h)",
                origins[0] + pd.Timedelta(hours=1), panel["timestamp"].max(),
                len(origins), HORIZON)

    if args.preset == "searched":
        found = json.loads(Path(args.config).read_text())
        cfg = dict(found["best_config"])
        from .patchtst_search import build_head
        cfg["distr_output"] = build_head(cfg.pop("head"))
        cfg.update({"max_epochs": 100, "num_batches_per_epoch": 200, "batch_size": 64})
        hparams = {"PatchTST": cfg}
    else:
        hparams = {"PatchTST": dict(PRESETS[args.preset])}
    logger.info("Preset %r -> %s  %s", args.preset, args.label, hparams["PatchTST"])

    t0 = time.time()
    preds = patchtst(panel, origins, hyperparameters=hparams, context=args.context,
                     time_limit=args.time_limit, model_dir=OUT / "_model", label=args.label)

    assert (preds["timestamp"] > preds["origin"]).all(), "leakage: timestamp <= origin"
    ts = pd.Series(sorted(preds.loc[preds.item_id == preds.item_id.iloc[0], "timestamp"].unique()))
    gaps = ts.diff().dropna().unique()
    assert len(gaps) == 1 and gaps[0] == pd.Timedelta(hours=1), f"gaps {gaps}"
    logger.info("Guards passed: no leakage, no gaps, no overlap")

    preds.to_parquet(OUT / "predictions_holdout45_patchtst.parquet", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "label": args.label, "preset": args.preset,
        "hyperparameters": {"PatchTST": {k: str(v) for k, v in hparams["PatchTST"].items()}},
        "window": [str(origins[0] + pd.Timedelta(hours=1)), str(panel["timestamp"].max())],
        "fit_cutoff": str(origins[0]), "n_origins": len(origins), "horizon_h": HORIZON,
        "context": args.context, "n_tanks": int(panel["item_id"].nunique()),
        "rows": int(len(preds)), "wall_clock_s": round(time.time() - t0, 1),
    }, indent=2))
    mae = (preds["actual"] - preds["pred"]).abs().mean()
    logger.info("Wrote %d rows -> %s   hourly MAE %.4f", len(preds), OUT, mae)


if __name__ == "__main__":
    main()
