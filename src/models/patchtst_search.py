"""Find PatchTST's best configuration on this dataset, so the comparison is fair.

The earlier PatchTST runs used AutoGluon's defaults and the settings the PatchTST paper uses for
hourly benchmarks. Both lost to Chronos-2, and the results page said so with an explicit caveat:
*two configurations is not an architecture search*. This module removes that caveat by running one.

The search began from the property that characterises this dataset — **~24 % of hourly readings
are exactly zero** — and the hypothesis that PatchTST's best route to beating a foundation model
would be a distribution head able to represent that zero atom. **That route is closed**, and
closing it is itself a result: every head that can put mass at exactly zero is a quantile-function
output, and gluonts' PatchTST computes its loss as a negative log-likelihood, which those outputs
do not implement. Measured head by head in ``head_viability.json``.

What remains is still worth searching, and includes one head that matters: ``Laplace``, whose
negative log-likelihood is equivalent to mean absolute error — so it trains against the same
quantity MASE scores. Alongside it the space covers context length (the PatchTST paper's hourly
settings sit at 336-720), patch geometry, capacity, the optimiser, and the target scaler, which
matters here because tank demand spans three orders of magnitude.

**Why a hand-rolled loop rather than AutoGluon's HPO:** the distribution head has to be passed as a
live GluonTS object, and Ray Tune's search-space serialiser cannot carry one. Sampling configs here
keeps the head a first-class axis, and makes every trial's config and score legible in one CSV.

**Leakage.** Every trial fits on data at or before the first *test* origin and is scored on
AutoGluon's internal validation windows, carved from the tail of that training data. No trial ever
sees a test row. The winner is then refitted and rolled forward over the real grid by
``patchtst_benchmark --preset best``.

    python -m src.models.patchtst_search --num-trials 20 --minutes 90
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import traceback
from pathlib import Path

import pandas as pd

from .backtest import KNOWN_COVARIATES, make_spec

logger = logging.getLogger(__name__)

OUT = Path("results/chronos2/patchtst_search")

# The three heads that actually train. Every head able to represent a point mass at zero —
# PiecewiseLinear, ImplicitQuantileNetwork — is a quantile-function output, and gluonts' PatchTST
# module computes its loss as a negative log-likelihood, which those outputs do not implement;
# NegativeBinomial and Beta reject the continuous, unbounded target. Measured and recorded in
# ``results/chronos2/patchtst_search/head_viability.json``. Laplace is the interesting one: its
# NLL is equivalent to MAE, so it trains against the metric the benchmark scores on.
HEADS = ["StudentT", "Laplace", "Normal"]


def build_head(name: str):
    from gluonts.torch.distributions import LaplaceOutput, NormalOutput, StudentTOutput

    return {"StudentT": StudentTOutput, "Laplace": LaplaceOutput, "Normal": NormalOutput}[name]()


def sample_config(rng: random.Random, *, contexts=(192, 336, 512, 720, 1024)) -> dict:
    """One draw from the space. ``nhead`` must divide ``d_model``; ``stride`` must not exceed
    ``patch_len``, or the patches would leave gaps in the context."""
    d_model = rng.choice([32, 64, 128])
    patch_len = rng.choice([8, 16, 24, 32])
    return {
        "head": rng.choice(HEADS),
        "context_length": rng.choice(list(contexts)),
        "patch_len": patch_len,
        "stride": rng.choice([s for s in (4, 8, 16) if s <= patch_len]),
        "d_model": d_model,
        "nhead": rng.choice([h for h in (4, 8) if d_model % h == 0]),
        "num_encoder_layers": rng.choice([2, 3, 4]),
        "lr": round(10 ** rng.uniform(-4, -2.5), 6),
        "weight_decay": round(10 ** rng.uniform(-8, -3), 10),
        "scaling": rng.choice(["mean", "std"]),
        # tank demand spans three orders of magnitude, so how the series is normalised before
        # the model ever sees it is a first-order choice, not a detail
        "target_scaler": rng.choice(["standard", "mean_abs", "robust"]),
    }


def to_hyperparameters(cfg: dict, *, max_epochs: int, num_batches: int) -> dict:
    hp = {k: v for k, v in cfg.items() if k != "head"}
    hp.update({"distr_output": build_head(cfg["head"]), "max_epochs": max_epochs,
               "num_batches_per_epoch": num_batches, "batch_size": 64})
    return hp


def _persist(rows: list[dict], append: bool) -> pd.DataFrame:
    """Write trials.csv, merging with any earlier phase so the two searches read as one table."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = OUT / "trials.csv"
    if append and path.exists():
        prior = pd.read_csv(path)
        prior = prior[prior.get("phase", pd.Series(dtype=str)) != df["phase"].iloc[0]] \
            if "phase" in prior.columns else prior
        df = pd.concat([prior, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def _to_tsdf(df: pd.DataFrame, target_col: str):
    from autogluon.timeseries import TimeSeriesDataFrame

    cols = ["item_id", "timestamp", target_col] + [c for c in KNOWN_COVARIATES if c in df.columns]
    return TimeSeriesDataFrame.from_data_frame(
        df[cols], id_column="item_id", timestamp_column="timestamp"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--num-trials", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=90, help="Wall-clock budget for the search.")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--num-batches", type=int, default=100)
    ap.add_argument("--num-val-windows", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--contexts", nargs="+", type=int, default=[192, 336, 512, 720, 1024],
                    help="Restrict the context axis. A wide, cheap sweep caps this to keep the "
                         "per-trial cost roughly constant; the refit uses the full range.")
    ap.add_argument("--phase", default="deep", help="Tag written into trials.csv and the config.")
    ap.add_argument("--append", action="store_true",
                    help="Merge into an existing trials.csv rather than replacing it.")
    ap.add_argument("--model-dir", default="results/chronos2/_patchtst_search")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from autogluon.timeseries import TimeSeriesPredictor

    from ..data.curate import load_curated_hourly

    panel = load_curated_hourly()
    spec = make_spec(panel)
    fit_cutoff = spec.origins[0]
    train = _to_tsdf(panel[panel["timestamp"] <= fit_cutoff], spec.target_col)

    logger.info("Tuning PatchTST at h=%d — %d trials, %.0f min budget",
                args.horizon, args.num_trials, args.minutes)
    logger.info("Fit window ends %s (the first TEST origin); validation is %d window(s) carved "
                "from the tail of that — no trial sees a test row",
                fit_cutoff, args.num_val_windows)

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    deadline = time.time() + args.minutes * 60
    rows, seen = [], set()

    for trial in range(1, args.num_trials + 1):
        if time.time() > deadline:
            logger.info("Budget exhausted after %d trials", trial - 1)
            break
        cfg = sample_config(rng, contexts=args.contexts)
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)

        t0 = time.time()
        try:
            predictor = TimeSeriesPredictor(
                path=f"{args.model_dir}/t{trial}", target=spec.target_col,
                prediction_length=args.horizon, freq="h", eval_metric="MASE",
                known_covariates_names=[c for c in KNOWN_COVARIATES if c in panel.columns],
                verbosity=0,
            )
            predictor.fit(
                train_data=train,
                hyperparameters={"PatchTST": to_hyperparameters(
                    cfg, max_epochs=args.max_epochs, num_batches=args.num_batches)},
                num_val_windows=args.num_val_windows,
                enable_ensemble=False,
            )
            val_mase = float(-predictor.leaderboard()["score_val"].iloc[0])
            status = "ok"
        except Exception as exc:                      # a bad head/shape combination is a datapoint
            val_mase, status = float("nan"), f"failed: {type(exc).__name__}"
            logger.warning("trial %d failed: %s", trial, exc)
            logger.debug(traceback.format_exc())

        secs = round(time.time() - t0, 1)
        rows.append({"trial": trial, "phase": args.phase, **cfg, "val_MASE": val_mase,
                     "seconds": secs, "status": status,
                     "max_epochs": args.max_epochs, "num_batches": args.num_batches})
        logger.info("trial %2d  val MASE %s  %5.0fs  head=%-14s ctx=%4d patch=%2d d=%3d L=%d "
                    "lr=%.2e scal=%s", trial,
                    f"{val_mase:.4f}" if val_mase == val_mase else "  n/a", secs,
                    cfg["head"], cfg["context_length"], cfg["patch_len"], cfg["d_model"],
                    cfg["num_encoder_layers"], cfg["lr"], cfg["scaling"])
        _persist(rows, args.append)

    trials = _persist(rows, args.append).sort_values("val_MASE")
    ok = trials[trials.status == "ok"]
    if ok.empty:
        raise SystemExit("every trial failed — nothing to select")

    best = ok.iloc[0]
    cfg = {k: (best[k].item() if hasattr(best[k], "item") else best[k])
           for k in sample_config(random.Random(0)) if k in best.index}
    payload = {
        "tuned_at_horizon": args.horizon,
        "trials_run": int(len(trials)), "trials_ok": int(len(ok)),
        "phases": sorted(trials["phase"].dropna().unique().tolist())
        if "phase" in trials.columns else [args.phase],
        "search_seconds": float(trials["seconds"].sum()),
        "fit_cutoff": str(fit_cutoff), "num_val_windows": args.num_val_windows,
        "search_max_epochs": args.max_epochs, "search_num_batches": args.num_batches,
        "best_val_MASE": float(best.val_MASE),
        "worst_val_MASE": float(ok.val_MASE.max()),
        "best_config": cfg,
        "val_MASE_by_head": ok.groupby("head")["val_MASE"].min().round(4).to_dict(),
    }
    (OUT / "best_config.json").write_text(json.dumps(payload, indent=2))
    logger.info("\n%s", ok[["trial", "head", "context_length", "patch_len", "d_model",
                            "num_encoder_layers", "lr", "scaling", "val_MASE"]].to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
