"""TimesFM 3.0 zero-shot forecasting on the shared rolling-origin grid.

TimesFM 3.0 (Google Research, August 2026) is the successor to ``TimesFM2.5-ZS``: a 330M-parameter
decoder ("stacked mixing transformer with variate attention") that also takes covariates and
several target series at once. Here it is run like the other three foundation models - pretrained
elsewhere, never fitted on this campus, conditioned on nothing but each tank's own past - so it is
a like-for-like opponent for ``Chronos2-ZS``, ``TimesFM2.5-ZS`` and ``Timestral-209M-ZS``.

Licence - benchmark only
------------------------
The code is Apache-2.0, but the 3.0 *weights* are under the TimesFM Non-Commercial License v1.0:
"internal benchmarking, academic research, and experimentation" are permitted; use "in direct or
indirect interactions with end users or production systems" is not, and the same holds for any
fine-tuned or otherwise adapted copy. This project chooses a model for a production campus system,
so TimesFM 3.0 is scored as a reference point and is **not deployable**: ``DEPLOYABLE = False`` is
written to the manifest, the final analysis keeps it out of the deployment candidates, and
``build_dock_bundle`` refuses it. (Production use is licensed only through Google Cloud, e.g.
BigQuery ML.) There is no official fine-tuning recipe for 3.0, so it is zero-shot only.

Checkpoint
----------
``google/timesfm-3.0-pytorch``, pinned to one Hugging Face revision, loaded through the official
``timesfm3`` package that ships with ``timesfm`` 3.0.x.

Protocol - the same as ``Chronos2-ZS``, ``TimesFM2.5-ZS`` and ``Timestral-209M-ZS``
-----------------------------------------------------------------------------------
* the same 24 origins, six horizons and 24 tanks, and the same 2,048-hour context truncated at
  each origin; one call per (origin, horizon) block, every tank a separate univariate series;
* the target only, no covariates;
* point forecast = the model's median;
* Google's own benchmark settings - those ``TimesFM3Evaluator`` applies to a univariate input:
  quantiles returned and sorted, symmetric averaging on, negative outputs clipped for a
  non-negative context - passed explicitly, then the benchmark's usual clip at zero;
* CPU, the same machine and thread count TimesFM 2.5 and Timestral were timed on.

The model and data plumbing - the leakage assertion, the merge onto the scored window, failure
isolation and the block cache - is ``timesfm_forecasting``'s, imported unchanged. Its block layout
is TimesFM 2.5's ``(point, 10 outputs)``, so the adapter below hands 3.0's quantiles over in that
layout: slot 0 the median, slots 1-9 the 0.1 ... 0.9 quantiles, picked out by level value.

Missing hours: TimesFM 3.0's own input pipeline strips leading NaNs and linearly interpolates the
rest (a gap that runs up to the origin is held at the last observed value) - the same as 2.5.
The adapter passes the raw context with its NaNs and fills nothing. A context that *starts* in a
gap is therefore shorter once stripped, and the package left-pads it, masked, to the batch's
context length. Its forecast depends on that padded length - not on which other tanks share the
batch (checked: identical with any partners) - and in every block of this grid at least one tank
has a full 2,048-hour context, so the padded length is always 2,048 - provided the whole block goes
through in one forward pass, which the adapter enforces (``--batch-size`` at least twice the tank
count). The manifest counts these contexts.

    python -m src.models.timesfm3_forecasting      # -> predictions_TimesFM3-ZS.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR
from .timesfm_forecasting import (
    N_OUTPUTS, QUANTILE_INDEX, BlockCache, RunStats, _hardware, _peak_rss_mb, _rss_mb,
    forecast_one, missing_data_audit,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "TimesFM3-ZS"
MODEL_ID = "google/timesfm-3.0-pytorch"
# Pinned so a later upload to the same Hugging Face repo cannot silently change the benchmark.
MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
LICENSE = {
    "code": "Apache-2.0",
    "weights": "TimesFM Non-Commercial License v1.0 - internal benchmarking, academic research "
               "and experimentation only; no use in production systems or with end users, and "
               "the same for fine-tuned or adapted copies",
}
DEPLOYABLE = False

# The levels handed to timesfm_forecasting's block layout, in its slot order: slot 0 is the
# median (TimesFM 2.5's mean slot), slots 1-9 the 0.1 ... 0.9 quantiles, so QUANTILE_INDEX applies.
SLOT_LEVELS = [0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
assert len(SLOT_LEVELS) == N_OUTPUTS
assert all(SLOT_LEVELS[i] == float(q) for q, i in QUANTILE_INDEX.items())

# What TimesFM3Evaluator.predict_batch sets for a univariate input before it hands the batch to
# TimesFM3Forecaster.predict_batch unchanged. Passed explicitly so they are part of the fingerprint.
FORECAST_FLAGS = {"return_quantiles": True, "use_symmetric_averaging": True,
                  "make_positive": True, "sort_quantiles": True, "use_znorm": False,
                  "padding_mode": "none"}

CACHE_DIR = RESULTS_DIR / "_timesfm3_cache"
MANIFEST_NAME = "timesfm3_manifest.json"
PARTIAL_DIR_NAME = "_timesfm3_partial"


# ────────────────────────────────────────────────────────────── the model

def _jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@dataclass
class TimesFM3Adapter:
    """The official model behind TimesFM 2.5's adapter interface: ``forecast(horizon, contexts)``.

    ``model`` is anything with ``TimesFM3Forecaster``'s ``predict_batch(contexts, horizon, ...)``
    signature yielding outputs with ``.quantiles (H, Q)`` and a ``config.quantiles`` list: the real
    forecaster in a run, a stub in the unit tests. ``fingerprint`` holds everything that
    determines the output and becomes part of every cache key.
    """

    model: Any
    fingerprint: dict

    @classmethod
    def load(cls, *, batch_size: int, device: str = "cpu", model_id: str = MODEL_ID,
             revision: str = MODEL_REVISION) -> "TimesFM3Adapter":
        import torch
        from timesfm3 import TimesFM3Forecaster

        logger.info("Loading %s @ %s on %s", model_id, revision[:10], device)
        model = TimesFM3Forecaster.from_pretrained(model_id, device=device, revision=revision,
                                                   per_core_batch_size=batch_size)
        config = {k: v for k, v in _jsonable(model.config).items()
                  if k not in ("checkpoint_path", "cache_dir", "force_download", "token",
                               "local_files_only", "revision", "device")}
        fingerprint = {
            "model_id": model_id,
            "revision": revision,
            "timesfm": version("timesfm"),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": str(model.device),
            "forecast_flags": FORECAST_FLAGS,
            # after from_pretrained() has synced the config with the checkpoint's own
            "model_config": config,
        }
        return cls(model=model, fingerprint=fingerprint)

    def forecast(self, horizon: int, contexts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Point forecast ``(n, horizon)`` and quantiles ``(n, horizon, 10)`` for ``contexts``,
        in timesfm_forecasting's block layout (see ``SLOT_LEVELS``)."""
        n = len(contexts)
        # One forward pass per block (two queries per series under symmetric averaging). Split into
        # several, a pass holding only gap-stripped series would pad them to a shorter length than
        # the 2,048 h a full context sets, and the forecast would depend on the split.
        batch = getattr(self.model.config, "per_core_batch_size", None)
        if batch is not None and batch < 2 * n:
            raise ValueError(f"per_core_batch_size {batch} < {2 * n}: a block of {n} series must "
                             "go through in one forward pass (see the module docstring)")
        # Fresh float copies: the package interpolates its inputs, and nothing should reach back
        # into the panel.
        inputs = [np.array(c, dtype=np.float32) for c in contexts]
        outs = list(self.model.predict_batch(contexts=inputs, horizon=horizon, **FORECAST_FLAGS))
        if len(outs) != n:
            raise ValueError(f"TimesFM 3.0 returned {len(outs)} outputs for {n} series")
        levels = np.asarray(self.model.config.quantiles, dtype=float)
        q = np.stack([np.asarray(o.quantiles, dtype=float) for o in outs])
        if q.shape != (n, horizon, len(levels)):
            raise ValueError(f"TimesFM 3.0 returned quantiles {q.shape}; expected "
                             f"({n}, {horizon}, {len(levels)})")
        idx = []
        for level in SLOT_LEVELS:
            hit = np.flatnonzero(np.isclose(levels, level, rtol=0, atol=1e-9))
            if len(hit) != 1:
                raise ValueError(f"quantile level {level} is not among TimesFM 3.0's {tuple(levels)}")
            idx.append(int(hit[0]))
        quantiles = q[..., idx]
        return quantiles[..., 0].copy(), quantiles


# ────────────────────────────────────────────────────────────── the grid

def run(fc: TimesFM3Adapter, panel: pd.DataFrame, spec, *, context: int,
        cache: BlockCache | None = None) -> tuple[pd.DataFrame, RunStats]:
    stats = RunStats()
    blocks, t0 = [], time.time()
    total = len(spec.origins) * len(spec.horizons)
    done = 0
    for origin in spec.origins:
        for horizon in spec.horizons:
            block = forecast_one(fc, panel, origin, horizon, target_col=spec.target_col,
                                 context=context, cache=cache, stats=stats)
            block["model"] = MODEL_NAME
            blocks.append(block)
            done += 1
            if done % 12 == 0 or done == total:
                rate = (time.time() - t0) / done
                logger.info("%s  %d/%d  %.2fs/call  cache hits %d  failures %d  eta %.1f min",
                            MODEL_NAME, done, total, rate, stats.cache_hits,
                            len(stats.failures), rate * (total - done) / 60)
    return pd.concat(blocks, ignore_index=True), stats


def leading_gap_audit(panel: pd.DataFrame, spec, context: int) -> dict:
    """Contexts that start in a gap (stripped, then padded back to the batch length), and whether
    every block keeps a full-length context to set that length."""
    from .backtest import history_before

    lead = 0
    every_block_full = True
    for origin in spec.origins:
        hist = history_before(panel, origin, context=context)
        # head(1), not first(): first() skips NaN and would never see a leading gap
        firsts = (hist.sort_values("timestamp").groupby("item_id").head(1)
                  .set_index("item_id")[spec.target_col])
        lens = hist.groupby("item_id").size()
        lead += int(firsts.isna().sum())
        every_block_full &= bool(((lens == lens.max()) & firsts.notna()).any())
    return {"contexts_starting_in_a_gap": lead,
            "every_block_has_a_full_length_context": every_block_full}


# ────────────────────────────────────────────────────────────── driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-origins", type=int, default=None,
                    help="Number of backtest origins (default: the grid default, 24).")
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--context", type=int, default=2048,
                    help="Context hours, truncated at each origin (default 2048, as Chronos-2).")
    ap.add_argument("--batch-size", type=int, default=48,
                    help="Queries per forward pass (each tank is two queries under symmetric "
                         "averaging). Speed only - each series is forecast independently.")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--max-tanks", type=int, default=None)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--no-cache", action="store_true",
                    help="Recompute every block (use for timing runs).")
    ap.add_argument("--revision", default=MODEL_REVISION)
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
    n_tanks = int(panel["item_id"].nunique())
    logger.info("Panel: %d rows, %d tanks", len(panel), n_tanks)
    logger.info("Backtest: %s", spec.describe())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rss_before_load = _rss_mb()
    t_load = time.time()
    fc = TimesFM3Adapter.load(batch_size=args.batch_size, revision=args.revision)
    load_s = time.time() - t_load
    rss_after_load = _rss_mb()
    logger.info("Loaded in %.1fs on %s", load_s, fc.fingerprint["device"])

    cache = None if args.no_cache else BlockCache(Path(args.cache_dir))
    t0 = time.time()
    preds, stats = run(fc, panel, spec, context=args.context, cache=cache)
    elapsed = time.time() - t0
    preds = preds[["item_id", "timestamp", "step", "actual", "pred", *QUANTILE_INDEX,
                   "origin", "horizon", "model"]]

    timing = {
        "wall_clock_s": round(elapsed, 1),
        "load_s": round(load_s, 1),
        "calls": stats.calls,
        "calls_computed": stats.computed,
        "cache_hits": stats.cache_hits,
        "inference_s": round(stats.inference_s, 2),
        "mean_call_s": round(stats.inference_s / stats.computed, 3) if stats.computed else None,
        "series_per_call": n_tanks,
        "per_series_forecast_ms": (round(1000 * stats.inference_s / (stats.computed * n_tanks), 1)
                                   if stats.computed else None),
        "rss_before_load_mb": rss_before_load,
        "rss_after_load_mb": rss_after_load,
        "peak_rss_mb": _peak_rss_mb(),
        "complete_compute": stats.cache_hits == 0,
        "measured_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest = {
        "model": MODEL_NAME,
        "display_name": "TimesFM 3.0 (zero-shot)",
        "model_id": fc.fingerprint["model_id"], "revision": fc.fingerprint["revision"],
        "license": LICENSE,
        "deployable": DEPLOYABLE,
        "packages": {"timesfm": fc.fingerprint["timesfm"], "torch": fc.fingerprint["torch"],
                     "numpy": np.__version__, "pandas": pd.__version__},
        "forecast_flags": FORECAST_FLAGS,
        "model_config": fc.fingerprint["model_config"],
        "regime": "zero-shot - no training, no fine-tuning (none is published for 3.0)",
        "covariates": "none - target only, each tank a separate univariate series",
        "point_forecast": "median (quantile 0.5)",
        "quantile_levels": [float(q) for q in QUANTILE_INDEX],
        "context": args.context, "n_origins": len(spec.origins), "horizons": spec.horizons,
        "origins": [str(o) for o in spec.origins],
        "tanks": sorted(panel["item_id"].unique().tolist()),
        "panel_rows": int(len(panel)),
        "panel_range": [str(panel["timestamp"].min()), str(panel["timestamp"].max())],
        "hardware": _hardware(fc.fingerprint["device"]),
        "timing": timing,
        "leakage_audit": {
            "rule": "context_end <= origin < forecast_start, for every (origin, horizon, tank)",
            "windows_checked": stats.windows_checked,
            "violations": stats.leakage_violations,
        },
        "missing_data": {**missing_data_audit(panel, spec, args.context),
                         **leading_gap_audit(panel, spec, args.context)},
        "failures": len(stats.failures),
        "failed": stats.failures,
        "rows": int(len(preds)),
    }

    if stats.failures:
        # A partial run never becomes the canonical file: it would either break row parity or,
        # scored without --strict, be averaged as though the failed series did not exist.
        part = out_dir / PARTIAL_DIR_NAME
        part.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(part / f"predictions_partial_{MODEL_NAME}.parquet", index=False)
        pd.DataFrame(stats.failures).to_csv(part / f"failures_{MODEL_NAME}.csv", index=False)
        (part / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str))
        logger.error("%d series-forecasts FAILED - canonical outputs left untouched; see %s",
                     len(stats.failures), part)
        sys.exit(1)

    path = out_dir / f"predictions_{MODEL_NAME}.parquet"
    manifest["path"] = str(path)
    manifest_path = out_dir / MANIFEST_NAME
    if not timing["complete_compute"] and manifest_path.exists():
        # A cached rerun measures cache reads, not inference. Keep the timing of the run that
        # actually computed every block, provided it was for the same model and config.
        prev = json.loads(manifest_path.read_text())
        if (prev.get("timing", {}).get("complete_compute")
                and prev.get("forecast_flags") == manifest["forecast_flags"]
                and prev.get("revision") == manifest["revision"]):
            manifest["last_cached_run"] = timing
            manifest["timing"] = prev["timing"]
            manifest["hardware"] = prev.get("hardware", manifest["hardware"])

    preds.to_parquet(path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("%s: %d rows in %.1f min (%d computed, %d cached), 0 failures, "
                "%d windows leakage-checked -> %s", MODEL_NAME, len(preds), elapsed / 60,
                stats.computed, stats.cache_hits, stats.windows_checked, path)
    logger.info("Manifest -> %s", manifest_path)


if __name__ == "__main__":
    main()
