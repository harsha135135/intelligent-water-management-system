"""Mistral Timestral zero-shot forecasting on the shared rolling-origin grid.

Timestral (Mistral AI, October 2026) is a family of pretrained time-series foundation models in
four sizes - 5M, 28M, 123M and 209M parameters - sharing one architecture and training recipe. It
is the third foundation model in the study and, like ``TimesFM2.5-ZS``, a like-for-like opponent
for ``Chronos2-ZS``: pretrained elsewhere, never fitted on this campus, and conditioned on nothing
but each tank's own past.

Checkpoints
-----------
``mistralai/Timestral-{5M,28M,123M,209M}``, loaded through Mistral's ``timestral`` package and
each pinned to one Hugging Face revision. The code is Apache-2.0. The weights are under a
"Modified MIT" licence: free to use, including in production, except that a licensee whose global
consolidated monthly revenue exceeds USD 20M has no rights under it. Both are recorded in the
manifest.

Only ``Timestral-209M-ZS`` is on the benchmark roster (``unified_analysis.ORDER``). The smaller
sizes write their own predictions files, which scoring skips as off-roster; ``--sizes-report``
scores all four on the same rows into ``timestral_sizes.csv``.

Protocol - the same as ``Chronos2-ZS`` and ``TimesFM2.5-ZS``
------------------------------------------------------------
* the same 24 origins, six horizons and 24 tanks, and the same 2,048-hour context truncated at
  each origin; one model call per (origin, horizon) block, every tank in the batch;
* the target only, no covariates;
* point forecast = the model's median;
* the package's default inference settings (symmetric averaging and its own non-negative clip on
  for a non-negative context), then the benchmark's usual clip at zero;
* CPU, the same machine and thread count TimesFM's timing was measured with.

The model and data plumbing - the leakage assertion, the merge onto the scored window, failure
isolation and the block cache - is ``timesfm_forecasting``'s, imported unchanged. Its block layout
is TimesFM's ``(point, 10 outputs)``, so the adapter below hands Timestral's quantiles over in that
layout: slot 0 the median, slots 1-9 the 0.1 ... 0.9 quantiles, picked out by level value.

Missing hours: Timestral takes NaN as missing, masks it and forward-fills it inside the model;
the adapter passes the raw context with its NaNs and fills nothing.

    python -m src.models.timestral_forecasting                # 209M, ~3 min on CPU
    python -m src.models.timestral_forecasting --size 5M      # off-roster sizes
    python -m src.models.timestral_forecasting --sizes-report # -> timestral_sizes.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .timesfm_forecasting import (
    N_OUTPUTS, QUANTILE_INDEX, BlockCache, RunStats, _hardware, _peak_rss_mb, _rss_mb,
    forecast_one, missing_data_audit,
)

logger = logging.getLogger(__name__)

SIZES = ["5M", "28M", "123M", "209M"]
ROSTER_SIZE = "209M"
# Pinned so a later upload to the same Hugging Face repo cannot silently change the benchmark.
REVISIONS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}
# The github.com/mistralai/timestral commit requirements.txt installs.
PACKAGE_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
LICENSE = {
    "code": "Apache-2.0",
    "weights": "Modified MIT - no rights for a licensee whose global consolidated monthly "
               "revenue exceeds USD 20M; attribution notice required",
}

# The levels handed to timesfm_forecasting's block layout, in its slot order: slot 0 is the
# median (TimesFM's mean slot), slots 1-9 the 0.1 ... 0.9 quantiles, so QUANTILE_INDEX applies.
SLOT_LEVELS = [0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
assert len(SLOT_LEVELS) == N_OUTPUTS
assert all(SLOT_LEVELS[i] == float(q) for q, i in QUANTILE_INDEX.items())

# timestral's own defaults, passed explicitly so they are part of the fingerprint.
FORECAST_FLAGS = {"symmetric_averaging": True, "clip_nonnegative": True}

CACHE_DIR = Path("results/chronos2/_timestral_cache")
PARTIAL_DIR_NAME = "_timestral_partial"
SIZES_CSV = "timestral_sizes.csv"


def model_name(size: str) -> str:
    return f"Timestral-{size}-ZS"


def manifest_name(size: str) -> str:
    return f"timestral_{size.lower()}_manifest.json"


MODEL_NAME = model_name(ROSTER_SIZE)
MANIFEST_NAME = manifest_name(ROSTER_SIZE)


# ────────────────────────────────────────────────────────────── the model

def _installed_commit() -> str | None:
    """The git commit pip recorded when ``timestral`` was installed from GitHub."""
    try:
        info = json.loads(distribution("timestral").read_text("direct_url.json") or "{}")
    except Exception:
        return None
    return info.get("vcs_info", {}).get("commit_id")


@dataclass
class TimestralForecaster:
    """The official model behind TimesFM's adapter interface: ``forecast(horizon, contexts)``.

    ``model`` is anything with timestral's ``forecast(context, prediction_length=..., ...)``
    signature returning ``.quantiles (B, V, H, Q)`` and ``.quantile_levels``: the real
    ``timestral.TimestralForecaster`` in a run, a stub in the unit tests. ``fingerprint`` holds
    everything that determines the output and becomes part of every cache key.
    """

    model: Any
    fingerprint: dict

    @classmethod
    def load(cls, *, size: str = ROSTER_SIZE, revision: str | None = None,
             device: str = "cpu") -> "TimestralForecaster":
        import timestral
        import torch

        model_id = f"mistralai/Timestral-{size}"
        revision = revision or REVISIONS[size]
        commit = _installed_commit()
        if commit != PACKAGE_COMMIT:
            logger.warning("timestral installed from %s, not the pinned %s", commit, PACKAGE_COMMIT)
        logger.info("Loading %s @ %s on %s", model_id, revision[:10], device)
        model = timestral.TimestralForecaster.from_pretrained(model_id, revision=revision,
                                                              device=device)
        c = model.config
        fingerprint = {
            "model_id": model_id,
            "revision": revision,
            "timestral": version("timestral"),
            "timestral_commit": commit,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": str(model.device),
            "forecast_config": {
                **FORECAST_FLAGS,
                "max_context": model.max_context,
                "per_shot_horizon_patches": model.per_shot_horizon_patches,
                "quantile_real_cap_k": model.quantile_real_cap_k,
                "patch_size": c.patch_size,
                "quantile_levels": list(model.quantile_levels),
            },
        }
        return cls(model=model, fingerprint=fingerprint)

    def forecast(self, horizon: int, contexts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Point forecast ``(n, horizon)`` and quantiles ``(n, horizon, 10)`` for ``contexts``,
        in timesfm_forecasting's block layout (see ``SLOT_LEVELS``)."""
        n = len(contexts)
        # One dense (n, T) array, fresh float copies left-padded with NaN: timestral reads NaN as
        # unobserved, so padding a shorter series is the same as not having those hours.
        width = max(len(c) for c in contexts)
        arr = np.full((n, width), np.nan, dtype=np.float64)
        for k, c in enumerate(contexts):
            if len(c):
                arr[k, width - len(c):] = np.asarray(c, dtype=np.float64)
        out = self.model.forecast(arr, prediction_length=horizon, **FORECAST_FLAGS)
        q = out.quantiles
        q = (q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q))
        levels = np.asarray(out.quantile_levels, dtype=float)
        if q.shape != (n, 1, horizon, len(levels)):
            raise ValueError(f"Timestral returned quantiles {q.shape}; expected "
                             f"({n}, 1, {horizon}, {len(levels)})")
        idx = []
        for level in SLOT_LEVELS:
            hit = np.flatnonzero(np.isclose(levels, level, rtol=0, atol=1e-9))
            if len(hit) != 1:
                raise ValueError(f"quantile level {level} is not among Timestral's {tuple(levels)}")
            idx.append(int(hit[0]))
        quantiles = q[:, 0][..., idx]
        return quantiles[..., 0].copy(), quantiles


# ────────────────────────────────────────────────────────────── the grid

def run(fc: TimestralForecaster, panel: pd.DataFrame, spec, *, context: int, name: str,
        cache: BlockCache | None = None) -> tuple[pd.DataFrame, RunStats]:
    stats = RunStats()
    blocks, t0 = [], time.time()
    total = len(spec.origins) * len(spec.horizons)
    done = 0
    for origin in spec.origins:
        for horizon in spec.horizons:
            block = forecast_one(fc, panel, origin, horizon, target_col=spec.target_col,
                                 context=context, cache=cache, stats=stats)
            block["model"] = name
            blocks.append(block)
            done += 1
            if done % 12 == 0 or done == total:
                rate = (time.time() - t0) / done
                logger.info("%s  %d/%d  %.2fs/call  cache hits %d  failures %d  eta %.1f min",
                            name, done, total, rate, stats.cache_hits,
                            len(stats.failures), rate * (total - done) / 60)
    return pd.concat(blocks, ignore_index=True), stats


# ────────────────────────────────────────────────────────────── sizes report

def sizes_report(out_dir: Path) -> pd.DataFrame:
    """Every Timestral size scored on the same rows, by the scorer's own functions."""
    from ..data.curate import load_curated_hourly
    from .metrics import aggregate_metrics, attach_scales, coverage, per_tank_metrics
    from .score_benchmark import interval_width

    frames, wall = [], {}
    for size in SIZES:
        path = out_dir / f"predictions_{model_name(size)}.parquet"
        if not path.exists():
            logger.warning("no %s - size %s left out", path.name, size)
            continue
        frames.append(pd.read_parquet(path))
        man = out_dir / manifest_name(size)
        if man.exists():
            timing = json.loads(man.read_text()).get("timing", {})
            wall[model_name(size)] = (timing.get("wall_clock_s")
                                      if timing.get("complete_compute") else None)
    if not frames:
        raise FileNotFoundError(f"no Timestral predictions under {out_dir}")
    preds = pd.concat(frames, ignore_index=True)
    rows = preds.groupby("model").size()
    if rows.nunique() != 1:
        raise AssertionError(f"sizes were forecast on different row counts:\n{rows}")

    scored = attach_scales(preds, load_curated_hourly(with_features=False))
    agg = aggregate_metrics(per_tank_metrics(scored))
    agg = agg.merge(coverage(scored), on=["model", "horizon"], how="left")
    agg = agg.merge(interval_width(scored), on=["model", "horizon"], how="left")
    agg.insert(1, "size", agg["model"].str.extract(r"Timestral-(\w+)-ZS", expand=False))
    agg["wall_clock_s"] = agg["model"].map(wall)
    order = {model_name(s): i for i, s in enumerate(SIZES)}
    agg = agg.sort_values(["horizon", "model"], key=lambda s: s.map(order) if s.name == "model"
                          else s).reset_index(drop=True)
    path = out_dir / SIZES_CSV
    agg.to_csv(path, index=False)
    mase = agg.pivot(index="size", columns="horizon", values="macro_mase")
    logger.info("Wrote %s - macro MASE:\n%s", path,
                mase.reindex([s for s in SIZES if s in mase.index]).round(4).to_string())
    return agg


# ────────────────────────────────────────────────────────────── driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=SIZES, default=ROSTER_SIZE)
    ap.add_argument("--n-origins", type=int, default=None,
                    help="Number of backtest origins (default: the grid default, 24).")
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--context", type=int, default=2048,
                    help="Context hours, truncated at each origin (default 2048, as Chronos-2).")
    ap.add_argument("--out-dir", default="results/chronos2")
    ap.add_argument("--max-tanks", type=int, default=None)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--no-cache", action="store_true",
                    help="Recompute every block (use for timing runs).")
    ap.add_argument("--revision", default=None, help="Default: the pinned revision for --size.")
    ap.add_argument("--sizes-report", action="store_true",
                    help=f"Score every size's predictions into {SIZES_CSV} and exit.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    if args.sizes_report:
        sizes_report(out_dir)
        return

    from ..data.curate import load_curated_hourly
    from .backtest import make_spec

    name = model_name(args.size)
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

    out_dir.mkdir(parents=True, exist_ok=True)

    rss_before_load = _rss_mb()
    t_load = time.time()
    fc = TimestralForecaster.load(size=args.size, revision=args.revision)
    load_s = time.time() - t_load
    rss_after_load = _rss_mb()
    logger.info("Loaded in %.1fs on %s; config %s", load_s, fc.fingerprint["device"],
                fc.fingerprint["forecast_config"])

    cache = None if args.no_cache else BlockCache(Path(args.cache_dir))
    t0 = time.time()
    preds, stats = run(fc, panel, spec, context=args.context, name=name, cache=cache)
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
    missing = missing_data_audit(panel, spec, args.context)
    missing["handling"] = (
        "raw context passed with NaNs; Timestral masks a missing hour as unobserved and "
        "forward-fills its input value inside the model (leading gaps become unobserved zeros). "
        "The adapter fills nothing. No value after the origin is used.")
    manifest = {
        "model": name,
        "display_name": f"Timestral {args.size} (zero-shot)",
        "on_benchmark_roster": args.size == ROSTER_SIZE,
        "model_id": fc.fingerprint["model_id"], "revision": fc.fingerprint["revision"],
        "license": LICENSE,
        "packages": {"timestral": fc.fingerprint["timestral"],
                     "timestral_commit": fc.fingerprint["timestral_commit"],
                     "torch": fc.fingerprint["torch"],
                     "numpy": np.__version__, "pandas": pd.__version__},
        "forecast_config": fc.fingerprint["forecast_config"],
        "regime": "zero-shot - no training, no fine-tuning",
        "covariates": "none - target only",
        "point_forecast": "median (quantile 0.5), Timestral's own median",
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
        "missing_data": missing,
        "failures": len(stats.failures),
        "failed": stats.failures,
        "rows": int(len(preds)),
    }

    mname = manifest_name(args.size)
    if stats.failures:
        # A partial run never becomes the canonical file: it would either break row parity or,
        # scored without --strict, be averaged as though the failed series did not exist.
        part = out_dir / PARTIAL_DIR_NAME
        part.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(part / f"predictions_partial_{name}.parquet", index=False)
        pd.DataFrame(stats.failures).to_csv(part / f"failures_{name}.csv", index=False)
        (part / mname).write_text(json.dumps(manifest, indent=2, default=str))
        logger.error("%d series-forecasts FAILED - canonical outputs left untouched; see %s",
                     len(stats.failures), part)
        sys.exit(1)

    path = out_dir / f"predictions_{name}.parquet"
    manifest["path"] = str(path)
    manifest_path = out_dir / mname
    if not timing["complete_compute"] and manifest_path.exists():
        # A cached rerun measures cache reads, not inference. Keep the timing of the run that
        # actually computed every block, provided it was for the same model and config.
        prev = json.loads(manifest_path.read_text())
        if (prev.get("timing", {}).get("complete_compute")
                and prev.get("forecast_config") == manifest["forecast_config"]
                and prev.get("revision") == manifest["revision"]):
            manifest["last_cached_run"] = timing
            manifest["timing"] = prev["timing"]
            manifest["hardware"] = prev.get("hardware", manifest["hardware"])

    preds.to_parquet(path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("%s: %d rows in %.1f min (%d computed, %d cached), 0 failures, "
                "%d windows leakage-checked -> %s", name, len(preds), elapsed / 60,
                stats.computed, stats.cache_hits, stats.windows_checked, path)
    logger.info("Manifest -> %s", manifest_path)


if __name__ == "__main__":
    main()
