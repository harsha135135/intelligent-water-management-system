"""TimesFM 2.5 zero-shot forecasting on the shared rolling-origin grid.

TimesFM (Das et al., ICML 2024) is Google Research's pretrained decoder-only time-series
foundation model. It is the second foundation model in the study and the most like-for-like
opponent Chronos-2 zero-shot has: pretrained elsewhere, never fitted on this campus, and
conditioned on nothing but each tank's own past.

Checkpoint
----------
``google/timesfm-2.5-200m-pytorch`` (200M backbone + 30M continuous-quantile head, Apache-2.0),
loaded through the official ``timesfm`` package and pinned to one Hugging Face revision. The
package's current release (3.0.x) still ships the 2.5 code path, with a quantile-flip fix that
the 2.0.x release lacks.

TimesFM 3.0 (``google/timesfm-3.0-pytorch``) is newer, but its weights are licensed for
non-commercial, non-production use only. This benchmark exists to choose the model for a campus
system that will run in production, so the checkpoint that could actually be deployed is the one
evaluated.

Protocol - the same as ``Chronos2-ZS``
--------------------------------------
* the same 24 origins, six horizons and 24 tanks, and the same 2,048-hour context truncated at
  each origin;
* the target only, no covariates, so the comparison with ``Chronos2-ZS`` is zero-shot against
  zero-shot;
* point forecast = the model's median, which is also what Chronos-2's ``predictions`` column holds;
* outputs clipped at zero like every other model's (``infer_is_positive`` already guarantees it
  for a non-negative context, so the clip is a no-op kept for symmetry).

Differences that remain, recorded in the manifest rather than hidden
--------------------------------------------------------------------
* **Missing hours.** TimesFM's own input pipeline strips leading NaNs and linearly interpolates
  the rest, holding the last observed value across a gap that runs up to the origin. Chronos-2
  masks gaps instead. Both see only the context: nothing after the origin reaches either model.
* **Device.** The official 2.5 PyTorch code runs on CUDA or CPU, not Apple MPS, so it runs on CPU
  here. Chronos-2's published wall clock was measured on MPS.

Caching
-------
Each (origin, horizon) block's raw model output is cached under
``results/chronos2/_timesfm_cache/``. The key covers the checkpoint revision, package versions,
the effective forecast config, device, origin, horizon, and a hash of every series' id and exact
context values. A cached block is used only if its key matches and it is complete and finite.
Writes are atomic, so an interrupted run cannot leave a half-written entry that later reads as
valid.

    python -m src.models.timesfm_forecasting      # ~3 min on CPU -> predictions_TimesFM2.5-ZS.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestSpec, actuals_after, history_before

logger = logging.getLogger(__name__)

MODEL_NAME = "TimesFM2.5-ZS"
MODEL_ID = "google/timesfm-2.5-200m-pytorch"
# Pinned so a later upload to the same Hugging Face repo cannot silently change the benchmark.
MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
LICENSE = "Apache-2.0"

# TimesFM returns (mean, q10, q20, ..., q90) on its last axis. Its point forecast is index 5,
# the median, the same statistic Chronos-2 reports as its point forecast.
POINT_INDEX = 5
QUANTILE_INDEX = {"0.1": 1, "0.5": 5, "0.9": 9}
N_OUTPUTS = 10

# The settings the official TimesFM 2.5 example uses.
FORECAST_FLAGS = {
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    "infer_is_positive": True,
    "fix_quantile_crossing": True,
}

CACHE_DIR = Path("results/chronos2/_timesfm_cache")
MANIFEST_NAME = "timesfm_manifest.json"
PARTIAL_DIR_NAME = "_timesfm_partial"


# ────────────────────────────────────────────────────────────── the model

@dataclass
class TimesFMForecaster:
    """The official model behind one call: ``forecast(horizon, contexts)``.

    ``model`` is anything with TimesFM's ``forecast(horizon=..., inputs=[...])`` signature: the
    real ``TimesFM_2p5_200M_torch`` in a run, a stub in the unit tests. ``fingerprint`` holds
    everything that determines the output and becomes part of every cache key.
    """

    model: Any
    fingerprint: dict

    @classmethod
    def load(cls, *, context: int, max_horizon: int, batch_size: int,
             model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> "TimesFMForecaster":
        import timesfm
        import torch

        logger.info("Loading %s @ %s", model_id, revision[:10])
        # Eager mode: torch.compile on a CPU build needs a working C++ toolchain, adds a
        # compile step to every run, and would make the benchmark depend on the compiler.
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_id, revision=revision, torch_compile=False)
        model.compile(timesfm.ForecastConfig(
            max_context=context, max_horizon=max_horizon, per_core_batch_size=batch_size,
            **FORECAST_FLAGS))
        fingerprint = {
            "model_id": model_id,
            "revision": revision,
            "timesfm": version("timesfm"),
            "torch": torch.__version__,
            "torch_compile": False,
            "device": str(model.model.device),
            # after compile() has rounded max_context / max_horizon to patch multiples
            "forecast_config": dataclasses.asdict(model.forecast_config),
        }
        return cls(model=model, fingerprint=fingerprint)

    def forecast(self, horizon: int, contexts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Point forecast ``(n, horizon)`` and quantiles ``(n, horizon, 10)`` for ``contexts``."""
        # Fresh float copies: TimesFM pads the list it is handed in place and interpolates the
        # arrays in place, and neither should reach back into the panel.
        inputs = [np.array(c, dtype=float) for c in contexts]
        point, quantiles = self.model.forecast(horizon=horizon, inputs=inputs)
        point, quantiles = np.asarray(point), np.asarray(quantiles)
        n = len(contexts)
        if point.shape != (n, horizon) or quantiles.shape != (n, horizon, N_OUTPUTS):
            raise ValueError(f"TimesFM returned point {point.shape} and quantiles "
                             f"{quantiles.shape}; expected ({n}, {horizon}) and "
                             f"({n}, {horizon}, {N_OUTPUTS})")
        return point, quantiles


# ────────────────────────────────────────────────────────────── cache

def cache_key(fingerprint: dict, origin: pd.Timestamp, horizon: int, context: int,
              series: list[tuple[str, np.ndarray]]) -> str:
    """Deterministic key for one (origin, horizon) block. Any change to the checkpoint, the
    package, the config, the grid or a single context value produces a different key."""
    h = hashlib.sha256()
    h.update(json.dumps({"fingerprint": fingerprint, "origin": str(pd.Timestamp(origin)),
                         "horizon": int(horizon), "context": int(context)},
                        sort_keys=True, default=str).encode())
    for item_id, values in series:
        h.update(str(item_id).encode() + b"\x00")
        # NaN has more than one bit pattern; hash a canonical one so equal contexts collide.
        canon = np.where(np.isnan(values), np.nan, np.asarray(values, dtype=np.float64))
        h.update(np.ascontiguousarray(canon).tobytes())
        h.update(b"\x01")
    return h.hexdigest()


class BlockCache:
    """Raw TimesFM output per (origin, horizon) block, one ``.npz`` per key.

    An entry is returned only if the file loads, its stored key matches, its shapes match what
    the caller expects and every value is finite, so a truncated, stale or corrupt entry is
    recomputed rather than trusted.
    """

    def __init__(self, root: Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str, n_series: int, horizon: int):
        if not self.root:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) != key:
                    return None
                point, quantiles = z["point"], z["quantiles"]
                compute_s = float(z["compute_s"])
        except Exception as exc:                            # corrupt entry: recompute it
            logger.warning("ignoring unreadable cache entry %s (%s)", path.name, exc)
            return None
        if (point.shape != (n_series, horizon)
                or quantiles.shape != (n_series, horizon, N_OUTPUTS)
                or not (np.isfinite(point).all() and np.isfinite(quantiles).all())):
            return None
        return point, quantiles, compute_s

    def put(self, key: str, point: np.ndarray, quantiles: np.ndarray, compute_s: float) -> None:
        if not self.root:
            return
        path = self._path(key)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:                          # a handle, so no suffix is appended
            np.savez(f, key=np.array(key), point=point, quantiles=quantiles,
                     compute_s=np.array(compute_s))
        os.replace(tmp, path)                               # atomic: never a half-written entry


# ────────────────────────────────────────────────────────────── one forecast

@dataclass
class RunStats:
    calls: int = 0
    computed: int = 0
    cache_hits: int = 0
    inference_s: float = 0.0          # measured model time of the blocks computed in this run
    cached_compute_s: float = 0.0     # model time originally measured for the cache hits
    windows_checked: int = 0
    leakage_violations: int = 0
    failures: list[dict] = field(default_factory=list)


def _contexts(hist: pd.DataFrame, items: list[str], target_col: str):
    by_item = {i: g for i, g in hist.groupby("item_id", sort=False)}
    out = []
    for item in items:
        g = by_item.get(item)
        if g is None or g.empty:
            out.append((item, None, None))
            continue
        g = g.sort_values("timestamp")
        out.append((item, g[target_col].to_numpy(dtype=float), g["timestamp"].iloc[-1]))
    return out


def forecast_one(
    fc: TimesFMForecaster,
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    *,
    target_col: str,
    context: int,
    cache: BlockCache | None = None,
    stats: RunStats | None = None,
) -> pd.DataFrame:
    """One (origin, horizon) forecast for every tank, in the benchmark's row schema.

    Mirrors ``chronos2_forecasting.forecast_one``: history truncated at the origin, the scored
    window taken from ``actuals_after``, predictions merged onto it on (tank, timestamp).
    A series TimesFM fails on is recorded in ``stats.failures`` and left with a missing
    prediction, never dropped and never filled.
    """
    stats = stats if stats is not None else RunStats()
    hist = history_before(panel, origin, context=context)
    fut = actuals_after(panel, origin, horizon)
    items = sorted(fut["item_id"].unique())
    first_fut = fut.groupby("item_id", sort=False)["timestamp"].min()

    series, ok_items = [], []
    for item, values, context_end in _contexts(hist, items, target_col):
        if values is None:
            stats.failures.append({"item_id": item, "origin": str(origin), "horizon": horizon,
                                   "error": "NoContext",
                                   "message": "no history at or before the origin"})
            continue
        # The leakage rule, checked for every window rather than assumed: the context ends at or
        # before the origin, and the first scored hour lies strictly after both.
        stats.windows_checked += 1
        if not (context_end <= origin < first_fut[item]) or len(values) > context:
            stats.leakage_violations += 1
            raise AssertionError(
                f"leakage: {item} context ends {context_end}, origin {origin}, forecast starts "
                f"{first_fut[item]}, context length {len(values)} > {context}")
        series.append((item, values))
        ok_items.append((item, context_end))

    point = quantiles = None
    if series:
        key = cache_key(fc.fingerprint, origin, horizon, context, series)
        hit = cache.get(key, len(series), horizon) if cache else None
        stats.calls += 1
        if hit is not None:
            point, quantiles, compute_s = hit
            stats.cache_hits += 1
            stats.cached_compute_s += compute_s
        else:
            t0 = time.perf_counter()
            try:
                point, quantiles = fc.forecast(horizon, [v for _, v in series])
                bad = ~(np.isfinite(point).all(axis=1) & np.isfinite(quantiles).all(axis=(1, 2)))
            except Exception as exc:                        # isolate the series that fails
                logger.warning("block origin=%s h=%d failed (%s: %s); retrying per series",
                               origin, horizon, type(exc).__name__, exc)
                point = np.full((len(series), horizon), np.nan, dtype=np.float32)
                quantiles = np.full((len(series), horizon, N_OUTPUTS), np.nan, dtype=np.float32)
                bad = np.zeros(len(series), dtype=bool)
                for k, (item, values) in enumerate(series):
                    try:
                        p, q = fc.forecast(horizon, [values])
                        point[k], quantiles[k] = p[0], q[0]
                        bad[k] = not (np.isfinite(p).all() and np.isfinite(q).all())
                    except Exception as exc_one:
                        bad[k] = False
                        stats.failures.append({
                            "item_id": item, "origin": str(origin), "horizon": horizon,
                            "error": type(exc_one).__name__, "message": str(exc_one)[:500]})
            compute_s = time.perf_counter() - t0
            stats.computed += 1
            stats.inference_s += compute_s
            for k in np.flatnonzero(bad):
                stats.failures.append({"item_id": series[k][0], "origin": str(origin),
                                       "horizon": horizon, "error": "NonFiniteOutput",
                                       "message": "TimesFM returned NaN or inf"})
                point[k], quantiles[k] = np.nan, np.nan
            failed_here = {f["item_id"] for f in stats.failures
                           if f["origin"] == str(origin) and f["horizon"] == horizon}
            if cache and not failed_here:                   # only complete blocks are cached
                cache.put(key, point, quantiles, compute_s)

    blocks = []
    for k, (item, context_end) in enumerate(ok_items):
        ts = pd.date_range(context_end + pd.Timedelta(hours=1), periods=horizon, freq="h")
        block = pd.DataFrame({"item_id": item, "timestamp": ts,
                              # Outflow is physically non-negative; a foundation model has no
                              # such prior (same clip as every other model in the benchmark).
                              "pred": np.clip(point[k], 0.0, None)})
        for col, idx in QUANTILE_INDEX.items():
            block[col] = np.clip(quantiles[k, :, idx], 0.0, None)
        blocks.append(block)
    cols = ["item_id", "timestamp", "pred", *QUANTILE_INDEX]
    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=cols)

    merged = fut[["item_id", "timestamp", "step", target_col]].merge(
        out, on=["item_id", "timestamp"], how="left")
    merged = merged.rename(columns={target_col: "actual"})
    merged["origin"] = origin
    merged["horizon"] = horizon

    # Every scored row must have a forecast unless its series is on the failure list.
    failed = {f["item_id"] for f in stats.failures
              if f["origin"] == str(origin) and f["horizon"] == horizon}
    unexplained = merged["pred"].isna() & ~merged["item_id"].isin(failed)
    if unexplained.any():
        raise AssertionError(f"{int(unexplained.sum())} rows at origin={origin} h={horizon} "
                             "have no forecast and no recorded failure - timestamps misaligned")
    return merged


# ────────────────────────────────────────────────────────────── the grid

def run(fc: TimesFMForecaster, panel: pd.DataFrame, spec: BacktestSpec, *, context: int,
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


def missing_data_audit(panel: pd.DataFrame, spec: BacktestSpec, context: int) -> dict:
    """How much of each context TimesFM's own preprocessing has to fill, over the unique
    (origin, tank) contexts. Descriptive only: the model receives the raw context with NaNs."""
    n = gaps = trailing = 0
    max_missing = max_trailing = 0
    for origin in spec.origins:
        hist = history_before(panel, origin, context=context)
        for _, g in hist.groupby("item_id", sort=False):
            y = g.sort_values("timestamp")[spec.target_col].to_numpy(dtype=float)
            miss = np.isnan(y)
            run_end = int(np.argmax(~miss[::-1])) if (~miss).any() else len(y)
            n += 1
            gaps += bool(miss.any())
            trailing += run_end > 0
            max_missing = max(max_missing, int(miss.sum()))
            max_trailing = max(max_trailing, run_end)
    return {
        "unique_contexts": n,
        "contexts_with_missing_hours": gaps,
        "contexts_ending_in_a_gap": trailing,
        "max_missing_hours_in_one_context": max_missing,
        "max_trailing_missing_hours": max_trailing,
        "handling": "raw context passed with NaNs; TimesFM's own pipeline strips leading NaNs "
                    "and linearly interpolates the rest within the context (a gap that runs up "
                    "to the origin is held at the last observed value). Chronos-2 masks gaps "
                    "instead. No value after the origin is used by either.",
    }


def _hardware(device: str) -> dict:
    import torch

    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        ram_gb = None
    return {"device": device, "cpu": cpu, "machine": platform.machine(),
            "os": platform.platform(), "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "ram_gb": ram_gb,
            "python": platform.python_version()}


def _rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024 ** 2, 1)
    except ImportError:
        return None


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 1024 ** 2 if sys.platform == "darwin" else peak / 1024, 1)  # bytes vs KB


# ────────────────────────────────────────────────────────────── driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-origins", type=int, default=None,
                    help="Number of backtest origins (default: the grid default, 24).")
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--context", type=int, default=2048,
                    help="Context hours, truncated at each origin (default 2048, as Chronos-2).")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Series per forward pass; capped at the tank count. Speed only - "
                         "each series is forecast independently of the others in its batch.")
    ap.add_argument("--out-dir", default="results/chronos2")
    ap.add_argument("--max-tanks", type=int, default=None)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--no-cache", action="store_true",
                    help="Recompute every block (use for timing runs).")
    ap.add_argument("--model-id", default=MODEL_ID)
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
    fc = TimesFMForecaster.load(context=args.context, max_horizon=spec.max_horizon,
                                batch_size=min(args.batch_size, n_tanks),
                                model_id=args.model_id, revision=args.revision)
    load_s = time.time() - t_load
    rss_after_load = _rss_mb()
    logger.info("Loaded in %.1fs on %s; effective config %s", load_s,
                fc.fingerprint["device"], fc.fingerprint["forecast_config"])

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
        "display_name": "TimesFM 2.5 (zero-shot)",
        "model_id": args.model_id, "revision": args.revision, "license": LICENSE,
        "packages": {"timesfm": version("timesfm"), "torch": fc.fingerprint["torch"],
                     "numpy": np.__version__, "pandas": pd.__version__},
        "forecast_config": fc.fingerprint["forecast_config"],
        "torch_compile": False,
        "regime": "zero-shot - no training, no fine-tuning",
        "covariates": "none - target only",
        "point_forecast": "median (quantile 0.5), TimesFM's own point output",
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
        "missing_data": missing_data_audit(panel, spec, args.context),
        "failures": len(stats.failures),
        "failed": stats.failures,
        "rows": int(len(preds)),
    }

    if stats.failures:
        # A partial run never becomes the canonical file: it would either break row parity or,
        # scored without --strict, be averaged as though the failed series did not exist.
        part = out_dir / PARTIAL_DIR_NAME
        part.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(part / "predictions_partial.parquet", index=False)
        pd.DataFrame(stats.failures).to_csv(part / "failures.csv", index=False)
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
                and prev.get("forecast_config") == manifest["forecast_config"]
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
