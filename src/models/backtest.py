"""Rolling-origin backtest harness shared by every model.

The single most important property here is that **every model is scored on identical origins and
identical rows**. The pre-existing results in this repo could not be compared to each other
(``results/autogluon`` used 26 series / 624 rows, ``results/patchtst`` used 24 series / 576 rows,
over different holdout dates), which is exactly the trap this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET_COL = "Outflow in KL"
HORIZONS = [6, 12, 24, 48, 72, 168]

KNOWN_COVARIATES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "is_holiday", "is_isa", "is_esa", "is_summer", "exam_proximity",
]
PAST_COVARIATES = ["Inflow in KL", "Opening Value in KL", "Closing Value in KL"]


@dataclass
class BacktestSpec:
    """Defines the evaluation grid. Constructed once and passed to every model."""
    origins: list[pd.Timestamp]
    horizons: list[int] = field(default_factory=lambda: list(HORIZONS))
    target_col: str = TARGET_COL

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def describe(self) -> str:
        return (
            f"{len(self.origins)} origins, {self.origins[0]} .. {self.origins[-1]}, "
            f"horizons={self.horizons}"
        )


def make_spec(
    panel: pd.DataFrame,
    *,
    n_origins: int = 24,
    origin_stride_hours: int = 23,
    horizons: list[int] | None = None,
) -> BacktestSpec:
    """Build the evaluation grid from the tail of the panel.

    Origins are spaced ``origin_stride_hours`` apart and placed so that **every** origin has a
    full ``max_horizon`` of actuals after it.

    The stride is deliberately **23 hours, not 24**. With a 24-hour stride every origin lands on
    the same clock hour, so the short horizons are only ever scored on one slice of the diurnal
    cycle: a 23:00 origin means h=6 always evaluates 00:00-05:00, the quiet overnight window,
    and never the morning refill peak that is actually hard to forecast. That inflates
    short-horizon scores. 23 is co-prime with 24, so the default 24 origins visit each
    hour-of-day exactly once and every horizon is scored across the whole diurnal cycle.
    """
    horizons = list(horizons or HORIZONS)
    max_h = max(horizons)
    end = panel["timestamp"].max()
    last_origin = end - pd.Timedelta(hours=max_h)
    origins = [
        last_origin - pd.Timedelta(hours=origin_stride_hours * k)
        for k in range(n_origins)
    ][::-1]
    return BacktestSpec(origins=origins, horizons=horizons)


def history_before(panel: pd.DataFrame, origin: pd.Timestamp, context: int | None) -> pd.DataFrame:
    """Rows strictly at or before the origin, optionally truncated to the last ``context`` hours."""
    hist = panel[panel["timestamp"] <= origin]
    if context is not None:
        hist = hist.groupby("item_id", sort=False).tail(context)
    return hist


def actuals_after(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """The ground-truth window a forecast from ``origin`` is scored against."""
    fut = panel[panel["timestamp"] > origin]
    fut = fut.groupby("item_id", sort=False).head(horizon).copy()
    fut["step"] = fut.groupby("item_id", sort=False).cumcount() + 1
    return fut


def assert_comparable(scored: pd.DataFrame) -> pd.DataFrame:
    """Every model must have evaluated the same rows at each horizon. Fail loudly if not."""
    counts = (
        scored.dropna(subset=["actual", "pred"])
        .groupby(["horizon", "model"]).size().unstack("model")
    )
    bad = counts[counts.nunique(axis=1) > 1]
    if len(bad):
        raise AssertionError(
            "Models were scored on different row counts - results are not comparable:\n"
            f"{bad.to_string()}"
        )
    return counts


def seasonal_naive_forecast(
    panel: pd.DataFrame, spec: BacktestSpec, *, season: int = 24, name: str | None = None
) -> pd.DataFrame:
    """Repeat the last ``season`` observed hours forward. This is the MASE/RMSSE reference."""
    model = name or f"SeasonalNaive-{season}"
    out = []
    for origin in spec.origins:
        hist = history_before(panel, origin, context=season * 4)
        last = {
            item_id: g.sort_values("timestamp")[spec.target_col].to_numpy()[-season:]
            for item_id, g in hist.groupby("item_id", sort=False)
        }
        for horizon in spec.horizons:
            fut = actuals_after(panel, origin, horizon)
            preds = []
            for item_id, g in fut.groupby("item_id", sort=False):
                window = last[item_id]
                # Tile the last season forward; nan-fill from the series mean if the window is
                # itself missing, so a sensor gap does not silently drop evaluation rows.
                if np.all(np.isnan(window)):
                    window = np.zeros(season)
                window = np.where(np.isnan(window), np.nanmean(window), window)
                reps = int(np.ceil(len(g) / season))
                preds.append(np.tile(window, reps)[: len(g)])
            block = fut[["item_id", "timestamp", "step"]].copy()
            block["pred"] = np.concatenate(preds) if preds else []
            block["actual"] = fut[spec.target_col].to_numpy()
            block["model"] = model
            block["origin"] = origin
            block["horizon"] = horizon
            out.append(block)
    return pd.concat(out, ignore_index=True)


def climatology_forecast(panel: pd.DataFrame, spec: BacktestSpec, *, lookback_days: int = 28) -> pd.DataFrame:
    """Mean of the same hour-of-week over the trailing window. A stronger naive than
    seasonal-naive on noisy series, and cheap enough to always include."""
    out = []
    for origin in spec.origins:
        hist = history_before(panel, origin, context=24 * lookback_days).copy()
        hist["how"] = hist["timestamp"].dt.dayofweek * 24 + hist["timestamp"].dt.hour
        profile = hist.groupby(["item_id", "how"])[spec.target_col].mean()
        overall = hist.groupby("item_id")[spec.target_col].mean()
        for horizon in spec.horizons:
            fut = actuals_after(panel, origin, horizon).copy()
            fut["how"] = fut["timestamp"].dt.dayofweek * 24 + fut["timestamp"].dt.hour
            idx = pd.MultiIndex.from_arrays([fut["item_id"], fut["how"]])
            pred = profile.reindex(idx).to_numpy()
            fallback = overall.reindex(fut["item_id"]).to_numpy()
            pred = np.where(np.isnan(pred), fallback, pred)
            block = fut[["item_id", "timestamp", "step"]].copy()
            block["pred"] = np.nan_to_num(pred)
            block["actual"] = fut[spec.target_col].to_numpy()
            block["model"] = "Climatology-HoW"
            block["origin"] = origin
            block["horizon"] = horizon
            out.append(block)
    return pd.concat(out, ignore_index=True)
