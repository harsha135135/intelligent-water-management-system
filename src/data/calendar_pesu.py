"""PESU academic calendar features.

The date lists here are lifted verbatim from ``eda/eda_deep.py`` (sections 9a-9f), where they
were validated against the anomaly register in ``PESU_WALTR_Anomaly_Validation.xlsx``. Keeping
them in one importable module means the EDA and the forecasting covariates cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAYS = pd.to_datetime([
    "2025-01-15", "2025-01-26", "2025-03-14", "2025-04-14", "2025-05-01",
    "2025-08-15", "2025-08-27", "2025-09-05", "2025-10-01", "2025-10-02",
    "2025-10-20", "2025-10-22", "2025-11-01", "2025-12-25",
    "2026-01-15", "2026-01-26", "2026-03-19", "2026-03-21", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-05-28",
])

ISA_DATES = pd.DatetimeIndex([]).append([
    pd.date_range("2025-03-03", "2025-03-08"), pd.date_range("2025-04-21", "2025-04-26"),
    pd.date_range("2025-09-22", "2025-09-26"), pd.date_range("2025-11-24", "2025-11-28"),
    pd.date_range("2026-03-02", "2026-03-07"), pd.date_range("2026-04-27", "2026-05-01"),
])

ESA_DATES = pd.DatetimeIndex([]).append([
    pd.date_range("2025-05-04", "2025-05-30"),
    pd.date_range("2025-12-01", "2025-12-30"),
    pd.date_range("2026-05-04", "2026-05-30"),
])

SUMMER_BREAK = pd.date_range("2025-06-01", "2025-08-03")
INTERSEM_BREAK = pd.date_range("2025-12-31", "2026-01-04")

EXAM_STARTS = sorted(pd.Timestamp(x) for x in [
    "2025-03-03", "2025-04-21", "2025-05-04", "2025-09-22",
    "2025-11-24", "2025-12-01", "2026-03-02", "2026-04-27", "2026-05-04",
])

# Precedence order matters: a holiday inside an ESA window is labelled "esa", matching eda_deep.py.
PHASES = ["regular", "holiday", "isa", "esa", "summer_break", "intersem_break"]


def _phase_for(day: pd.Timestamp) -> str:
    if day in SUMMER_BREAK:
        return "summer_break"
    if day in INTERSEM_BREAK:
        return "intersem_break"
    if day in ESA_DATES:
        return "esa"
    if day in ISA_DATES:
        return "isa"
    if day in HOLIDAYS:
        return "holiday"
    return "regular"


def phase_series(days: pd.DatetimeIndex) -> pd.Series:
    """Mutually-exclusive academic phase label, one per calendar day."""
    unique = pd.DatetimeIndex(days).normalize().unique()
    lookup = {d: _phase_for(d) for d in unique}
    return pd.Series(pd.DatetimeIndex(days).normalize().map(lookup), index=None)


def days_to_next_exam(days: pd.DatetimeIndex) -> np.ndarray:
    """Forward distance in days to the next exam-block start; 999 when none remain."""
    normalized = pd.DatetimeIndex(days).normalize()
    starts = np.array([e.value for e in EXAM_STARTS])
    vals = normalized.values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(starts, vals, side="left")
    out = np.full(len(vals), 999, dtype=np.int64)
    valid = idx < len(starts)
    out[valid] = (starts[idx[valid]] - vals[valid]) // (24 * 3600 * 1_000_000_000)
    return out


def add_academic_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Attach academic-calendar covariates. All are known in advance, so all are safe as
    known-future covariates for the forecaster."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    day = pd.DatetimeIndex(ts).normalize()

    out["phase"] = phase_series(day).to_numpy()
    out["is_holiday"] = day.isin(HOLIDAYS).astype(int)
    out["is_isa"] = day.isin(ISA_DATES).astype(int)
    out["is_esa"] = day.isin(ESA_DATES).astype(int)
    out["is_summer"] = day.isin(SUMMER_BREAK).astype(int)
    out["is_intersem"] = day.isin(INTERSEM_BREAK).astype(int)
    out["days_to_exam"] = days_to_next_exam(day)
    # Saturating transform: the difference between "exam in 40 days" and "in 300 days" is noise.
    out["exam_proximity"] = np.exp(-out["days_to_exam"] / 14.0)
    return out
