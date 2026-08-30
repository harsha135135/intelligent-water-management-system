"""Curated loading of the PESU hourly tank dataset.

Why this module exists
----------------------
``dataset/`` contains **26** directories but only **24** physical tanks. Two tanks were
re-scraped under a slightly different directory name and both copies were kept:

    GJBC_BLOCK_1_A1_BOY_S  (477 days, -> 2026-04-22)  is a superset of
    GJBC_BLOCK_1_A1_BOYS   (439 days, -> 2026-03-16)

    GJBC_BLOCK_1_A2_GIRL_S (477 days, -> 2026-04-22)  is a superset of
    GJBC_BLOCK_1_A2_GIRLS  (439 days, -> 2026-03-16)

Both pairs carry an identical ``Tank Name``, identical ``Tank Dimensions`` and identical daily
totals; every day in the short copy is present in the long copy, plus 38 more. Loading the
directory tree naively double-counts these two tanks, which is what earlier runs did (26 series,
624 holdout rows). We keep the longer copy and relabel it to the canonical id.

The module also guarantees a **gapless hourly index** per tank, which ``Chronos2Pipeline.predict_df``
requires (it validates that timestamps have a regular frequency).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar_pesu import add_academic_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

TARGET_COL = "Outflow in KL"
ITEM_ID_COL = "item_id"
TIMESTAMP_COL = "timestamp"

VALUE_COLS = [
    "Inflow in KL",
    "Outflow in KL",
    "Opening Value in KL",
    "Closing Value in KL",
]

# Directory name -> canonical tank id. The value side is the id we keep.
TANK_ALIASES = {
    "GJBC_BLOCK_1_A1_BOY_S": "GJBC_BLOCK_1_A1_BOYS",
    "GJBC_BLOCK_1_A2_GIRL_S": "GJBC_BLOCK_1_A2_GIRLS",
}
# Stale short copies, superseded by the aliased directories above.
STALE_TANK_DIRS = ["GJBC_BLOCK_1_A1_BOYS", "GJBC_BLOCK_1_A2_GIRLS"]

EXPECTED_TANKS = 24


def _load_upstream_loader():
    """Import ``load_hourly_json_dataset`` from the existing AutoGluon module without importing
    autogluon itself at module-import time (it is slow and not always needed)."""
    path = PROJECT_ROOT / "src" / "models" / "autogluon_hourly_forecasting.py"
    spec = importlib.util.spec_from_file_location("_ag_hourly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_tank_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the stale duplicate directories and relabel the surviving copies."""
    before = df[ITEM_ID_COL].nunique()
    df = df[~df[ITEM_ID_COL].isin(STALE_TANK_DIRS)].copy()
    df[ITEM_ID_COL] = df[ITEM_ID_COL].replace(TANK_ALIASES)
    after = df[ITEM_ID_COL].nunique()
    logger.info("Tank de-duplication: %d directories -> %d tanks", before, after)
    return df


def reindex_gapless_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each tank onto a complete hourly range between its own first and last reading.

    Missing hours become NaN rather than being dropped. Chronos-2 tolerates NaN in the target
    (it masks them); silently collapsing the gap instead would shift every subsequent timestamp
    and corrupt the 24-hour seasonality.
    """
    frames = []
    for item_id, group in df.groupby(ITEM_ID_COL, sort=True):
        group = group.sort_values(TIMESTAMP_COL).drop_duplicates(TIMESTAMP_COL, keep="last")
        full = pd.date_range(
            group[TIMESTAMP_COL].iloc[0], group[TIMESTAMP_COL].iloc[-1], freq="h"
        )
        group = group.set_index(TIMESTAMP_COL).reindex(full)
        group.index.name = TIMESTAMP_COL
        group[ITEM_ID_COL] = item_id
        for col in ("tank_name", "tank_type", "tank_shape"):
            if col in group.columns:
                group[col] = group[col].ffill().bfill()
        frames.append(group.reset_index())
    return pd.concat(frames, ignore_index=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic clock features. All are known at forecast time."""
    out = df.copy()
    ts = pd.to_datetime(out[TIMESTAMP_COL])
    out["hour"] = ts.dt.hour.astype(int)
    out["day_of_week"] = ts.dt.dayofweek.astype(int)
    out["day_of_month"] = ts.dt.day.astype(int)
    out["month"] = ts.dt.month.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    return out


def load_curated_hourly(
    dataset_dir: Path | str = DATASET_DIR,
    *,
    with_features: bool = True,
    max_files_per_tank: int | None = None,
) -> pd.DataFrame:
    """Load the full hourly panel: 24 tanks, gapless hourly index, covariates attached."""
    dataset_dir = Path(dataset_dir)
    upstream = _load_upstream_loader()
    raw = upstream.load_hourly_json_dataset(dataset_dir, max_files_per_tank=max_files_per_tank)

    df = apply_tank_aliases(raw)
    if df[ITEM_ID_COL].nunique() != EXPECTED_TANKS:
        raise ValueError(
            f"Expected {EXPECTED_TANKS} tanks after de-duplication, got "
            f"{df[ITEM_ID_COL].nunique()}: {sorted(df[ITEM_ID_COL].unique())}"
        )

    df = reindex_gapless_hourly(df)

    dup = df.duplicated([ITEM_ID_COL, TIMESTAMP_COL]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (item_id, timestamp) rows survived curation")

    if with_features:
        df = add_time_features(df)
        df = add_academic_features(df, timestamp_col=TIMESTAMP_COL)

    return df.sort_values([ITEM_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)


def tank_metadata(dataset_dir: Path | str = DATASET_DIR) -> pd.DataFrame:
    """Per-tank static metadata, including the ``Tank Dimensions`` string the upstream loader
    discards. Read from the first non-empty JSON of each surviving tank directory."""
    import json

    dataset_dir = Path(dataset_dir)
    rows = []
    for tank_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        name = tank_dir.name
        if name in STALE_TANK_DIRS:
            continue
        item_id = TANK_ALIASES.get(name, name)
        for path in sorted(tank_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not payload.get("data"):
                continue
            rows.append({
                "item_id": item_id,
                "tank_name": payload.get("Tank Name", item_id),
                "tank_type": payload.get("Tank Type"),
                "tank_shape": payload.get("Tank Shape"),
                "tank_dimensions": payload.get("Tank Dimensions"),
            })
            break
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tank coverage summary, useful as a curation receipt."""
    rows = []
    for item_id, g in df.groupby(ITEM_ID_COL, sort=True):
        target = g[TARGET_COL]
        rows.append({
            "item_id": item_id,
            "rows": len(g),
            "start": g[TIMESTAMP_COL].min(),
            "end": g[TIMESTAMP_COL].max(),
            "missing_hours": int(target.isna().sum()),
            "missing_pct": round(100 * target.isna().mean(), 2),
            "zero_pct": round(100 * (target == 0).mean(), 2),
            "mean_kl": round(float(target.mean()), 4),
            "max_kl": round(float(target.max()), 2),
        })
    return pd.DataFrame(rows).sort_values("mean_kl", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = load_curated_hourly()
    print(f"\nrows={len(frame):,}  tanks={frame[ITEM_ID_COL].nunique()}  "
          f"range={frame[TIMESTAMP_COL].min()} .. {frame[TIMESTAMP_COL].max()}")
    print(f"columns: {list(frame.columns)}\n")
    print(summarize(frame).to_string(index=False))
