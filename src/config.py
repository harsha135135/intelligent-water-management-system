"""Where the pipeline reads its data and writes its results.

Every default is the published study's own path, so with no environment variables set nothing
changes. A second study — a newer data snapshot evaluated on a later window — points the whole
pipeline elsewhere without touching the published files:

    PESU_DATASET_DIR=dataset_2026-10-22 PESU_RESULTS_DIR=results/refresh_2026-10-22 \\
        python -m src.models.score_benchmark --strict

``PESU_EDA_DIR`` moves the EDA outputs the scorer reads (trust tiers, mass balance), since those
are computed from the data too. Report builders write into ``<results>/reports/`` whenever the
results directory is not the published one, so a second study never overwrites the published
page, deck, explorer or ``docs/review_summary.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLISHED_RESULTS = Path("results/chronos2")


def _path(env: str, default: Path, *, anchor: Path | None = None) -> Path:
    value = os.environ.get(env)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() or anchor is None else anchor / p


DATASET_DIR = _path("PESU_DATASET_DIR", PROJECT_ROOT / "dataset", anchor=PROJECT_ROOT)
RESULTS_DIR = _path("PESU_RESULTS_DIR", PUBLISHED_RESULTS)
EDA_DIR = _path("PESU_EDA_DIR", Path("eda"))

# True for the published study; report builders only write to docs/ and reports/ when it is.
IS_PUBLISHED = RESULTS_DIR == PUBLISHED_RESULTS
