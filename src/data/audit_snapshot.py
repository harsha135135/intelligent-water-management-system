"""Audit a refreshed data snapshot before any model is run on it.

Checks, against the published ``dataset/``:

* every day file the two share is byte-identical, except the days deliberately re-fetched
  (``--repaired``) — so the snapshot extends the published data and never silently rewrites it;
* per tank: first and last hour with a reading, empty days, monthly coverage, and how the new
  months compare with the old;
* the curated panel still has exactly the expected tanks.

Writes ``audit.json``, ``coverage_by_month.csv`` and ``audit.md`` into ``--out``.

    PESU_DATASET_DIR=dataset_2026-10-22 python -m src.data.audit_snapshot \\
        --out results/refresh_2026-10-22/audit --repaired 2026-03-16 2026-03-25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import DATASET_DIR, PROJECT_ROOT
from .curate import STALE_TANK_DIRS, TARGET_COL, load_curated_hourly

PUBLISHED = PROJECT_ROOT / "dataset"


def compare_files(snap: Path, base: Path, repaired_days: set[str],
                  repaired_files: set[str]) -> dict:
    same = changed_expected = changed_unexpected = only_base = 0
    unexpected = []
    for f in sorted(base.rglob("*.json")):
        rel = f.relative_to(base)
        g = snap / rel
        if not g.exists():
            only_base += 1
            unexpected.append(f"missing from snapshot: {rel}")
            continue
        if f.read_bytes() == g.read_bytes():
            same += 1
        elif f.stem in repaired_days or str(rel) in repaired_files:
            changed_expected += 1
        else:
            changed_unexpected += 1
            unexpected.append(f"changed: {rel}")
    new = sum(1 for g in snap.rglob("*.json") if not (base / g.relative_to(snap)).exists())
    return {"identical": same, "repaired": changed_expected, "changed_unexpectedly":
            changed_unexpected, "missing_from_snapshot": only_base, "new_files": new,
            "problems": unexpected[:50]}


def revisions(panel: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """How a re-downloaded history differs from the published one, per tank, on the hours both
    have: Waltr recomputes inflow and outflow from the level readings, and has revised that
    computation since ``dataset/`` was downloaded."""
    m = base.merge(panel, on=["item_id", "timestamp"], suffixes=("_old", "_new"))
    rows = []
    for tank, g in m.groupby("item_id"):
        row = {"tank": tank, "hours_compared": int(len(g))}
        for col in ("Outflow in KL", "Inflow in KL", "Closing Value in KL"):
            o, n = g[f"{col}_old"], g[f"{col}_new"]
            ok = o.notna() & n.notna()
            short = col.split()[0].lower()
            row[f"{short}_hours_changed"] = int((o[ok] - n[ok]).abs().gt(1e-9).sum())
            if short != "closing":
                row[f"{short}_total_old_kl"] = round(float(o[ok].sum()), 1)
                row[f"{short}_total_new_kl"] = round(float(n[ok].sum()), 1)
                row[f"{short}_change_pct"] = (round(100 * (n[ok].sum() - o[ok].sum()) / o[ok].sum(), 1)
                                              if o[ok].sum() else None)
        row["outflow_zero_pct_old"] = round(100 * float((g["Outflow in KL_old"] == 0).mean()), 1)
        row["outflow_zero_pct_new"] = round(100 * float((g["Outflow in KL_new"] == 0).mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def per_tank(panel: pd.DataFrame, published_end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for tank, g in panel.groupby("item_id"):
        obs = g.dropna(subset=[TARGET_COL])
        new = g[g.timestamp > published_end]
        new_obs = new.dropna(subset=[TARGET_COL])
        old = g[g.timestamp <= published_end].dropna(subset=[TARGET_COL])
        rows.append({
            "tank": tank,
            "first_reading": obs.timestamp.min(), "last_reading": obs.timestamp.max(),
            "hours_observed": int(len(obs)),
            "coverage_pct": round(100 * len(obs) / max(len(g), 1), 1),
            "new_hours": int(len(new)), "new_hours_observed": int(len(new_obs)),
            "new_coverage_pct": round(100 * len(new_obs) / len(new), 1) if len(new) else None,
            "mean_kl_h_before": round(float(old[TARGET_COL].mean()), 4) if len(old) else None,
            "mean_kl_h_new": round(float(new_obs[TARGET_COL].mean()), 4) if len(new_obs) else None,
            "zero_pct_new": round(100 * float((new_obs[TARGET_COL] == 0).mean()), 1)
            if len(new_obs) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repaired", nargs="*", default=[],
                    help="days re-fetched on purpose (YYYY-MM-DD), allowed to differ")
    ap.add_argument("--repaired-files", nargs="*", default=[],
                    help="single files re-fetched on purpose (FOLDER/YYYY-MM-DD.json)")
    ap.add_argument("--revised", action="store_true",
                    help="the whole history was re-downloaded after Waltr revised its flow "
                         "figures: report the revisions instead of failing on them")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = compare_files(DATASET_DIR, PUBLISHED, set(args.repaired), set(args.repaired_files))
    panel = load_curated_hourly(with_features=False)          # raises unless 24 tanks
    base = load_curated_hourly(PUBLISHED, with_features=False)
    published_end = base["timestamp"].max()
    tanks = per_tank(panel, published_end)
    rev = revisions(panel, base) if args.revised else None
    if rev is not None:
        rev.to_csv(args.out / "revisions_vs_published.csv", index=False)

    month = (panel.assign(month=panel.timestamp.dt.to_period("M").astype(str),
                          obs=panel[TARGET_COL].notna())
             .groupby(["item_id", "month"]).obs.mean().mul(100).round(1).unstack())
    month.to_csv(args.out / "coverage_by_month.csv")

    went_dark = tanks[(tanks.new_hours > 0) & (tanks.new_coverage_pct < 50)].tank.tolist()
    summary = {
        "dataset_dir": str(DATASET_DIR), "files": files,
        "tanks": int(panel.item_id.nunique()),
        "panel_rows": int(len(panel)),
        "panel_range": [str(panel.timestamp.min()), str(panel.timestamp.max())],
        "published_end": str(published_end),
        "tanks_under_50pct_coverage_after_published_end": went_dark,
        "stale_dirs_ignored": STALE_TANK_DIRS,
    }
    (args.out / "audit.json").write_text(json.dumps(
        {**summary, "per_tank": tanks.astype(str).to_dict("records")}, indent=2))

    show = tanks.copy()
    for c in ("first_reading", "last_reading"):
        show[c] = show[c].dt.strftime("%Y-%m-%d %H:00")
    lines = [f"# Data audit — `{DATASET_DIR.name}`", "",
             f"* Panel: {summary['panel_rows']:,} hourly rows, {summary['tanks']} tanks, "
             f"{summary['panel_range'][0]} → {summary['panel_range'][1]} "
             f"(published data ended {published_end}).",
             f"* Files: {files['identical']:,} identical to `dataset/`, {files['repaired']} "
             f"re-fetched on purpose, {files['new_files']:,} new, "
             f"**{files['changed_unexpectedly']} changed unexpectedly**, "
             f"{files['missing_from_snapshot']} missing.",
             f"* Tanks under 50 % coverage since the published end: "
             f"{', '.join(went_dark) or 'none'}.", "",
             show.to_markdown(index=False), ""]
    if rev is not None:
        tot_o, tot_n = rev.outflow_total_old_kl.sum(), rev.outflow_total_new_kl.sum()
        lines += ["## Revisions against the published `dataset/`", "",
                  f"Waltr's current figures, on the {int(rev.hours_compared.sum()):,} tank-hours "
                  f"both downloads cover: campus outflow {tot_o:,.1f} → {tot_n:,.1f} KL "
                  f"({100 * (tot_n - tot_o) / tot_o:+.1f} %); level readings changed on "
                  f"{int(rev.closing_hours_changed.sum()):,} tank-hours.", "",
                  rev.to_markdown(index=False), ""]
    (args.out / "audit.md").write_text("\n".join(lines))
    print("\n".join(lines[:6]))
    if (files["changed_unexpectedly"] and not args.revised) or files["missing_from_snapshot"]:
        raise SystemExit("snapshot rewrites published files - see audit.json 'problems'")


if __name__ == "__main__":
    main()
