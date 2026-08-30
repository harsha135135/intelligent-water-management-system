"""Score every model's predictions on the shared backtest grid and emit the review tables.

Outputs written to ``--out-dir`` (default ``results/chronos2``):

* ``metrics_per_tank.csv``      - one row per (model, horizon, tank); the raw material.
* ``metrics_by_horizon.csv``    - one row per (model, horizon); macro + volume-weighted rollup.
* ``per_tank_comparison.csv``   - Chronos-2 vs NPTS vs SeasonalNaive per tank and horizon, with
                                  improvement %, winner and the sensor trust tier from the EDA.
* ``benchmark_table.md``        - the complete benchmark document (every model, every horizon,
                                  every metric, plus per-tank, covariate and calibration tables).

Nothing here refits or re-forecasts anything: it reads ``predictions_*.parquet`` produced by the
completed rolling-origin run. Metrics that cannot be computed are left blank, never imputed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.curate import load_curated_hourly
from .backtest import assert_comparable
from .metrics import aggregate_metrics, attach_scales, coverage, per_tank_metrics

logger = logging.getLogger(__name__)

HORIZON_LABEL = {6: "6h", 12: "12h", 24: "1d", 48: "2d", 72: "3d", 168: "7d"}
HORIZON_ORDER = [6, 12, 24, 48, 72, 168]

# The three-model headline. NPTS is the incumbent: the shipping AutoGluon WeightedEnsemble
# carries weight 1.0 on it. SeasonalNaive-24 is the reference floor that MASE/RMSSE scale to.
CHRONOS = "Chronos2-ZS"
INCUMBENT = "NPTS"
REFERENCE = "SeasonalNaive"
HEADLINE = [CHRONOS, INCUMBENT, REFERENCE]

# Chronos-2 covariate variants, in increasing order of compute.
VARIANTS = ["Chronos2-ZS", "Chronos2-COV-LEAN", "Chronos2-COV-XL", "Chronos2-COV"]

TRUST_PATH = Path("eda/tank_trust.json")
QUALITY_PATH = Path("eda/tank_mass_balance.csv")
RUN_MANIFEST = Path("results/chronos2/run_manifest.json")


def _fmt(v, nd: int = 4) -> str:
    """Blank for anything not measured. Never a placeholder number."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return ""
    return f"{v:.{nd}f}"


def _pct(v, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return ""
    return f"{v:+.{nd}f}%"


def load_all_predictions(out_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(out_dir.glob("predictions_*.parquet")):
        df = pd.read_parquet(path)
        frames.append(df)
        logger.info("loaded %-46s %8d rows  models=%s",
                    path.name, len(df), sorted(df["model"].unique()))
    if not frames:
        raise FileNotFoundError(f"No predictions_*.parquet under {out_dir}")
    return pd.concat(frames, ignore_index=True)


def load_sensor_quality() -> pd.DataFrame:
    """Trust tier and missingness per tank, from the EDA. Empty frame if the EDA has not run."""
    if not TRUST_PATH.exists():
        logger.warning("%s missing - trust tier column will be blank", TRUST_PATH)
        return pd.DataFrame(columns=["item_id", "sensor_trust_tier"])
    trust = json.loads(TRUST_PATH.read_text())
    q = pd.DataFrame({"item_id": list(trust), "sensor_trust_tier": list(trust.values())})
    if QUALITY_PATH.exists():
        mb = pd.read_csv(QUALITY_PATH)[
            ["item_id", "missing_pct", "zero_pct", "rel_resid", "mean_kl"]
        ].rename(columns={"mean_kl": "panel_mean_kl_h"})
        q = q.merge(mb, on="item_id", how="left")
    return q


def interval_width(scored: pd.DataFrame, lo: str = "0.1", hi: str = "0.9") -> pd.DataFrame:
    """Mean p10-p90 width. Read next to coverage: a narrow band with low coverage is
    overconfidence, a wide band with high coverage is underconfidence."""
    if lo not in scored.columns or hi not in scored.columns:
        return pd.DataFrame(columns=["model", "horizon", "p10_p90_width"])
    # Same row set as coverage(), so the two are read on identical evaluation rows.
    s = scored.dropna(subset=["actual", lo, hi])
    return (s.assign(w=s[hi] - s[lo])
             .groupby(["model", "horizon"], sort=False)["w"].mean()
             .reset_index().rename(columns={"w": "p10_p90_width"}))


# ── per-tank three-way comparison ────────────────────────────────────────────

def per_tank_comparison(per_tank: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Chronos-2 vs NPTS vs SeasonalNaive for every (tank, horizon).

    ``winner`` is decided on MASE, the scale-free metric, because a raw-MAE winner on a tank
    that moves 0.001 KL/h is not a meaningful statement. Where a model's MASE is undefined
    (constant history -> zero seasonal-naive denominator) the cell is left blank and the winner
    is blank too, rather than being decided on a metric that does not exist.
    """
    wide = {}
    for tag, model in (("chronos2", CHRONOS), ("npts", INCUMBENT), ("seasonal_naive", REFERENCE)):
        g = per_tank[per_tank["model"] == model].set_index(["item_id", "horizon"])
        wide[tag] = g

    idx = wide["chronos2"].index.union(wide["npts"].index).union(wide["seasonal_naive"].index)
    out = pd.DataFrame(index=idx).sort_index()
    out.index.names = ["item_id", "horizon"]

    out["n_observations"] = wide["chronos2"]["n"].reindex(idx)
    out["mean_actual_kl_h"] = wide["chronos2"]["mean_actual"].reindex(idx)
    for metric in ("mae", "rmse", "mase", "rmsse"):
        for tag in ("chronos2", "npts", "seasonal_naive"):
            out[f"{tag}_{metric}"] = wide[tag][metric].reindex(idx)

    c, n, s = out["chronos2_mase"], out["npts_mase"], out["seasonal_naive_mase"]
    out["chronos2_vs_npts_pct"] = 100 * (n - c) / n
    out["chronos2_vs_naive_pct"] = 100 * (s - c) / s
    out["chronos2_vs_npts_pct_mae"] = (
        100 * (out["npts_mae"] - out["chronos2_mae"]) / out["npts_mae"])

    defined = c.notna() & n.notna() & s.notna()
    best = pd.concat([c, n, s], axis=1).idxmin(axis=1)
    winner = best.map({"chronos2_mase": "Chronos-2",
                       "npts_mase": "NPTS",
                       "seasonal_naive_mase": "SeasonalNaive"})
    out["winner"] = winner.where(defined)
    out["beats_seasonal_naive_reference"] = (c < 1.0).where(c.notna())

    out = out.reset_index().rename(columns={"item_id": "tank"})
    out["horizon_label"] = out["horizon"].map(HORIZON_LABEL)
    out = out.merge(quality.rename(columns={"item_id": "tank"}), on="tank", how="left")

    cols = ["tank", "horizon", "horizon_label", "n_observations", "mean_actual_kl_h",
            "chronos2_mae", "npts_mae", "seasonal_naive_mae",
            "chronos2_rmse", "npts_rmse", "seasonal_naive_rmse",
            "chronos2_mase", "npts_mase", "seasonal_naive_mase",
            "chronos2_rmsse", "npts_rmsse", "seasonal_naive_rmsse",
            "chronos2_vs_npts_pct", "chronos2_vs_npts_pct_mae", "chronos2_vs_naive_pct",
            "winner", "beats_seasonal_naive_reference", "sensor_trust_tier"]
    cols += [c for c in ("missing_pct", "zero_pct", "rel_resid", "panel_mean_kl_h")
             if c in out.columns]
    horder = {h: i for i, h in enumerate(HORIZON_ORDER)}
    return out[cols].sort_values(
        ["horizon", "tank"], key=lambda s: s.map(horder) if s.name == "horizon" else s
    ).reset_index(drop=True)


# ── markdown ─────────────────────────────────────────────────────────────────

def _headline_block(agg: pd.DataFrame) -> list[str]:
    lines = ["## 2. Headline comparison - Chronos-2 vs NPTS vs SeasonalNaive-24", "",
             "`Chronos-2` = `Chronos2-ZS` (zero-shot, the proposed model). `NPTS` = the "
             "incumbent. `SeasonalNaive-24` = reference baseline.", "",
             "`abs diff` and `% impr` are always *NPTS minus Chronos-2*, so a positive number "
             "means Chronos-2 is better. Lower is better for all four metrics.", ""]
    have = set(agg["model"])
    if not set(HEADLINE) <= have:
        lines += [f"> Not all headline models present. Found: {sorted(have)}", ""]
        return lines
    for metric in ("mae", "rmse", "mase", "rmsse"):
        unit = " (KL/h)" if metric in ("mae", "rmse") else " (scale-free)"
        lines.append(f"### {metric.upper()}{unit}")
        lines.append("")
        lines.append("| Horizon | Chronos-2 | NPTS | SeasonalNaive-24 | abs diff vs NPTS | "
                     "% impr vs NPTS | abs diff vs naive | % impr vs naive |")
        lines.append("|---|---|---|---|---|---|---|---|")
        p = agg.pivot(index="horizon", columns="model", values=f"macro_{metric}")
        for h in HORIZON_ORDER:
            if h not in p.index:
                continue
            cv, nv, sv = p.loc[h, CHRONOS], p.loc[h, INCUMBENT], p.loc[h, REFERENCE]
            lines.append(
                f"| {HORIZON_LABEL[h]} | **{_fmt(cv)}** | {_fmt(nv)} | {_fmt(sv)} | "
                f"{_fmt(nv - cv)} | {_pct(100 * (nv - cv) / nv)} | "
                f"{_fmt(sv - cv)} | {_pct(100 * (sv - cv) / sv)} |")
        lines.append("")
    return lines


def _all_models_block(agg: pd.DataFrame) -> list[str]:
    lines = ["## 3. Every model, every horizon, every metric", "",
             "Sorted by MASE within each horizon. `vw_*` weights each tank by its mean demand "
             "(campus-KL view); `macro_*` weights every tank equally.", ""]
    for horizon in [h for h in HORIZON_ORDER if h in set(agg["horizon"])]:
        block = agg[agg["horizon"] == horizon].sort_values("macro_mase")
        lines.append(f"### Horizon {HORIZON_LABEL[horizon]} ({horizon}h)")
        lines.append("")
        lines.append("| Model | MAE | RMSE | MASE | RMSSE | vw MAE | vw RMSE | rows | "
                     "tanks (scaled) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in block.itertuples(index=False):
            mark = "**" if r.model in HEADLINE else ""
            lines.append(
                f"| {mark}{r.model}{mark} | {_fmt(r.macro_mae)} | {_fmt(r.macro_rmse)} | "
                f"{_fmt(r.macro_mase)} | {_fmt(r.macro_rmsse)} | {_fmt(r.vw_mae)} | "
                f"{_fmt(r.vw_rmse)} | {int(r.rows_evaluated):,} | "
                f"{int(r.n_tanks)} ({int(r.n_tanks_scaled)}) |")
        lines.append("")
    return lines


def _covariate_block(agg: pd.DataFrame) -> list[str]:
    present = [v for v in VARIANTS if v in set(agg["model"])]
    if len(present) < 2:
        return []
    runtimes = {}
    if RUN_MANIFEST.exists():
        man = json.loads(RUN_MANIFEST.read_text())
        runtimes = {k: v.get("wall_clock_s") for k, v in man.get("variants", {}).items()}
    lines = ["## 4. Covariate variants - does conditioning on covariates pay for itself?", "",
             "All four variants use the same backbone (`amazon/chronos-2`, 119.5M params) on the "
             "same grid. Runtime is the measured wall clock for the full 24-origin x 6-horizon "
             "backtest from `run_manifest.json`; blank where not recorded.", ""]
    for metric in ("mase", "mae"):
        lines.append(f"### macro {metric.upper()} by horizon")
        lines.append("")
        lines.append("| Variant | " + " | ".join(HORIZON_LABEL[h] for h in HORIZON_ORDER)
                     + " | runtime (min) | vs ZS at 7d |")
        lines.append("|---" * (len(HORIZON_ORDER) + 3) + "|")
        p = agg[agg["model"].isin(present)].pivot(
            index="model", columns="horizon", values=f"macro_{metric}")
        zs = p.loc[CHRONOS] if CHRONOS in p.index else None
        for v in present:
            cells = [_fmt(p.loc[v, h]) if h in p.columns else "" for h in HORIZON_ORDER]
            rt = runtimes.get(v)
            delta = ""
            if zs is not None and 168 in p.columns:
                delta = _pct(100 * (zs[168] - p.loc[v, 168]) / zs[168])
            lines.append(f"| {v} | " + " | ".join(cells) + " | "
                         + (f"{rt / 60:.1f}" if rt else "") + f" | {delta} |")
        lines.append("")
    spread = agg[agg["model"].isin(present)].groupby("horizon")["macro_mase"].agg(
        lambda s: s.max() - s.min())
    lines.append("| Horizon | MASE spread across the four variants |")
    lines.append("|---|---|")
    for h in HORIZON_ORDER:
        if h in spread.index:
            lines.append(f"| {HORIZON_LABEL[h]} | {_fmt(spread[h], 5)} |")
    lines.append("")
    return lines


def _calibration_block(agg: pd.DataFrame) -> list[str]:
    if "p10_p90_coverage" not in agg.columns:
        return []
    lines = ["## 5. Uncertainty calibration - p10-p90 interval", "",
             "Nominal coverage for a p10-p90 band is **0.80**. Below 0.80 the band is too narrow "
             "(overconfident); above 0.80 it is too wide (underconfident). Width is the mean "
             "`p90 - p10` in KL/h - coverage and width must be read together, since a band can "
             "always be made to cover by being made useless.", "",
             "### Empirical p10-p90 coverage (nominal 0.80)", ""]
    lines.append("| Model | " + " | ".join(HORIZON_LABEL[h] for h in HORIZON_ORDER) + " |")
    lines.append("|---" * (len(HORIZON_ORDER) + 1) + "|")
    p = agg.pivot(index="model", columns="horizon", values="p10_p90_coverage")
    order = [m for m in HEADLINE if m in p.index] + sorted(set(p.index) - set(HEADLINE))
    for m in order:
        mark = "**" if m in HEADLINE else ""
        cells = [_fmt(p.loc[m, h], 3) if h in p.columns else "" for h in HORIZON_ORDER]
        lines.append(f"| {mark}{m}{mark} | " + " | ".join(cells) + " |")
    lines.append("")
    if "p10_p90_width" in agg.columns:
        lines += ["### Mean p10-p90 width (KL/h)", ""]
        lines.append("| Model | " + " | ".join(HORIZON_LABEL[h] for h in HORIZON_ORDER) + " |")
        lines.append("|---" * (len(HORIZON_ORDER) + 1) + "|")
        w = agg.pivot(index="model", columns="horizon", values="p10_p90_width")
        for m in order:
            mark = "**" if m in HEADLINE else ""
            cells = [_fmt(w.loc[m, h], 3) if h in w.columns else "" for h in HORIZON_ORDER]
            lines.append(f"| {mark}{m}{mark} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _per_tank_block(cmp_df: pd.DataFrame) -> list[str]:
    if cmp_df.empty:
        return []
    lines = ["## 6. Per-tank comparison", "",
             "Every tank is shown at every horizon - nothing is filtered out for looking bad. "
             "`winner` is decided on MASE. Full machine-readable version: "
             "[`per_tank_comparison.csv`](per_tank_comparison.csv).", ""]

    lines += ["### 6.1 Tanks won, by horizon and metric (Chronos-2 vs NPTS, 24 tanks)", ""]
    lines.append("| Horizon | MAE | RMSE | MASE | RMSSE | 3-way MASE winner (C / N / SN) |")
    lines.append("|---|---|---|---|---|---|")
    for h in HORIZON_ORDER:
        d = cmp_df[cmp_df["horizon"] == h]
        if d.empty:
            continue
        wins = []
        for m in ("mae", "rmse", "mase", "rmsse"):
            ok = d[f"chronos2_{m}"].notna() & d[f"npts_{m}"].notna()
            wins.append(f"{int((d.loc[ok, f'chronos2_{m}'] < d.loc[ok, f'npts_{m}']).sum())}"
                        f"/{int(ok.sum())}")
        vc = d["winner"].value_counts()
        three = (f"{int(vc.get('Chronos-2', 0))} / {int(vc.get('NPTS', 0))} / "
                 f"{int(vc.get('SeasonalNaive', 0))}")
        lines.append(f"| {HORIZON_LABEL[h]} | " + " | ".join(wins) + f" | {three} |")
    lines.append("")

    lines += ["### 6.2 Chronos-2 MASE by tank and horizon", "",
              "Values above 1.00 are worse than the seasonal-naive reference.", ""]
    p = cmp_df.pivot(index="tank", columns="horizon", values="chronos2_mase")
    tier = cmp_df.drop_duplicates("tank").set_index("tank")["sensor_trust_tier"]
    p = p.reindex(columns=[h for h in HORIZON_ORDER if h in p.columns])
    lines.append("| Tank | tier | " + " | ".join(HORIZON_LABEL[h] for h in p.columns) + " |")
    lines.append("|---" * (len(p.columns) + 2) + "|")
    for tank, row in p.sort_values(p.columns[2] if len(p.columns) > 2 else p.columns[0]).iterrows():
        cells = [_fmt(v, 3) for v in row]
        lines.append(f"| `{tank}` | {tier.get(tank, '')} | " + " | ".join(cells) + " |")
    lines.append("")

    for h in HORIZON_ORDER:
        d = cmp_df[cmp_df["horizon"] == h]
        if d.empty:
            continue
        lines += [f"### 6.3 Full per-tank table at {HORIZON_LABEL[h]}", "",
                  "| Tank | tier | n rows | mean actual (KL/h) | C2 MAE | NPTS MAE | SN MAE | "
                  "C2 MASE | NPTS MASE | SN MASE | C2 RMSSE | NPTS RMSSE | SN RMSSE | "
                  "% impr vs NPTS (MASE) | winner |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in d.sort_values("chronos2_vs_npts_pct", ascending=False).itertuples(index=False):
            lines.append(
                f"| `{r.tank}` | {r.sensor_trust_tier or ''} | "
                f"{'' if pd.isna(r.n_observations) else int(r.n_observations)} | "
                f"{_fmt(r.mean_actual_kl_h)} | "
                f"{_fmt(r.chronos2_mae)} | {_fmt(r.npts_mae)} | {_fmt(r.seasonal_naive_mae)} | "
                f"{_fmt(r.chronos2_mase)} | {_fmt(r.npts_mase)} | {_fmt(r.seasonal_naive_mase)} | "
                f"{_fmt(r.chronos2_rmsse)} | {_fmt(r.npts_rmsse)} | "
                f"{_fmt(r.seasonal_naive_rmsse)} | {_pct(r.chronos2_vs_npts_pct)} | "
                f"{r.winner if isinstance(r.winner, str) else ''} |")
        lines.append("")
    return lines


def to_markdown(agg: pd.DataFrame, cmp_df: pd.DataFrame, counts: pd.DataFrame | None,
                grid: dict) -> str:
    lines = [
        "# Chronos-2 benchmark - PESU campus water demand",
        "",
        "Complete benchmark tables. Generated by `python -m src.models.score_benchmark` from "
        "the completed rolling-origin predictions in `results/chronos2/predictions_*.parquet`; "
        "no model is refitted or re-forecast here.",
        "",
        "Lower is better for every metric in this document. `macro_*` averages tanks equally; "
        "MASE/RMSSE divide each tank's error by that tank's own seasonal-naive (m=24) error, so "
        "**1.0 means as accurate as seasonal naive**, below 1.0 means better. MASE and RMSSE are "
        "ratios, *not* percentages. Blank cells are metrics that could not be measured; no cell "
        "is imputed.",
        "",
        "## 1. Evaluation grid",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Tanks | {grid['tanks']} physical tanks (26 dataset directories de-duplicated) |",
        f"| Forecast origins | {grid['origins']} |",
        f"| Origin stride | {grid['stride']} hours (co-prime with 24) |",
        f"| Hours-of-day covered by origins | {grid['hours_covered']} of 24 |",
        f"| Horizons | {', '.join(HORIZON_LABEL[h] for h in HORIZON_ORDER)} |",
        f"| Models scored | {grid['n_models']} |",
        f"| Origin range | {grid['first_origin']} .. {grid['last_origin']} |",
        f"| Rows scored per model | {grid['rows_per_model']:,} |",
        "",
    ]
    if counts is not None and len(counts):
        lines += ["### Rows evaluated per (horizon, model) - identical by construction", "",
                  "| Horizon | " + " | ".join(counts.columns) + " |",
                  "|---" * (len(counts.columns) + 1) + "|"]
        for h in HORIZON_ORDER:
            if h in counts.index:
                row = counts.loc[h]
                lines.append(f"| {HORIZON_LABEL[h]} | "
                             + " | ".join(f"{int(v):,}" for v in row) + " |")
        lines += ["", f"Distinct row counts within each horizon: "
                      f"{sorted(counts.nunique(axis=1).unique().tolist())} "
                      f"(1 means every model scored exactly the same rows).", ""]

    lines += _headline_block(agg)
    lines += _all_models_block(agg)
    lines += _covariate_block(agg)
    lines += _calibration_block(agg)
    lines += _per_tank_block(cmp_df)

    if not agg.empty:
        best = agg.loc[agg.groupby("horizon")["macro_mase"].idxmin()]
        lines += ["## 7. Best model per horizon (by macro MASE, all models)", "",
                  "| Horizon | Best model | MASE | RMSSE | MAE | RMSE |",
                  "|---|---|---|---|---|---|"]
        for r in best.sort_values("horizon", key=lambda s: s.map(
                {h: i for i, h in enumerate(HORIZON_ORDER)})).itertuples(index=False):
            lines.append(f"| {HORIZON_LABEL.get(r.horizon, r.horizon)} | {r.model} | "
                         f"{_fmt(r.macro_mase)} | {_fmt(r.macro_rmsse)} | {_fmt(r.macro_mae)} | "
                         f"{_fmt(r.macro_rmse)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/chronos2")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if models were scored on different row counts.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    preds = load_all_predictions(out_dir)
    panel = load_curated_hourly(with_features=False)

    scored = attach_scales(preds, panel)

    leak = int((scored["timestamp"] <= scored["origin"]).sum())
    if leak:
        raise AssertionError(f"{leak} evaluation rows are at or before their forecast origin")

    counts = None
    try:
        counts = assert_comparable(scored)
        logger.info("Row-count parity OK:\n%s", counts.to_string())
    except AssertionError as exc:
        if args.strict:
            raise
        logger.warning("%s", exc)

    per_tank = per_tank_metrics(scored)
    agg = aggregate_metrics(per_tank)
    cov = coverage(scored)
    if len(cov):
        agg = agg.merge(cov, on=["model", "horizon"], how="left")
    wid = interval_width(scored)
    if len(wid):
        agg = agg.merge(wid, on=["model", "horizon"], how="left")

    quality = load_sensor_quality()
    cmp_df = per_tank_comparison(per_tank, quality)

    origins = sorted(pd.to_datetime(pd.Series(scored["origin"].unique())))
    strides = sorted({int(d.total_seconds() // 3600) for d in np.diff(origins)})
    grid = {
        "tanks": int(scored["item_id"].nunique()),
        "origins": len(origins),
        "stride": strides[0] if len(strides) == 1 else f"{strides} (NOT uniform)",
        "hours_covered": len({t.hour for t in origins}),
        "n_models": int(scored["model"].nunique()),
        "first_origin": str(origins[0]),
        "last_origin": str(origins[-1]),
        "rows_per_model": int(len(scored.dropna(subset=["actual", "pred"]))
                              / max(scored["model"].nunique(), 1)),
    }

    per_tank.to_csv(out_dir / "metrics_per_tank.csv", index=False)
    agg.to_csv(out_dir / "metrics_by_horizon.csv", index=False)
    cmp_df.to_csv(out_dir / "per_tank_comparison.csv", index=False)
    (out_dir / "benchmark_table.md").write_text(to_markdown(agg, cmp_df, counts, grid))

    logger.info("Wrote metrics_per_tank.csv, metrics_by_horizon.csv, "
                "per_tank_comparison.csv, benchmark_table.md")
    show = ["model", "horizon", "macro_mae", "macro_rmse", "macro_mase", "macro_rmsse"]
    print()
    print(agg[show].to_string(index=False))


if __name__ == "__main__":
    main()
