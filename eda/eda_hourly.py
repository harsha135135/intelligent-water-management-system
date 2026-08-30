"""Hourly-resolution EDA for PESU campus water demand.

The 56 plots already in ``eda/plots/`` (from ``eda_deep.py``) are almost entirely **daily
aggregates**, yet every production model forecasts **hourly**. This module adds the hourly-level
studies the modelling actually depends on, plus a sensor-integrity check that had never been run.

Sections
  A. Mass-balance sensor integrity  -> per-tank trust tier  (writes eda/tank_trust.json)
  B. Hourly forecastability          -> ACF/PACF, double seasonality, seasonal-naive error floor
  C. Horizon-specific feature study  -> which covariates matter at 6h vs 7d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.curate import TARGET_COL, load_curated_hourly  # noqa: E402

BASE = Path(__file__).resolve().parent
PLOT_DIR = BASE / "plots_hourly"
REPORT = BASE / "eda_hourly_report.txt"
TRUST_JSON = BASE / "tank_trust.json"

HORIZONS = [6, 12, 24, 48, 72, 168]
_lines: list[str] = []
_fig_n = [0]


def rprint(*args) -> None:
    text = " ".join(str(a) for a in args)
    print(text)
    _lines.append(text)


def rsep(title: str, char: str = "-", width: int = 88) -> None:
    rprint("\n" + char * width)
    rprint(title)
    rprint(char * width)


def savefig(name: str, fig=None) -> None:
    _fig_n[0] += 1
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"{_fig_n[0]:02d}_{name}.png"
    (fig or plt.gcf()).savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig or plt.gcf())
    rprint(f"    [plot] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Section A — mass-balance sensor integrity
# ─────────────────────────────────────────────────────────────────────────────

def section_a_mass_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Check the physical identity  Opening + Inflow - Outflow == Closing.

    This is the most direct sensor-integrity test the data allows and it had never been run
    anywhere in the repo. A tank whose mass balance does not close cannot be trusted to supply a
    forecast target, no matter how good the model is.
    """
    rsep("SECTION A - MASS-BALANCE SENSOR INTEGRITY", "=")
    rprint("Identity tested per hour:  Opening + Inflow - Outflow - Closing = residual")
    rprint("A healthy sensor gives residual ~ 0. Persistent non-zero residual means the meter and")
    rprint("the level probe disagree, i.e. the reported outflow is not what the tank actually did.")

    d = df.copy()
    d["residual"] = (
        d["Opening Value in KL"] + d["Inflow in KL"] - d["Outflow in KL"] - d["Closing Value in KL"]
    )
    # Carry-over: this hour's closing should be next hour's opening.
    d["carry"] = d.groupby("item_id")["Opening Value in KL"].shift(-1) - d["Closing Value in KL"]

    rows = []
    for item_id, g in d.groupby("item_id", sort=True):
        res, carry = g["residual"], g["carry"]
        target = g[TARGET_COL]
        n = len(g)
        observed = int(target.notna().sum())
        rows.append({
            "item_id": item_id,
            "rows": n,
            "missing_pct": round(100 * target.isna().mean(), 2),
            "zero_pct": round(100 * (target == 0).mean(), 2),
            "mean_kl": round(float(target.mean()), 4),
            "resid_mae": round(float(res.abs().mean()), 4),
            "resid_p99": round(float(res.abs().quantile(0.99)), 4),
            # Relative breach: residual judged against the tank's own typical hourly flow, not a
            # fixed 0.1 KL. An absolute threshold would flag every high-flow tank and clear every
            # dead one, which is exactly backwards.
            "rel_resid": round(float(res.abs().mean() / max(target.mean(), 1e-9)), 4),
            "breach_pct": round(100 * float(
                (res.abs() > max(0.1, 0.25 * target.mean())).mean()), 2),
            "carry_break_pct": round(100 * float(
                (carry.abs() > max(0.1, 0.25 * target.mean())).mean()), 2),
            "observed": observed,
        })
    mb = pd.DataFrame(rows)

    # Trust tiering. Thresholds are deliberately blunt and stated, not tuned.
    def tier(r) -> str:
        if r["mean_kl"] < 0.01 or r["zero_pct"] > 90:
            return "dead"          # no usable signal to forecast
        if r["missing_pct"] > 10 or r["zero_pct"] > 45 or r["rel_resid"] > 0.10:
            return "degraded"      # forecastable, but treat intervals with suspicion
        return "healthy"

    mb["trust"] = mb.apply(tier, axis=1)
    mb = mb.sort_values(["trust", "mean_kl"], ascending=[True, False]).reset_index(drop=True)

    rprint("\nPer-tank mass-balance and trust tier:")
    rprint(mb.to_string(index=False))
    rprint("\nTier counts: " + ", ".join(
        f"{k}={v}" for k, v in mb["trust"].value_counts().items()))
    rprint("\nTier rules:  dead      = mean < 0.01 KL/h or >90% zero hours")
    rprint("             degraded  = >10% missing, or >45% zero hours, or rel_resid > 0.10")
    rprint("             healthy   = everything else")
    rprint("rel_resid = mean|residual| / mean outflow, i.e. balance error as a fraction of the")
    rprint("tank's own throughput. Scale-free, so it does not punish high-flow tanks.")

    TRUST_JSON.write_text(json.dumps(
        {r["item_id"]: r["trust"] for _, r in mb.iterrows()}, indent=2, sort_keys=True))
    rprint(f"\nWrote {TRUST_JSON}")

    # Plot A1 — residual magnitude and breach rate
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    order = mb.sort_values("resid_mae")
    colors = {"healthy": "#10B981", "degraded": "#F59E0B", "dead": "#EF4444"}
    axes[0].barh(order["item_id"], order["resid_mae"],
                 color=[colors[t] for t in order["trust"]])
    axes[0].set_xlabel("Mean |residual| (KL)")
    axes[0].set_title("Mass-balance error per tank\nOpening + Inflow - Outflow - Closing")
    axes[0].set_xscale("symlog", linthresh=1e-3)
    order2 = mb.sort_values("breach_pct")
    axes[1].barh(order2["item_id"], order2["breach_pct"],
                 color=[colors[t] for t in order2["trust"]])
    axes[1].set_xlabel("% of hours breaching max(0.1 KL, 25% of mean flow)")
    axes[1].set_title("Balance breach rate per tank")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    axes[1].legend(handles, colors.keys(), title="trust tier", loc="lower right")
    plt.tight_layout()
    savefig("massbalance_residual_per_tank", fig)

    # Plot A2 — data availability map
    fig, ax = plt.subplots(figsize=(16, 7))
    piv = (df.assign(day=df["timestamp"].dt.normalize())
             .pivot_table(index="item_id", columns="day",
                          values=TARGET_COL, aggfunc=lambda s: s.notna().sum()))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=24,
                   interpolation="nearest")
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
    step = max(1, len(piv.columns) // 14)
    ax.set_xticks(range(0, len(piv.columns), step),
                  [str(c.date()) for c in piv.columns[::step]], rotation=45, ha="right")
    ax.set_title("Hourly records present per tank-day (24 = complete)")
    fig.colorbar(im, ax=ax, label="records / day")
    plt.tight_layout()
    savefig("massbalance_availability_map", fig)

    return mb


# ─────────────────────────────────────────────────────────────────────────────
# Section B — hourly forecastability
# ─────────────────────────────────────────────────────────────────────────────

def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(x[: len(x) - k], x[k:]) / denom for k in range(nlags + 1)])


def section_b_forecastability(df: pd.DataFrame, mb: pd.DataFrame) -> pd.DataFrame:
    rsep("SECTION B - HOURLY FORECASTABILITY", "=")
    rprint("All 56 existing plots are daily aggregates. Everything below is at the hourly")
    rprint("resolution the models actually operate at.")

    healthy = mb.loc[mb["trust"] != "dead", "item_id"].tolist()
    nlags = 200

    # B1 — hourly ACF, all non-dead tanks
    fig, ax = plt.subplots(figsize=(15, 7))
    for item_id in healthy:
        y = df.loc[df["item_id"] == item_id, TARGET_COL].to_numpy()
        ax.plot(_acf(y, nlags), lw=0.9, alpha=0.65, label=item_id)
    for lag, lab in [(24, "24h"), (48, "48h"), (168, "168h (1 week)")]:
        ax.axvline(lag, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.text(lag + 2, 0.92, lab, fontsize=8, rotation=90, va="top")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Hourly autocorrelation per tank (lags 0-200)")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    plt.tight_layout()
    savefig("hourly_acf_all_tanks", fig)

    # B2 — double seasonality strength + signal decay by horizon
    rows = []
    for item_id in healthy:
        y = df.loc[df["item_id"] == item_id, TARGET_COL].to_numpy()
        a = _acf(y, 200)
        rec = {"item_id": item_id, "acf_24": a[24], "acf_168": a[168] if len(a) > 168 else np.nan}
        for h in HORIZONS:
            rec[f"acf_{h}"] = a[h] if len(a) > h else np.nan
        rows.append(rec)
    decay = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(decay["acf_24"], decay["acf_168"], s=45, color="#1E78E8")
    for r in decay.itertuples(index=False):
        axes[0].annotate(r.item_id, (r.acf_24, r.acf_168), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
    axes[0].axhline(0, color="k", lw=0.7); axes[0].axvline(0, color="k", lw=0.7)
    axes[0].set_xlabel("ACF at lag 24 (daily)")
    axes[0].set_ylabel("ACF at lag 168 (weekly)")
    axes[0].set_title("Double seasonality: daily vs weekly strength")

    for r in decay.itertuples(index=False):
        axes[1].plot(HORIZONS, [getattr(r, f"acf_{h}") for h in HORIZONS],
                     marker="o", lw=1, alpha=0.6)
    axes[1].set_xscale("log"); axes[1].set_xticks(HORIZONS, [str(h) for h in HORIZONS])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("horizon (hours)"); axes[1].set_ylabel("ACF at that lag")
    axes[1].set_title("Signal decay across the 6 forecast horizons")
    plt.tight_layout()
    savefig("hourly_double_seasonality_and_decay", fig)

    rprint("\nSeasonality strength per tank (ACF at seasonal lags):")
    rprint(decay[["item_id", "acf_24", "acf_168"]].sort_values(
        "acf_24", ascending=False).to_string(index=False))

    # B3 — seasonal-naive error floor: the MASE/RMSSE denominator itself
    rsep("B3 - Seasonal-naive error floor (the MASE / RMSSE denominator)")
    rprint("MASE and RMSSE divide model error by these numbers. A tank with a tiny scale is a")
    rprint("tank where beating seasonal naive is hard, regardless of its absolute KL error.")
    rows = []
    for item_id, g in df.groupby("item_id", sort=True):
        y = g[TARGET_COL].to_numpy()
        diffs = y[24:] - y[:-24]
        rows.append({
            "item_id": item_id,
            "scale_mae": round(float(np.nanmean(np.abs(diffs))), 4),
            "scale_rmse": round(float(np.sqrt(np.nanmean(diffs ** 2))), 4),
            "mean_kl": round(float(np.nanmean(y)), 4),
        })
    floor = pd.DataFrame(rows).sort_values("scale_mae", ascending=False)
    floor["scale_over_mean"] = (floor["scale_mae"] / floor["mean_kl"].replace(0, np.nan)).round(3)
    rprint(floor.to_string(index=False))

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.barh(floor["item_id"], floor["scale_mae"], color="#2B8BF5")
    ax.set_xlabel("seasonal-naive MAE at m=24 (KL) - the MASE denominator")
    ax.set_title("Per-tank difficulty floor")
    plt.tight_layout()
    savefig("hourly_seasonal_naive_floor", fig)

    return decay.merge(floor, on="item_id", how="outer")


# ─────────────────────────────────────────────────────────────────────────────
# Section C — horizon-specific feature study
# ─────────────────────────────────────────────────────────────────────────────

def section_c_features(df: pd.DataFrame, mb: pd.DataFrame) -> pd.DataFrame:
    rsep("SECTION C - HORIZON-SPECIFIC FEATURE STUDY", "=")
    rprint("Which covariates carry information about demand h hours ahead? This decides what")
    rprint("goes into Chronos-2's future_df, and it is horizon-dependent: lag terms dominate")
    rprint("short horizons, calendar terms dominate long ones.")

    from sklearn.feature_selection import mutual_info_regression

    healthy = mb.loc[mb["trust"] == "healthy", "item_id"].tolist()
    sub = df[df["item_id"].isin(healthy)].copy()

    feature_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
        "is_holiday", "is_isa", "is_esa", "is_summer", "exam_proximity",
    ]
    records = []
    for h in HORIZONS:
        frames = []
        for _, g in sub.groupby("item_id", sort=False):
            g = g.sort_values("timestamp").copy()
            # Target h hours ahead; lag features are what we would know at the origin.
            g["y_future"] = g[TARGET_COL].shift(-h)
            g["lag_1"] = g[TARGET_COL]
            g["lag_24"] = g[TARGET_COL].shift(23)
            g["lag_168"] = g[TARGET_COL].shift(167)
            g["roll_24"] = g[TARGET_COL].rolling(24).mean()
            frames.append(g)
        block = pd.concat(frames, ignore_index=True)
        cols = feature_cols + ["lag_1", "lag_24", "lag_168", "roll_24"]
        block = block.dropna(subset=cols + ["y_future"])
        if len(block) > 60000:
            block = block.sample(60000, random_state=0)
        mi = mutual_info_regression(block[cols], block["y_future"], random_state=0)
        for col, val in zip(cols, mi):
            records.append({"horizon": h, "feature": col, "mi": float(val)})

    mi_df = pd.DataFrame(records)
    piv = mi_df.pivot(index="feature", columns="horizon", values="mi")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    rprint("\nMutual information with demand h hours ahead (higher = more informative):")
    rprint(piv.round(4).to_string())

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(piv.columns)), [f"{c}h" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > piv.to_numpy().max() * 0.55 else "#0E1726")
    ax.set_title("Feature informativeness by forecast horizon\n(mutual information, healthy tanks)")
    fig.colorbar(im, ax=ax, label="mutual information")
    plt.tight_layout()
    savefig("features_mi_by_horizon", fig)

    # Normalised view: what dominates *within* each horizon.
    share = piv / piv.sum(axis=0)
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(piv.columns))
    for feat in share.index:
        ax.bar([f"{c}h" for c in share.columns], share.loc[feat], bottom=bottom, label=feat)
        bottom += share.loc[feat].to_numpy()
    ax.set_ylabel("share of total MI at that horizon")
    ax.set_title("Where the predictive signal comes from, by horizon")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    savefig("features_mi_share_by_horizon", fig)

    return piv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=None)
    args = ap.parse_args()

    df = load_curated_hourly(args.dataset_dir) if args.dataset_dir else load_curated_hourly()

    rsep("PESU CAMPUS WATER DEMAND - HOURLY EDA", "=")
    rprint(f"rows={len(df):,}  tanks={df['item_id'].nunique()}  "
           f"range={df['timestamp'].min()} .. {df['timestamp'].max()}")

    mb = section_a_mass_balance(df)
    fc = section_b_forecastability(df, mb)
    mi = section_c_features(df, mb)

    rsep("SUMMARY", "=")
    rprint(f"Tanks by trust tier : {dict(mb['trust'].value_counts())}")
    rprint(f"Plots written       : {_fig_n[0]} -> {PLOT_DIR}")
    rprint(f"Trust tiers         : {TRUST_JSON}")

    mb.to_csv(BASE / "tank_mass_balance.csv", index=False)
    fc.to_csv(BASE / "tank_forecastability.csv", index=False)
    mi.to_csv(BASE / "feature_mi_by_horizon.csv")
    REPORT.write_text("\n".join(_lines))
    rprint(f"Report              : {REPORT}")
    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
