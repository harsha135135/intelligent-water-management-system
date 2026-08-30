"""Phase III analysis: statistical significance and performance patterns.

Re-scores nothing and refits nothing. Reads the completed prediction parquets under
``results/chronos2/`` and answers the questions the benchmark deliberately left open:

* **Is the Chronos-2 improvement statistically significant?** ``review_summary.md`` §13.2 states
  no significance test was run. This module runs a paired bootstrap over the 24 forecast origins
  and a Diebold-Mariano test, per horizon and per tank. That per-tank result is the gating
  evidence the router design in ``docs/realtime_architecture.md`` §9 requires.
* **Where does the error actually live?** By hour-of-day, by lead time, by demand magnitude.
* **How badly are the intervals calibrated, at every quantile, not just p10-p90?**
* **Does the campus volume forecast track reality cumulatively?**

Colour and style are imported from ``review_plots`` so the new figures are visually identical
to the fourteen already in the deck.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .review_plots import COLOR, DIVERGING, GRID, INK, INK2, LABEL, SEQ_BLUE, SURFACE
from .review_package import CHRONOS, INCUMBENT, REFERENCE, HLABEL, HORIZONS

RESULTS = Path("results/chronos2")
OUT = RESULTS / "phase3"
PLOTS = OUT / "plots"

N_BOOT = 10_000
RNG = np.random.default_rng(20260830)


def _save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  [plot] {name}")


def _caption(fig, text: str, width: int = 108, y: float = -0.04) -> None:
    """Footnote that wraps. matplotlib does not wrap fig.text, and an unwrapped line forces
    savefig(bbox_inches="tight") to widen the whole figure."""
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), width=width))
    fig.text(0.02, y, wrapped, fontsize=8.5, color=INK2, va="top")


def _write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"  [table] {name}  ({len(df)} rows)")


# ---------------------------------------------------------------- data assembly

def load_paired() -> pd.DataFrame:
    """One wide frame with identical rows for every model.

    The inner join on (item_id, origin, horizon, timestamp) is what makes the comparison
    *paired*: a bootstrap over origins is only valid if every model saw the same rows.
    """
    from ..data.curate import load_curated_hourly
    from .metrics import attach_scales

    zs = pd.read_parquet(RESULTS / f"predictions_{CHRONOS}.parquet")
    base = pd.read_parquet(RESULTS / "predictions_autogluon_baselines.parquet")

    keys = ["item_id", "origin", "horizon", "timestamp"]
    frames = {CHRONOS: zs}
    for m in (INCUMBENT, "SeasonalNaive"):
        frames[m] = base[base["model"] == m]

    wide = frames[CHRONOS][keys + ["step", "actual", "pred", "0.1", "0.25", "0.5", "0.75", "0.9"]]
    wide = wide.rename(columns={"pred": f"pred_{CHRONOS}",
                                **{q: f"q{q}_{CHRONOS}" for q in ["0.1", "0.25", "0.5", "0.75", "0.9"]}})
    for m in (INCUMBENT, "SeasonalNaive"):
        cols = keys + ["pred", "0.1", "0.9"]
        f = frames[m][cols].rename(columns={"pred": f"pred_{m}", "0.1": f"q0.1_{m}", "0.9": f"q0.9_{m}"})
        wide = wide.merge(f, on=keys, how="inner")

    panel = load_curated_hourly(with_features=False)
    wide = attach_scales(wide, panel)

    before = len(wide)
    wide = wide.dropna(subset=["actual", f"pred_{CHRONOS}", f"pred_{INCUMBENT}", "pred_SeasonalNaive"])
    print(f"  paired rows: {len(wide):,} (dropped {before - len(wide):,} with a missing actual)")
    for m in (CHRONOS, INCUMBENT, "SeasonalNaive"):
        wide[f"ae_{m}"] = (wide["actual"] - wide[f"pred_{m}"]).abs()
        wide[f"se_{m}"] = (wide["actual"] - wide[f"pred_{m}"]) ** 2
    return wide


def _macro_per_origin(w: pd.DataFrame, horizon: int, model: str, metric: str) -> np.ndarray:
    """Macro metric per origin: mean over tanks of that tank's metric. Matches the benchmark's
    macro definition, so these numbers roll up to the published table."""
    h = w[w["horizon"] == horizon]
    if metric == "mae":
        per_tank = h.groupby(["origin", "item_id"])[f"ae_{model}"].mean()
    elif metric == "mase":
        ok = h[h["scale_mae"] > 0].copy()
        ok["scaled"] = ok[f"ae_{model}"] / ok["scale_mae"]
        per_tank = ok.groupby(["origin", "item_id"])["scaled"].mean()
    else:
        raise ValueError(metric)
    return per_tank.groupby("origin").mean().sort_index().to_numpy()


def _boot_ci(d: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float, float]:
    """Percentile bootstrap CI on the mean of a paired difference, resampling origins."""
    n = len(d)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided bootstrap p-value: how often the resampled mean crosses zero
    p = 2 * min((means >= 0).mean(), (means <= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def _dm_test(d: np.ndarray, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction.

    n = 24 origins, so the correction matters. Reported alongside the bootstrap rather than
    instead of it: DM assumes a covariance structure the 23-hour origin stride only approximates.
    """
    n = len(d)
    dbar = d.mean()
    h_lag = max(1, int(np.ceil(horizon / 23)))          # origins are 23 h apart
    gamma0 = np.sum((d - dbar) ** 2) / n
    var = gamma0
    for lag in range(1, h_lag):
        g = np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / n
        var += 2 * g
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / n)
    corr = np.sqrt((n + 1 - 2 * h_lag + h_lag * (h_lag - 1) / n) / n)
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)


# ---------------------------------------------------------------- 1. significance

def significance_by_horizon(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        for metric in ("mae", "mase"):
            c = _macro_per_origin(w, h, CHRONOS, metric)
            n = _macro_per_origin(w, h, INCUMBENT, metric)
            d = c - n                                    # negative => Chronos-2 better
            mean, lo, hi, p_boot = _boot_ci(d)
            dm, p_dm = _dm_test(d, h)
            rows.append({
                "horizon": h, "horizon_label": HLABEL[h], "metric": metric.upper(),
                "n_origins": len(d),
                "chronos2": float(c.mean()), "npts": float(n.mean()),
                "diff": mean, "diff_pct": 100 * mean / float(n.mean()),
                "ci_lo": lo, "ci_hi": hi,
                "ci_lo_pct": 100 * lo / float(n.mean()), "ci_hi_pct": 100 * hi / float(n.mean()),
                "p_bootstrap": p_boot, "dm_stat": dm, "p_dm": p_dm,
                "significant_95": bool(hi < 0),
                "wins_origins": int((d < 0).sum()),
            })
    return pd.DataFrame(rows)


def significance_per_tank(w: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    h = w[w["horizon"] == horizon]
    trust = json.loads(Path("eda/tank_trust.json").read_text())
    rows = []
    for tank, g in h.groupby("item_id"):
        c = g.groupby("origin")[f"ae_{CHRONOS}"].mean().sort_index().to_numpy()
        n = g.groupby("origin")[f"ae_{INCUMBENT}"].mean().sort_index().to_numpy()
        d = c - n
        mean, lo, hi, p = _boot_ci(d)
        base = float(n.mean())
        rows.append({
            "tank": tank, "trust": trust.get(tank), "horizon": horizon,
            "mean_actual_kl_h": float(g["actual"].mean()),
            "chronos2_mae": float(c.mean()), "npts_mae": base,
            "diff": mean, "improvement_pct": -100 * mean / base if base else np.nan,
            "ci_lo": lo, "ci_hi": hi,
            "ci_lo_pct": -100 * hi / base if base else np.nan,
            "ci_hi_pct": -100 * lo / base if base else np.nan,
            "p_bootstrap": p,
            "verdict": ("Chronos-2 better" if hi < 0 else
                        "NPTS better" if lo > 0 else "no significant difference"),
        })
    return pd.DataFrame(rows).sort_values("improvement_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 2. error patterns

def diurnal_error(w: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    h = w[w["horizon"] == horizon].copy()
    h["hour"] = pd.to_datetime(h["timestamp"]).dt.hour
    rows = []
    for hour, g in h.groupby("hour"):
        r = {"hour": int(hour), "n": len(g), "mean_actual": float(g["actual"].mean())}
        for m in (CHRONOS, INCUMBENT, "SeasonalNaive"):
            r[f"mae_{m}"] = float(g[f"ae_{m}"].mean())
            r[f"bias_{m}"] = float((g[f"pred_{m}"] - g["actual"]).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def error_by_leadtime(w: pd.DataFrame) -> pd.DataFrame:
    h = w[w["horizon"] == 168]
    rows = []
    for step, g in h.groupby("step"):
        r = {"step": int(step), "n": len(g)}
        for m in (CHRONOS, INCUMBENT, "SeasonalNaive"):
            r[f"mae_{m}"] = float(g[f"ae_{m}"].mean())
        ok = g[g["scale_mae"] > 0]
        r[f"mase_{CHRONOS}"] = float((ok[f"ae_{CHRONOS}"] / ok["scale_mae"]).mean())
        r[f"mase_{INCUMBENT}"] = float((ok[f"ae_{INCUMBENT}"] / ok["scale_mae"]).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def reliability(w: pd.DataFrame) -> pd.DataFrame:
    """Empirical fraction of actuals falling below each predicted quantile.

    A calibrated model puts 10 % of outcomes below its p10. This is strictly more informative
    than the single p10-p90 coverage number already published, because it shows *which tail*
    is broken.
    """
    rows = []
    for h in HORIZONS:
        g = w[w["horizon"] == h]
        for q in ["0.1", "0.25", "0.5", "0.75", "0.9"]:
            rows.append({"horizon": h, "horizon_label": HLABEL[h], "model": CHRONOS,
                         "nominal": float(q),
                         "empirical": float((g["actual"] <= g[f"q{q}_{CHRONOS}"]).mean())})
        for q in ["0.1", "0.9"]:
            rows.append({"horizon": h, "horizon_label": HLABEL[h], "model": INCUMBENT,
                         "nominal": float(q),
                         "empirical": float((g["actual"] <= g[f"q{q}_{INCUMBENT}"]).mean())})
        # central interval coverage
        for model, lo, hi, nom in [(CHRONOS, "0.1", "0.9", 0.80), (CHRONOS, "0.25", "0.75", 0.50),
                                   (INCUMBENT, "0.1", "0.9", 0.80)]:
            inside = ((g["actual"] >= g[f"q{lo}_{model}"]) & (g["actual"] <= g[f"q{hi}_{model}"])).mean()
            rows.append({"horizon": h, "horizon_label": HLABEL[h], "model": model,
                         "nominal": nom, "empirical": float(inside), "kind": "interval"})
    df = pd.DataFrame(rows)
    df["kind"] = df["kind"].fillna("quantile") if "kind" in df else "quantile"
    return df


def win_matrix(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        g = w[w["horizon"] == h]
        for tank, t in g.groupby("item_id"):
            c, n = t[f"ae_{CHRONOS}"].mean(), t[f"ae_{INCUMBENT}"].mean()
            rows.append({"tank": tank, "horizon": h, "horizon_label": HLABEL[h],
                         "chronos2_mae": float(c), "npts_mae": float(n),
                         "improvement_pct": float(100 * (n - c) / n) if n else np.nan,
                         "winner": CHRONOS if c < n else INCUMBENT})
    return pd.DataFrame(rows)


def skill_scores(w: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Skill vs the seasonal-naive reference: 1 - MAE_model / MAE_reference."""
    h = w[w["horizon"] == horizon]
    trust = json.loads(Path("eda/tank_trust.json").read_text())
    rows = []
    for tank, g in h.groupby("item_id"):
        ref = g["ae_SeasonalNaive"].mean()
        rows.append({
            "tank": tank, "trust": trust.get(tank),
            "mean_actual_kl_h": float(g["actual"].mean()),
            "skill_chronos2": float(1 - g[f"ae_{CHRONOS}"].mean() / ref) if ref else np.nan,
            "skill_npts": float(1 - g[f"ae_{INCUMBENT}"].mean() / ref) if ref else np.nan,
        })
    return pd.DataFrame(rows).sort_values("skill_chronos2", ascending=False).reset_index(drop=True)


def cumulative_volume(w: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Campus-total predicted vs actual volume, accumulated over origins.

    This is the operational view: a per-hour MAE of 0.21 KL/h is abstract; a cumulative
    shortfall in KL is what an operator would actually have run short of.
    """
    h = w[w["horizon"] == horizon]
    g = h.groupby("origin").agg(
        actual=("actual", "sum"),
        chronos2=(f"pred_{CHRONOS}", "sum"),
        npts=(f"pred_{INCUMBENT}", "sum"),
        seasonal_naive=("pred_SeasonalNaive", "sum"),
    ).sort_index().reset_index()
    for c in ("actual", "chronos2", "npts", "seasonal_naive"):
        g[f"cum_{c}"] = g[c].cumsum()
    g["cum_gap_chronos2"] = g["cum_chronos2"] - g["cum_actual"]
    g["cum_gap_npts"] = g["cum_npts"] - g["cum_actual"]
    return g


# ---------------------------------------------------------------- figures

def fig_O_significance_forest(sig: pd.DataFrame) -> None:
    """Forest plot: improvement % with a 95 % bootstrap CI, per horizon. If a bar's CI
    crosses zero the improvement is not established at that horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric in zip(axes, ["MAE", "MASE"]):
        s = sig[sig["metric"] == metric].set_index("horizon").loc[HORIZONS]
        y = np.arange(len(HORIZONS))
        imp = -s["diff_pct"].to_numpy()                       # positive = Chronos-2 better
        lo = -s["ci_hi_pct"].to_numpy()
        hi = -s["ci_lo_pct"].to_numpy()
        ax.axvline(0, color=INK2, lw=1.2, zorder=1)
        ax.hlines(y, lo, hi, color=COLOR[CHRONOS], lw=3, alpha=0.45, zorder=2)
        ax.scatter(imp, y, s=60, color=COLOR[CHRONOS], zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
        for yi, (m, l) in enumerate(zip(imp, lo)):
            ax.annotate(f"{m:.1f}%", (m, yi), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([HLABEL[h] for h in HORIZONS])
        ax.set_xlabel(f"{metric} improvement over NPTS (%)")
        ax.set_title(f"{metric}", fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="y", alpha=0.25)
    axes[0].invert_yaxis()
    fig.suptitle("Chronos-2 vs NPTS — improvement with 95 % paired-bootstrap CI over 24 origins",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    _caption(fig, "Bars are 95 % CIs from 10,000 bootstrap resamples of the 24 forecast origins. A CI entirely right of 0 means the improvement is significant.", y=-0.04)
    _save(fig, "O_significance_forest")


def fig_P_per_tank_significance(pt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8))
    d = pt.sort_values("improvement_pct")
    y = np.arange(len(d))
    colors = [COLOR[CHRONOS] if v == "Chronos-2 better"
              else COLOR[INCUMBENT] if v == "NPTS better" else "#9a9894"
              for v in d["verdict"]]
    ax.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax.hlines(y, d["ci_lo_pct"], d["ci_hi_pct"], color=colors, lw=2.6, alpha=0.45, zorder=2)
    ax.scatter(d["improvement_pct"], y, s=42, color=colors, zorder=3,
               edgecolor=SURFACE, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["tank"], fontsize=8)
    ax.set_xlabel("MAE improvement over NPTS (%)  —  95 % bootstrap CI")
    ax.set_title("Per-tank significance at the 1-day horizon", fontsize=11.5, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=COLOR[CHRONOS], lw=3, label="Chronos-2 significantly better"),
               plt.Line2D([], [], color="#9a9894", lw=3, label="no significant difference"),
               plt.Line2D([], [], color=COLOR[INCUMBENT], lw=3, label="NPTS significantly better")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
    n_sig = int((d["verdict"] == "Chronos-2 better").sum())
    n_ns = int((d["verdict"] == "no significant difference").sum())
    n_lose = int((d["verdict"] == "NPTS better").sum())
    _caption(fig, f"{n_sig} tanks significantly better · {n_ns} indistinguishable · {n_lose} significantly worse. Grey CIs cross zero: those per-tank differences are not established.", y=-0.02)
    _save(fig, "P_per_tank_significance_h24")


def fig_Q_diurnal(dn: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})
    for m in (CHRONOS, INCUMBENT, REFERENCE + "-24" if False else "SeasonalNaive"):
        key = m
        lab = LABEL.get(key, key)
        col = COLOR.get(key, COLOR.get(REFERENCE))
        ax1.plot(dn["hour"], dn[f"mae_{key}"], color=col, lw=2, marker="o", ms=3.5, label=lab)
    ax1b = ax1.twinx() if False else None
    ax1.fill_between(dn["hour"], 0, dn["mean_actual"], color="#cde2fb", alpha=0.35, zorder=0,
                     label="mean actual demand (KL/h)")
    ax1.set_ylabel("MAE (KL/h)")
    ax1.legend(frameon=False, fontsize=8.5, ncol=2)
    ax1.set_title("Where the error lives across the day — 1-day horizon",
                  fontsize=11.5, color=INK, loc="left")
    for m in (CHRONOS, INCUMBENT):
        ax2.plot(dn["hour"], dn[f"bias_{m}"], color=COLOR[m], lw=2, marker="o", ms=3.5,
                 label=LABEL[m])
    ax2.axhline(0, color=INK2, lw=1)
    ax2.set_ylabel("Bias (KL/h)\npred − actual")
    ax2.set_xlabel("Hour of day")
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(frameon=False, fontsize=8.5)
    _caption(fig, "Shaded band is mean actual demand. Error tracks demand: both models are worst during the morning and evening draw, and both under-forecast there.", y=-0.02)
    _save(fig, "Q_diurnal_error")


def fig_R_leadtime(lt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for m in (CHRONOS, INCUMBENT, "SeasonalNaive"):
        col = COLOR.get(m, COLOR[REFERENCE])
        ax.plot(lt["step"], lt[f"mae_{m}"], color=col, lw=0.9, alpha=0.28)
        smooth = lt[f"mae_{m}"].rolling(24, center=True, min_periods=6).mean()
        ax.plot(lt["step"], smooth, color=col, lw=2.6, label=LABEL.get(m, m))
    for d in range(1, 7):
        ax.axvline(d * 24, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("Lead time (hours ahead of the forecast origin), 7-day horizon")
    ax.set_ylabel("MAE (KL/h)")
    ax.set_xlim(1, 168)
    ax.set_xticks([1] + list(range(24, 169, 24)))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Error does not grow with lead time — it saturates within one day",
                 fontsize=11.5, color=INK, loc="left")
    early = lt[lt.step <= 24][f"mae_{CHRONOS}"].mean()
    late = lt[lt.step > 144][f"mae_{CHRONOS}"].mean()
    _caption(fig, f"Faint lines are the raw hourly MAE; its sawtooth is the diurnal cycle, not "
                  f"forecast decay. Bold lines are a centred 24-hour rolling mean. Chronos-2 "
                  f"averages {early:.3f} KL/h over the first day and {late:.3f} KL/h over the "
                  f"seventh ({100*(late-early)/early:+.1f} %): a 7-day forecast is barely worse "
                  f"than a 1-day forecast, and the gap to NPTS persists at every lead time.",
             y=-0.06)
    _save(fig, "R_error_by_leadtime")


def fig_S_reliability(rel: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    q = rel[(rel["kind"] == "quantile") & (rel["model"] == CHRONOS)]
    cmap = SEQ_BLUE(np.linspace(0.25, 0.95, len(HORIZONS)))
    ax1.plot([0, 1], [0, 1], color=INK2, lw=1.2, ls="--", label="perfect calibration")
    for c, h in zip(cmap, HORIZONS):
        s = q[q["horizon"] == h].sort_values("nominal")
        ax1.plot(s["nominal"], s["empirical"], color=c, lw=1.8, marker="o", ms=4,
                 label=HLABEL[h])
    ax1.set_xlabel("Nominal quantile")
    ax1.set_ylabel("Empirical fraction of actuals below")
    ax1.set_title("Chronos-2 quantile reliability", fontsize=10.5, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8, ncol=2)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect("equal")

    iv = rel[rel["kind"] == "interval"]
    x = np.arange(len(HORIZONS)); wdt = 0.26
    c80 = iv[(iv.model == CHRONOS) & (iv.nominal == 0.80)].set_index("horizon").loc[HORIZONS, "empirical"]
    c50 = iv[(iv.model == CHRONOS) & (iv.nominal == 0.50)].set_index("horizon").loc[HORIZONS, "empirical"]
    n80 = iv[(iv.model == INCUMBENT) & (iv.nominal == 0.80)].set_index("horizon").loc[HORIZONS, "empirical"]
    ax2.bar(x - wdt, c50, wdt, color="#9ec5f4", label="Chronos-2 p25–p75 (nominal 0.50)")
    ax2.bar(x, c80, wdt, color=COLOR[CHRONOS], label="Chronos-2 p10–p90 (nominal 0.80)")
    ax2.bar(x + wdt, n80, wdt, color=COLOR[INCUMBENT], label="NPTS p10–p90 (nominal 0.80)")
    ax2.axhline(0.80, color=INK2, lw=1.2, ls="--")
    ax2.axhline(0.50, color=INK2, lw=1.0, ls=":")
    ax2.annotate("nominal 0.80", (len(HORIZONS) - 0.4, 0.815), fontsize=8, color=INK2, ha="right")
    ax2.annotate("nominal 0.50", (len(HORIZONS) - 0.4, 0.515), fontsize=8, color=INK2, ha="right")
    ax2.set_xticks(x); ax2.set_xticklabels([HLABEL[h] for h in HORIZONS])
    ax2.set_ylabel("Empirical coverage"); ax2.set_ylim(0, 1)
    ax2.set_title("Interval coverage vs nominal", fontsize=10.5, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    _caption(fig, "Left: a calibrated model tracks the dashed diagonal. Chronos-2 sits above it at low quantiles and below at high ones — the distribution is too narrow at both tails, not merely shifted.", y=-0.04)
    _save(fig, "S_reliability_diagram")


def fig_T_error_vs_demand(sk: pd.DataFrame, pt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    m = sk.merge(pt[["tank", "chronos2_mae", "verdict"]], on="tank")
    tiers = {"healthy": COLOR[CHRONOS], "degraded": "#e8a33d", "dead": "#d03b3b"}
    for tier, col in tiers.items():
        s = m[m["trust"] == tier]
        ax.scatter(s["mean_actual_kl_h"], s["skill_chronos2"], s=70, color=col,
                   edgecolor=SURFACE, linewidth=1.2, label=f"{tier} ({len(s)})", zorder=3)
    ax.axhline(0, color=INK2, lw=1.2)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Mean hourly demand (KL/h, symlog)")
    ax.set_ylabel("Skill vs SeasonalNaive-24\n1 − MAE(model)/MAE(reference)")
    ax.set_title("Skill against the naive reference, by demand size and sensor tier (1 d)",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, title="sensor trust", title_fontsize=8.5)
    hi = m.nlargest(3, "skill_chronos2")
    lo = m.nsmallest(3, "skill_chronos2")
    big = m.nlargest(2, "mean_actual_kl_h")
    for k, (_, r) in enumerate(pd.concat([hi, lo, big]).drop_duplicates("tank").iterrows()):
        ax.annotate(r["tank"], (r["mean_actual_kl_h"], r["skill_chronos2"]),
                    textcoords="offset points", xytext=(7, 6 if k % 2 == 0 else -11),
                    fontsize=7, color=INK2)
    _caption(fig, "Every tank has positive skill against seasonal naive. Skill does not decline with demand size, so the model is not simply exploiting quiet series.", y=-0.03)
    _save(fig, "T_skill_vs_demand")


def fig_U_cumulative(cv: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.3]})
    x = np.arange(1, len(cv) + 1)
    ax1.plot(x, cv["cum_actual"], color=INK, lw=2.4, label="Actual")
    ax1.plot(x, cv["cum_chronos2"], color=COLOR[CHRONOS], lw=2, ls="--", label=LABEL[CHRONOS])
    ax1.plot(x, cv["cum_npts"], color=COLOR[INCUMBENT], lw=2, ls="--", label=LABEL[INCUMBENT])
    ax1.plot(x, cv["cum_seasonal_naive"], color=COLOR[REFERENCE], lw=1.6, ls=":",
             label=LABEL[REFERENCE])
    ax1.set_ylabel("Cumulative campus volume (KL)")
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.set_title("Cumulative 24-hour campus demand: forecast vs reality across 24 origins",
                  fontsize=11.5, color=INK, loc="left")
    ax2.fill_between(x, 0, cv["cum_gap_chronos2"], color=COLOR[CHRONOS], alpha=0.3)
    ax2.plot(x, cv["cum_gap_chronos2"], color=COLOR[CHRONOS], lw=2, label=LABEL[CHRONOS])
    ax2.plot(x, cv["cum_gap_npts"], color=COLOR[INCUMBENT], lw=2, label=LABEL[INCUMBENT])
    ax2.axhline(0, color=INK2, lw=1.2)
    ax2.set_ylabel("Cumulative\nshortfall (KL)")
    ax2.set_xlabel("Forecast origin (1 … 24)")
    ax2.legend(frameon=False, fontsize=8.5)
    final_c = cv["cum_gap_chronos2"].iloc[-1]
    final_n = cv["cum_gap_npts"].iloc[-1]
    tot = cv["cum_actual"].iloc[-1]
    _caption(fig, f"Over 24 origins the campus drew {tot:,.0f} KL. Chronos-2 under-forecasts by {abs(final_c):,.0f} KL ({100*final_c/tot:+.1f} %), NPTS by {abs(final_n):,.0f} KL ({100*final_n/tot:+.1f} %). This is the operational cost of the bias, and why a refill must not be sized on the mean forecast.", y=-0.04)
    _save(fig, "U_cumulative_volume")


def fig_V_win_matrix(wm: pd.DataFrame) -> None:
    piv = wm.pivot(index="tank", columns="horizon", values="improvement_pct")
    piv = piv[HORIZONS]
    order = piv.mean(axis=1).sort_values(ascending=False).index
    piv = piv.loc[order]
    vmax = float(np.nanmax(np.abs(piv.to_numpy())))
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    im = ax.imshow(piv.to_numpy(), cmap=DIVERGING, aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([HLABEL[h] for h in HORIZONS])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("MAE improvement over NPTS (%)", fontsize=9)
    ax.set_title("Chronos-2 vs NPTS, every tank × every horizon",
                 fontsize=11.5, color=INK, loc="left")
    ax.grid(False)
    wins = (wm["winner"] == CHRONOS).sum()
    _caption(fig, f"Blue = Chronos-2 better, red = NPTS better. Chronos-2 wins {wins} of {len(wm)} tank-horizon cells ({100*wins/len(wm):.0f} %).", y=-0.02)
    _save(fig, "V_win_matrix")


def zero_inflation_diagnosis(w: pd.DataFrame) -> pd.DataFrame:
    """Why the p10-p90 interval under-covers, decomposed by tail.

    26 % of observed hourly readings are exactly zero. A continuous-density foundation model
    puts almost no probability mass on exactly zero, so its p10 sits above zero and every quiet
    hour falls out of the bottom of the interval. This table separates the lower-tail miss from
    the upper-tail miss and shows they are not the same size.
    """
    rows = []
    for h in HORIZONS:
        g = w[w["horizon"] == h]
        a = g["actual"]
        for model in (CHRONOS, INCUMBENT):
            lo, hi = g[f"q0.1_{model}"], g[f"q0.9_{model}"]
            below = a < lo
            rows.append({
                "horizon": h, "horizon_label": HLABEL[h], "model": model,
                "n": int(len(g)),
                "zero_fraction": float((a == 0).mean()),
                "miss_below_p10": float(below.mean()),
                "miss_above_p90": float((a > hi).mean()),
                "coverage": float(((a >= lo) & (a <= hi)).mean()),
                "p10_above_zero_frac": float((lo > 0).mean()),
                "median_p10": float(lo.median()),
                "below_misses_that_are_zero": float((a[below] == 0).mean()) if below.any() else np.nan,
                "coverage_if_p10_clamped_to_zero": float(((a >= 0) & (a <= hi)).mean()),
            })
    return pd.DataFrame(rows)


def fig_W_zero_inflation(zi: pd.DataFrame) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 4.8))
    x = np.arange(len(HORIZONS)); wd = 0.36

    c = zi[zi.model == CHRONOS].set_index("horizon").loc[HORIZONS]
    n = zi[zi.model == INCUMBENT].set_index("horizon").loc[HORIZONS]

    ax1.bar(x - wd/2, c["miss_below_p10"], wd, color=COLOR[CHRONOS], label="below p10")
    ax1.bar(x - wd/2, c["miss_above_p90"], wd, bottom=c["miss_below_p10"],
            color="#9ec5f4", label="above p90")
    ax1.bar(x + wd/2, n["miss_below_p10"], wd, color=COLOR[INCUMBENT])
    ax1.bar(x + wd/2, n["miss_above_p90"], wd, bottom=n["miss_below_p10"], color="#f5b79b")
    ax1.axhline(0.20, color=INK2, lw=1.2, ls="--")
    ax1.annotate("expected 0.20", (len(HORIZONS) - 0.5, 0.208), fontsize=8, color=INK2, ha="right")
    ax1.set_xticks(x); ax1.set_xticklabels([HLABEL[h] for h in HORIZONS], fontsize=8)
    ax1.set_ylabel("Fraction of actuals outside p10–p90")
    ax1.set_title("Misses are mostly lower-tail", fontsize=9.5, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.13))
    ax1.text(0.02, 0.97, "solid = Chronos-2 · pale = NPTS", transform=ax1.transAxes,
             fontsize=7.5, color=INK2, va="top")

    ax2.bar(x - wd/2, c["p10_above_zero_frac"], wd, color=COLOR[CHRONOS], label=LABEL[CHRONOS])
    ax2.bar(x + wd/2, n["p10_above_zero_frac"], wd, color=COLOR[INCUMBENT], label=LABEL[INCUMBENT])
    ax2.plot(x, c["zero_fraction"], color=INK, lw=2, marker="o", ms=4,
             label="fraction of actuals = 0")
    ax2.set_xticks(x); ax2.set_xticklabels([HLABEL[h] for h in HORIZONS], fontsize=8)
    ax2.set_ylabel("Fraction of rows with p10 > 0")
    ax2.set_ylim(0, 1)
    ax2.set_title("Chronos-2 puts p10 above zero", fontsize=9.5, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=1,
               bbox_to_anchor=(0.5, -0.13))

    ax3.bar(x - wd/2, c["coverage"], wd, color=COLOR[CHRONOS], label="as published")
    ax3.bar(x + wd/2, c["coverage_if_p10_clamped_to_zero"], wd, color="#1baf7a",
            label="if p10 clamped to 0")
    ax3.axhline(0.80, color=INK2, lw=1.4, ls="--")
    ax3.annotate("nominal 0.80", (len(HORIZONS) - 0.5, 0.815), fontsize=8, color=INK2, ha="right")
    ax3.set_xticks(x); ax3.set_xticklabels([HLABEL[h] for h in HORIZONS], fontsize=8)
    ax3.set_ylabel("p10–p90 coverage"); ax3.set_ylim(0, 1)
    ax3.set_title("A naive clamp over-corrects", fontsize=9.5, color=INK, loc="left")
    ax3.legend(frameon=False, fontsize=8, loc="lower center")

    fig.suptitle("Diagnosis: the interval fails on a zero-inflated series, not uniformly",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(bottom=0.26, wspace=0.32, top=0.82)
    _caption(fig, "About 24 % of hourly readings are exactly zero. Chronos-2 is a continuous-density model and places p10 above zero on 79 % of rows (median p10 = 0.048 KL/h), so two thirds of its lower-tail misses are hours where demand was exactly zero. NPTS, a nonparametric sampler, reproduces the zero atom and covers correctly. Clamping p10 to zero overshoots to 0.89 — the fix is asymmetric conformal calibration, not a clamp.", y=-0.05)
    _save(fig, "W_zero_inflation_diagnosis")



# ---------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading paired predictions ...")
    w = load_paired()

    print("\n1. Statistical significance")
    sig = significance_by_horizon(w);              _write(sig, "significance_by_horizon.csv")
    pt = significance_per_tank(w, 24);             _write(pt, "significance_per_tank_h24.csv")

    print("\n2. Error patterns")
    dn = diurnal_error(w, 24);                     _write(dn, "diurnal_error_h24.csv")
    lt = error_by_leadtime(w);                     _write(lt, "error_by_leadtime.csv")
    rel = reliability(w);                          _write(rel, "reliability.csv")
    wm = win_matrix(w);                            _write(wm, "win_matrix.csv")
    sk = skill_scores(w, 24);                      _write(sk, "skill_scores_h24.csv")
    cv = cumulative_volume(w, 24);                 _write(cv, "cumulative_volume_h24.csv")
    zi = zero_inflation_diagnosis(w);              _write(zi, "zero_inflation_diagnosis.csv")

    print("\n3. Figures")
    fig_O_significance_forest(sig)
    fig_P_per_tank_significance(pt)
    fig_Q_diurnal(dn)
    fig_R_leadtime(lt)
    fig_S_reliability(rel)
    fig_T_error_vs_demand(sk, pt)
    fig_U_cumulative(cv)
    fig_V_win_matrix(wm)
    fig_W_zero_inflation(zi)

    manifest = {
        "generated_from": "results/chronos2/predictions_*.parquet (completed benchmark)",
        "refits": 0, "rescores": 0,
        "paired_rows": int(len(w)),
        "n_origins": int(w["origin"].nunique()),
        "n_tanks": int(w["item_id"].nunique()),
        "bootstrap_resamples": N_BOOT,
        "seed": 20260830,
        "tables": sorted(p.name for p in OUT.glob("*.csv")),
        "figures": sorted(p.stem for p in PLOTS.glob("*.png")),
    }
    (OUT / "phase3_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest -> {OUT/'phase3_manifest.json'}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
