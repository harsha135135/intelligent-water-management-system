"""Review-ready figures built from the COMPLETED benchmark. Re-scores nothing.

Colour follows the validated categorical palette (blue / orange / aqua), assigned in fixed
order by *entity*, never by rank, so a model keeps its colour in every figure:

    Chronos-2  #2a78d6   NPTS  #eb6834   SeasonalNaive-24  #1baf7a

Validated with the palette checker at all-pairs, light surface: worst CVD deltaE 9.2,
worst normal-vision deltaE 24.0, all gates PASS. Aqua sits below 3:1 contrast on the light
surface, so the relief rule applies - every figure carries a legend, series are direct-labelled
where they do not collide, and each figure has an accompanying CSV table.

Magnitude uses the single-hue blue sequential ramp; the signed improvement chart uses the
blue<->red diverging pair with a neutral gray midpoint. No rainbow, no dual axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .review_package import (
    CHRONOS, HLABEL, HORIZONS, INCUMBENT, OUT, PLOTS, REFERENCE,
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

COLOR = {CHRONOS: "#2a78d6", INCUMBENT: "#eb6834", REFERENCE: "#1baf7a"}
LABEL = {CHRONOS: "Chronos-2 (zero-shot)", INCUMBENT: "NPTS (incumbent)",
         REFERENCE: "SeasonalNaive-24"}

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_br", ["#d03b3b", "#e8a0a0", "#f0efec", "#86b6ef", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  [plot] {name}.png / .svg")


def _style_horizon_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([HLABEL[h] for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.grid(axis="x", alpha=0.4)


# ── A-D: metric vs horizon ───────────────────────────────────────────────────

def metric_vs_horizon(mac: pd.DataFrame, metric: str, title: str, ylabel: str,
                      name: str, ref_line: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for model in (CHRONOS, INCUMBENT, REFERENCE):
        s = mac[mac["model"] == model].sort_values("horizon")
        ax.plot(s["horizon"], s[metric], color=COLOR[model], lw=2,
                marker="o", ms=8, mfc=COLOR[model], mec=SURFACE, mew=2,
                label=LABEL[model], zorder=3)
        # Direct label at the right end so identity is never colour-alone.
        ax.annotate(LABEL[model].split(" (")[0],
                    (s["horizon"].iloc[-1], s[metric].iloc[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    color=COLOR[model], fontsize=9, va="center", fontweight="600")

    if ref_line:
        ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        # Anchored at the right end: at the left end the seasonal-naive curve sits on 1.0 and
        # the label would land on top of it.
        ax.annotate("1.0 = as accurate as seasonal naive", (168, 1.0),
                    xytext=(0, -14), textcoords="offset points", ha="right",
                    fontsize=8.5, color=INK2)

    _style_horizon_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="600", color=INK, loc="left", pad=12)
    # Below the axes: the lower-right corner is where the direct labels live.
    ax.legend(frameon=False, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.26), ncol=3)
    ax.set_xlim(5.2, 260)
    _save(fig, name)


# ── E: per-tank MASE at 24h ──────────────────────────────────────────────────

def per_tank_mase(cmp3: pd.DataFrame, horizon: int = 24) -> None:
    d = cmp3[cmp3["horizon"] == horizon].sort_values("mase_chronos2")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 9))
    bh = 0.38
    ax.barh(y + bh / 2, d["mase_chronos2"], height=bh, color=COLOR[CHRONOS],
            label=LABEL[CHRONOS], zorder=3)
    ax.barh(y - bh / 2, d["mase_npts"], height=bh, color=COLOR[INCUMBENT],
            label=LABEL[INCUMBENT], zorder=3)
    ax.axvline(1.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=4)
    # Anchored low: the top rows are the worst tanks and their bars run past x = 1.0.
    ax.annotate("MASE 1.0\nseasonal naive", (1.0, 1.2), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    ax.set_yticks(y, d["item_id"], fontsize=8.5)
    ax.set_xlabel("MASE (lower is better)")
    ax.set_title(f"Per-tank accuracy at {HLABEL[horizon]} — every tank shown, including "
                 f"where Chronos-2 loses",
                 fontsize=13, fontweight="600", color=INK, loc="left", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    _save(fig, f"E_per_tank_mase_{HLABEL[horizon]}")


# ── F: per-tank improvement ──────────────────────────────────────────────────

def per_tank_improvement(cmp3: pd.DataFrame, horizon: int = 24) -> None:
    d = cmp3[cmp3["horizon"] == horizon].sort_values("pct_impr_mase")
    vals = d["pct_impr_mase"].to_numpy()
    lim = max(abs(vals.min()), abs(vals.max()))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(np.arange(len(d)), vals, color=DIVERGING(norm(vals)), zorder=3)
    ax.axvline(0, color=INK2, lw=1.2, zorder=4)
    ax.set_yticks(np.arange(len(d)), d["item_id"], fontsize=8.5)
    ax.set_xlabel("MASE reduction vs NPTS (%)   ←  NPTS better | Chronos-2 better  →")
    ax.set_title(f"Where Chronos-2 helps and where it hurts, at {HLABEL[horizon]}",
                 fontsize=13, fontweight="600", color=INK, loc="left", pad=12)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.0f}%", (v, i), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    _save(fig, f"F_per_tank_improvement_{HLABEL[horizon]}")


# ── G: heatmap ───────────────────────────────────────────────────────────────

def per_tank_heatmap(pt: pd.DataFrame) -> None:
    d = pt[pt["model"] == CHRONOS].pivot(index="item_id", columns="horizon", values="mase")
    d = d.reindex(columns=HORIZONS).sort_values(24)
    fig, ax = plt.subplots(figsize=(8.5, 9))
    im = ax.imshow(d.to_numpy(), aspect="auto", cmap=SEQ_BLUE, vmin=0,
                   vmax=float(np.nanmax(d.to_numpy())))
    ax.set_xticks(range(len(HORIZONS)), [HLABEL[h] for h in HORIZONS])
    ax.set_yticks(range(len(d)), d.index, fontsize=8.5)
    hi = float(np.nanmax(d.to_numpy()))
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > hi * 0.55 else INK)
    ax.set_title("Chronos-2 MASE by tank and horizon\ndarker = worse; values above 1.00 "
                 "lose to seasonal naive",
                 fontsize=13, fontweight="600", color=INK, loc="left", pad=12)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, label="MASE", fraction=0.035, pad=0.03)
    _save(fig, "G_per_tank_mase_heatmap")


# ── H: error distribution ────────────────────────────────────────────────────

def error_distribution(scored: pd.DataFrame, horizon: int = 24) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    bins = np.linspace(-2, 2, 81)
    for model in (CHRONOS, INCUMBENT):
        e = scored[(scored["model"] == model) & (scored["horizon"] == horizon)]["error"]
        axes[0].hist(e, bins=bins, histtype="step", lw=2, color=COLOR[model],
                     label=f"{LABEL[model]}  (n={len(e):,})", density=True, zorder=3)
    axes[0].axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
    axes[0].set_xlabel("Forecast error (actual − predicted), KL/h")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Error distribution at {HLABEL[horizon]}",
                      fontsize=12, fontweight="600", color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=9)

    qs = np.arange(0.5, 100, 0.5)
    for model in (CHRONOS, INCUMBENT):
        e = scored[(scored["model"] == model) & (scored["horizon"] == horizon)]["abs_error"]
        axes[1].plot(qs, np.percentile(e, qs), color=COLOR[model], lw=2,
                     label=LABEL[model], zorder=3)
    axes[1].set_xlabel("percentile of predictions")
    axes[1].set_ylabel("absolute error (KL/h)")
    axes[1].set_yscale("log")
    axes[1].set_title("Absolute error by percentile — the tail is what matters operationally",
                      fontsize=12, fontweight="600", color=INK, loc="left")
    axes[1].legend(frameon=False, fontsize=9)
    _save(fig, f"H_error_distribution_{HLABEL[horizon]}")


# ── I: actual vs predicted, representative tanks ─────────────────────────────

def pick_representatives(pt: pd.DataFrame, trust: dict) -> dict:
    """Choose four tanks by measured role, not by hand, so the panel cannot be cherry-picked."""
    d = pt[(pt["model"] == CHRONOS) & (pt["horizon"] == 24)].copy()
    live = d[d["item_id"].map(lambda t: trust.get(t) != "dead")]
    ranked = live.sort_values("mean_actual", ascending=False)
    hardest = live.sort_values("mase", ascending=False).iloc[0]["item_id"]
    pool = [t for t in ranked["item_id"] if t != hardest]
    return {
        "high demand": pool[0],
        "medium demand": pool[len(pool) // 2],
        "low demand": pool[-1],
        "hardest (highest MASE)": hardest,
    }


def actual_vs_pred(scored: pd.DataFrame, reps: dict, horizon: int = 24) -> None:
    origin = scored["origin"].max()
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for ax, (role, item) in zip(axes.ravel(), reps.items()):
        s = scored[(scored["item_id"] == item) & (scored["horizon"] == horizon)
                   & (scored["origin"] == origin)]
        a = s[s["model"] == CHRONOS].sort_values("timestamp")
        n = s[s["model"] == INCUMBENT].sort_values("timestamp")
        t = np.arange(len(a))
        ax.fill_between(t, a["0.1"], a["0.9"], color=COLOR[CHRONOS], alpha=0.16,
                        lw=0, label="Chronos-2 p10–p90", zorder=2)
        ax.plot(t, a["actual"], color=INK, lw=2, label="Actual", zorder=5)
        ax.plot(t, a["pred"], color=COLOR[CHRONOS], lw=2, label=LABEL[CHRONOS], zorder=4)
        ax.plot(t, n["pred"], color=COLOR[INCUMBENT], lw=2, ls=(0, (5, 2)),
                label=LABEL[INCUMBENT], zorder=3)
        mase = float(a["abs_error"].sum() / a["scale_mae"].sum()) if len(a) else np.nan
        ax.set_title(f"{item}  ·  {role}\nmean {a['actual'].mean():.2f} KL/h  ·  "
                     f"Chronos-2 MASE {mase:.2f}",
                     fontsize=10.5, fontweight="600", color=INK, loc="left")
        ax.set_xlabel("hours after forecast origin")
        ax.set_ylabel("outflow (KL/h)")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"Actual vs predicted — {HLABEL[horizon]} forecast from the final origin "
                 f"({origin:%Y-%m-%d %H:%M})",
                 fontsize=13, fontweight="600", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, f"I_actual_vs_predicted_{HLABEL[horizon]}")


# ── J: final 7-day holdout ───────────────────────────────────────────────────

def final_holdout_plot(scored: pd.DataFrame, reps: dict) -> None:
    origin = scored["origin"].max()
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for ax, (role, item) in zip(axes.ravel(), reps.items()):
        s = scored[(scored["item_id"] == item) & (scored["horizon"] == 168)
                   & (scored["origin"] == origin)]
        a = s[s["model"] == CHRONOS].sort_values("timestamp")
        n = s[s["model"] == INCUMBENT].sort_values("timestamp")
        ts = a["timestamp"]
        ax.fill_between(ts, a["0.1"], a["0.9"], color=COLOR[CHRONOS], alpha=0.16,
                        lw=0, label="Chronos-2 p10–p90", zorder=2)
        ax.plot(ts, a["actual"], color=INK, lw=1.6, label="Actual", zorder=5)
        ax.plot(ts, a["pred"], color=COLOR[CHRONOS], lw=1.8, label=LABEL[CHRONOS], zorder=4)
        ax.plot(n["timestamp"], n["pred"], color=COLOR[INCUMBENT], lw=1.6, ls=(0, (5, 2)),
                label=LABEL[INCUMBENT], zorder=3)
        for d in pd.date_range(ts.min().normalize(), ts.max(), freq="D"):
            ax.axvline(d, color=GRID, lw=0.8, zorder=1)
        ax.set_title(f"{item}  ·  {role}", fontsize=10.5, fontweight="600",
                     color=INK, loc="left")
        ax.set_ylabel("outflow (KL/h)")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("FINAL HOLDOUT — 7-day forecast over the last contiguous week of data "
                 f"({origin + pd.Timedelta(hours=1):%d %b} – "
                 f"{origin + pd.Timedelta(hours=168):%d %b %Y})",
                 fontsize=13, fontweight="600", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "J_final_holdout_7day")


# ── K: calibration (supports the uncertainty section) ────────────────────────

def calibration(mac: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for model in (CHRONOS, INCUMBENT, REFERENCE):
        s = mac[mac["model"] == model].sort_values("horizon")
        ax.plot(s["horizon"], s["p10_p90_coverage"], color=COLOR[model], lw=2,
                marker="o", ms=8, mfc=COLOR[model], mec=SURFACE, mew=2,
                label=LABEL[model], zorder=3)
        ax.annotate(LABEL[model].split(" (")[0],
                    (s["horizon"].iloc[-1], s["p10_p90_coverage"].iloc[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    color=COLOR[model], fontsize=9, va="center", fontweight="600")
    ax.axhline(0.80, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.annotate("nominal 0.80", (6, 0.80), xytext=(0, 6), textcoords="offset points",
                fontsize=8.5, color=INK2)
    _style_horizon_axis(ax)
    ax.set_ylabel("empirical p10–p90 coverage")
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(5.2, 300)
    ax.set_title("Interval calibration — Chronos-2 is overconfident, NPTS is well calibrated",
                 fontsize=13, fontweight="600", color=INK, loc="left", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _save(fig, "K_interval_calibration")


# ── L: model-variant selection ───────────────────────────────────────────────

def variant_selection(mac: pd.DataFrame, runtimes: dict) -> None:
    variants = [v for v in ("Chronos2-ZS", "Chronos2-COV", "Chronos2-COV-LEAN",
                            "Chronos2-COV-XL") if v in set(mac["model"])]
    d = mac[(mac["model"].isin(variants)) & (mac["horizon"] == 6)].set_index("model")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = [runtimes[v] for v in variants]
    y = [d.loc[v, "mase"] for v in variants]
    ax.scatter(x, y, s=150, color=COLOR[CHRONOS], edgecolor=SURFACE, linewidth=2, zorder=3)
    for v, xi, yi in zip(variants, x, y):
        ax.annotate(f"{v.replace('Chronos2-', '')}\n{yi:.4f} · {xi:.1f} min",
                    (xi, yi), xytext=(8, 8), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.set_xlabel("backtest runtime (minutes, Apple MPS)")
    ax.set_ylabel("macro MASE at 6h")
    spread = float(d["mase"].max() - d["mase"].min())
    ax.set_title("Accuracy bought per minute of compute\n"
                 f"the whole spread across the four variants is {spread:.4f} MASE "
                 f"({100 * spread / float(d['mase'].max()):.2f}%) — zero-shot is the rational "
                 "default",
                 fontsize=13, fontweight="600", color=INK, loc="left", pad=12)
    ax.margins(0.18)
    # This axis is deliberately zoomed to separate four near-identical points, so the zoom is
    # named rather than left for the reader to discover.
    lo, hi = ax.get_ylim()
    npts = mac[(mac["model"] == INCUMBENT) & (mac["horizon"] == 6)]
    note = f"note: the y-axis spans only {hi - lo:.4f} MASE"
    if not npts.empty:
        note += (f"; {LABEL[INCUMBENT]} scores {float(npts['mase'].iloc[0]):.4f} at this "
                 f"horizon, far above the top of this chart")
    ax.annotate(note, (0.0, -0.16), xycoords="axes fraction", fontsize=8.5, color=INK2)
    _save(fig, "L_variant_selection")


# ── M: how many tanks does Chronos-2 actually win? ───────────────────────────

def tanks_won(cmp_all: pd.DataFrame) -> None:
    """Part-to-whole: of the 24 tanks, how many does each model win at each horizon?

    This is the figure that stops the macro average from doing the arguing. The win is decided
    on MASE, and every tank is counted - there is no "excluding dead sensors" bar.
    """
    horizons = [h for h in HORIZONS if h in set(cmp_all["horizon"])]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(horizons))[::-1]
    total = 0

    for i, h in enumerate(horizons):
        d = cmp_all[cmp_all["horizon"] == h]
        total = int(d["winner"].notna().sum())
        counts = {CHRONOS: int((d["winner"] == "Chronos-2").sum()),
                  INCUMBENT: int((d["winner"] == "NPTS").sum()),
                  REFERENCE: int((d["winner"] == "SeasonalNaive").sum())}
        left = 0.0
        for model, count in counts.items():
            if count == 0:
                continue
            ax.barh(y[i], count, left=left, height=0.62, color=COLOR[model],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{count}  ({100 * count / total:.0f}%)",
                        (left + count / 2, y[i]), ha="center", va="center",
                        fontsize=9.5, fontweight="600", color="white", zorder=4)
            left += count

    ax.set_yticks(y, [HLABEL[h] for h in horizons], fontsize=10)
    ax.set_xlabel(f"number of tanks won, out of {total} (winner decided on MASE)")
    ax.set_ylabel("forecast horizon")
    ax.set_xlim(0, total)
    ax.set_title("M · How many tanks each model actually wins\n"
                 "Chronos-2 wins the majority at every horizon — but never all of them",
                 fontsize=13, fontweight="600", color=INK, loc="left", pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in
               (CHRONOS, INCUMBENT, REFERENCE)]
    ax.legend(handles, [LABEL[m] for m in (CHRONOS, INCUMBENT, REFERENCE)],
              frameon=False, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), ncol=3)
    ax.annotate("SeasonalNaive-24 wins 0 tanks at every horizon — it is the reference "
                "floor, not a contender",
                (0.5, -0.24), xycoords="axes fraction", ha="center", fontsize=8.5,
                color=INK2)
    ax.grid(axis="y", visible=False)
    _save(fig, "M_tanks_won")


# ── N: covariates vs zero-shot ───────────────────────────────────────────────

def covariate_vs_zeroshot(mac: pd.DataFrame, runtimes: dict) -> None:
    """Does conditioning on covariates buy accuracy worth its compute?

    The left panel plots the *difference* from zero-shot rather than four absolute curves: at
    this scale the absolute lines are indistinguishable, and overlaying them would imply a
    precision the numbers do not have. The three covariate variants are shaded along the
    single-hue blue ramp in order of compute cost, so darker = more expensive.
    """
    variants = [v for v in ("Chronos2-COV-LEAN", "Chronos2-COV-XL", "Chronos2-COV")
                if v in set(mac["model"]) and v in runtimes]
    variants.sort(key=lambda v: runtimes[v])
    if not variants or CHRONOS not in set(mac["model"]):
        return
    shades = [SEQ_BLUE(x) for x in np.linspace(0.35, 0.9, len(variants))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    zs = mac[mac["model"] == CHRONOS].set_index("horizon")["mase"]
    hs = [h for h in HORIZONS if h in zs.index]
    x = np.arange(len(hs))
    width = 0.8 / len(variants)

    for k, (v, colr) in enumerate(zip(variants, shades)):
        s = mac[mac["model"] == v].set_index("horizon")["mase"]
        delta = [zs[h] - s[h] for h in hs]          # positive = better than zero-shot
        pos = x + (k - (len(variants) - 1) / 2) * width
        axes[0].bar(pos, delta, width=width * 0.92, color=colr, zorder=3,
                    label=f"{v.replace('Chronos2-', '')}  ({runtimes[v]:.1f} min)")
        for xi, dv in zip(pos, delta):
            axes[0].annotate(f"{dv:+.4f}", (xi, dv),
                             xytext=(0, 4 if dv >= 0 else -4), textcoords="offset points",
                             ha="center", va="bottom" if dv >= 0 else "top",
                             fontsize=7, color=INK2, rotation=90)
    axes[0].axhline(0, color=COLOR[CHRONOS], lw=1.8, zorder=4)
    axes[0].annotate(f"zero line = Chronos-2 zero-shot "
                     f"({runtimes.get(CHRONOS, float('nan')):.1f} min)",
                     (0.99, 0.97), xycoords="axes fraction", ha="right", va="top",
                     fontsize=8.5, color=COLOR[CHRONOS], fontweight="600")
    axes[0].set_xticks(x, [HLABEL[h] for h in hs])
    axes[0].set_xlabel("forecast horizon")
    axes[0].set_ylabel("MASE reduction vs zero-shot\n(positive = covariates help)")
    axes[0].set_title("Accuracy gained by adding covariates",
                      fontsize=12, fontweight="600", color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0].margins(y=0.35)

    order = [CHRONOS] + variants
    cols = [COLOR[CHRONOS]] + shades
    for v, colr in zip(order, cols):
        s = mac[(mac["model"] == v) & (mac["horizon"] == 168)]
        if s.empty or v not in runtimes:
            continue
        axes[1].scatter(runtimes[v], float(s["mase"].iloc[0]), s=170, color=colr,
                        edgecolor=SURFACE, linewidth=2, zorder=3)
        axes[1].annotate(f"{v.replace('Chronos2-', '')}\n{float(s['mase'].iloc[0]):.4f}",
                         (runtimes[v], float(s["mase"].iloc[0])), xytext=(9, 6),
                         textcoords="offset points", fontsize=8.5, color=INK2)
    # The variant spread is ~0.003 MASE. Plotted on its own it fills the axis and looks
    # decisive, so the axis is extended to include the incumbent: the honest comparison is
    # "how big is this spread next to the gap we are actually claiming".
    npts = mac[(mac["model"] == INCUMBENT) & (mac["horizon"] == 168)]
    if not npts.empty:
        nv = float(npts["mase"].iloc[0])
        axes[1].axhline(nv, color=COLOR[INCUMBENT], lw=1.8, ls=(0, (5, 2)), zorder=2)
        axes[1].annotate(f"{LABEL[INCUMBENT]} — {nv:.4f}", (0.5, nv), xytext=(0, 6),
                         textcoords="offset points", fontsize=9, color=COLOR[INCUMBENT],
                         fontweight="600")
        lo = min(float(mac[(mac["model"].isin(order)) & (mac["horizon"] == 168)]["mase"].min()),
                 nv)
        axes[1].set_ylim(lo - 0.008, nv + 0.010)
    axes[1].set_xlabel("full-backtest runtime (minutes, Apple MPS)")
    axes[1].set_ylabel("macro MASE at 7 d")
    axes[1].set_title("What that gain costs, next to the gap being claimed",
                      fontsize=12, fontweight="600", color=INK, loc="left")
    axes[1].margins(x=0.22)

    fig.suptitle("N · Covariates vs zero-shot — the whole spread is under 0.003 MASE, "
                 "against a 0.040 MASE gap to the incumbent",
                 fontsize=13, fontweight="600", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "N_covariate_vs_zeroshot")


def main() -> None:
    import json
    from .review_package import (
        coverage, load_scored, macro, per_tank_comparison, per_tank_metrics,
        EXCLUDED_MODELS,
    )

    trust = json.loads(Path("eda/tank_trust.json").read_text())
    scored = load_scored()
    scored = scored[~scored["model"].isin(EXCLUDED_MODELS)]
    pt = per_tank_metrics(scored)
    mac = macro(pt).merge(coverage(scored), on=["model", "horizon"], how="left")
    cmp3 = per_tank_comparison(pt, horizons=(6, 24, 168))

    print("building figures...")
    metric_vs_horizon(mac, "mase", "A · MASE by forecast horizon", "MASE",
                      "A_mase_vs_horizon", ref_line=True)
    metric_vs_horizon(mac, "rmse", "B · RMSE by forecast horizon", "RMSE (KL/h)",
                      "B_rmse_vs_horizon")
    metric_vs_horizon(mac, "mae", "C · MAE by forecast horizon", "MAE (KL/h)",
                      "C_mae_vs_horizon")
    metric_vs_horizon(mac, "rmsse", "D · RMSSE by forecast horizon", "RMSSE",
                      "D_rmsse_vs_horizon")
    per_tank_mase(cmp3, 24)
    per_tank_improvement(cmp3, 24)
    per_tank_heatmap(pt)
    error_distribution(scored, 24)

    reps = pick_representatives(pt, trust)
    print("  representatives:", reps)
    actual_vs_pred(scored, reps, 24)
    final_holdout_plot(scored, reps)
    calibration(mac)

    # Wall-clock from the completed run manifest; no re-timing.
    manifest = json.loads((Path("results/chronos2") / "run_manifest.json").read_text())
    runtimes = {k: v["wall_clock_s"] / 60 for k, v in manifest["variants"].items()}
    variant_selection(mac, runtimes)

    # Three-way winner comes from the same helper that writes per_tank_comparison.csv, so the
    # figure and the published table cannot drift apart.
    from .score_benchmark import load_sensor_quality
    from .score_benchmark import per_tank_comparison as three_way
    tanks_won(three_way(pt, load_sensor_quality()))
    covariate_vs_zeroshot(mac, runtimes)

    (OUT / "representative_tanks.json").write_text(json.dumps(reps, indent=2))
    print(f"\nfigures -> {PLOTS}")


if __name__ == "__main__":
    main()
