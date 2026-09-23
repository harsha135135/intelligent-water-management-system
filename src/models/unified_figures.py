"""The unified figure set — twelve figures, every one carrying every model.

Replaces three overlapping sets (A-N Chronos-2 vs NPTS, O-W diagnostics, AA-AD PatchTST). Each
of those answered one comparison, so the same fact appeared in three shapes and no single figure
showed the whole field. These twelve do, and nothing here is drawn twice.

Colour is assigned by **family**, fixed in ``unified_analysis.FAMILY``, never by rank — so a
model keeps its colour in every figure and cannot change identity by scoring differently:

    foundation  blue      trained deep  purple
    incumbent   orange    statistical   ochre        reference  green

Reads only the CSVs written by ``unified_analysis``; recomputes nothing.

    python -m src.models.unified_figures     # ~25 s -> results/chronos2/unified/plots/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .unified_analysis import (
    CHRONOS, FAMILY, HLABEL, HORIZONS, INCUMBENT, LABEL, ORDER, OUT, PLOTS, REFERENCE, SHORT,
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

# Identity colours. The three established ones are unchanged from the earlier figures, so a
# reader who has seen the old deck reads the new one the same way.
COLOR = {
    "Chronos2-ZS": "#2a78d6", "Chronos2-COV-XL": "#5598e7",
    "Chronos2-COV": "#82b6f0", "Chronos2-COV-LEAN": "#a8ccf4",
    "PatchTST-Tuned": "#6d3fc4", "PatchTST": "#a78bfa",
    "NPTS": "#eb6834",
    "ETS": "#8a6d3b", "Theta": "#b08a4f", "DynamicOptimizedTheta": "#cbab77",
    "SeasonalNaive": "#1baf7a",
}
FAMILY_COLOR = {"foundation": "#2a78d6", "trained deep": "#6d3fc4",
                "statistical": "#b0762b", "reference": "#1baf7a"}

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#c0392b", "#e8a0a0", "#f4f3f0", "#86b6ef", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
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
    print(f"  [plot] {name}")


def _title(ax, text, sub=None):
    """Title above a subtitle, both left-aligned to the axes. The subtitle is placed in offset
    points so it never lands on the title however tall the figure is."""
    ax.set_title(text, fontsize=13, fontweight="600", color=INK, loc="left",
                 pad=30 if sub else 11)
    if sub:
        ax.annotate(sub, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 9),
                    textcoords="offset points", fontsize=9.4, color=INK2, ha="left", va="bottom")


def _end_labels(ax, entries, *, fontsize=8.5, dx=9):
    """Direct-label series at their right-hand end, pushed apart so none overlaps another.

    Series that genuinely coincide — the four Chronos-2 variants differ by 0.003 MASE — would
    otherwise print on top of each other and read as one illegible smudge. The labels are nudged
    apart in display space; the lines themselves are never moved.
    """
    if not entries:
        return
    ax.figure.canvas.draw()                      # transforms must be current to position text
    # transData is in *device* pixels, so the minimum gap has to be derived from the font size
    # at the figure's dpi — a fixed pixel constant is only right at 72 dpi and collides at 150.
    min_gap_px = fontsize / 72.0 * ax.figure.dpi * 1.30
    pts = [(ax.transData.transform((x, y)), t, c, w) for x, y, t, c, w in entries]
    items = sorted(pts, key=lambda e: e[0][1])
    ys = [e[0][1] for e in items]
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            gap = ys[i] - ys[i - 1]
            if gap < min_gap_px:
                shift = (min_gap_px - gap) / 2
                ys[i - 1] -= shift
                ys[i] += shift
                moved = True
        if not moved:
            break
    # Back to data coordinates before annotating: figure-pixel positions do not survive
    # savefig(bbox_inches="tight"), which re-crops the canvas and shifts everything.
    inv = ax.transData.inverted()
    for ((px, _py), text, colour, weight), newy in zip(items, ys):
        xdata, ydata = inv.transform((px, newy))
        ax.annotate(text, xy=(xdata, ydata), xycoords="data",
                    xytext=(dx, 0), textcoords="offset points",
                    color=colour, fontsize=fontsize, va="center", ha="left",
                    fontweight=weight, annotation_clip=False)


def _horizon_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([HLABEL[h] for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.grid(axis="x", alpha=0.4)


def _order(df, col="model"):
    have = [m for m in ORDER if m in set(df[col])]
    return have + sorted(set(df[col]) - set(ORDER))


def _family_legend(ax, loc="lower right"):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=3, label=f) for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, frameon=False, loc=loc, fontsize=9, title="model family",
              title_fontsize=9)


# ── U1 · the leaderboard, as a curve ─────────────────────────────────────────

def u1_leaderboard(lb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ends = []
    for m in _order(lb):
        s = lb[lb.model == m].sort_values("horizon")
        lead = m == CHRONOS
        ax.plot(s.horizon, s.macro_mase, color=COLOR[m], lw=2.6 if lead else 1.6,
                marker="o", ms=8 if lead else 5, mfc=COLOR[m], mec=SURFACE, mew=1.8,
                zorder=4 if lead else 3, alpha=1.0 if lead else 0.9)
        ends.append((s.horizon.iloc[-1], s.macro_mase.iloc[-1], SHORT[m], COLOR[m],
                     "700" if lead else "600"))
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("seasonal-naive reference = 1.0", (HORIZONS[0], 1.0), xytext=(0, 6),
                textcoords="offset points", color=INK2, fontsize=8.5)
    _horizon_axis(ax)
    ax.set_ylabel("macro MASE  (lower is better)")
    ax.set_xlim(HORIZONS[0] * 0.88, HORIZONS[-1] * 3.4)
    _end_labels(ax, ends)
    _title(ax, "Every model, every horizon",
           "Macro MASE on the identical 188,664 rows. The four Chronos-2 variants coincide — "
           "labels are pushed apart to stay legible, the lines are not.")
    _save(fig, "U1_leaderboard_mase")


# ── U2 · the same table as a heatmap, with rank ──────────────────────────────

def u2_heatmap(lb: pd.DataFrame) -> None:
    models = _order(lb)
    p = lb.pivot(index="model", columns="horizon", values="macro_mase").loc[models, HORIZONS]
    r = lb.pivot(index="model", columns="horizon", values="rank_mase").loc[models, HORIZONS]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    im = ax.imshow(p.to_numpy(), cmap=SEQ, aspect="auto")
    ax.set_xticks(range(len(HORIZONS)), [HLABEL[h] for h in HORIZONS])
    ax.set_yticks(range(len(models)), [LABEL[m] for m in models], fontsize=9.5)
    for i in range(len(models)):
        for j in range(len(HORIZONS)):
            v, rk = p.iloc[i, j], int(r.iloc[i, j])
            dark = v > np.nanpercentile(p.to_numpy(), 62)
            ax.text(j, i - 0.10, f"{v:.3f}", ha="center", va="center", fontsize=9.2,
                    color="#ffffff" if dark else INK,
                    fontweight="700" if rk == 1 else "500")
            ax.text(j, i + 0.22, f"#{rk}", ha="center", va="center", fontsize=7.4,
                    color="#dbe7f5" if dark else INK2)
    for k, m in enumerate(models):
        ax.add_patch(plt.Rectangle((-0.5, k - 0.5), len(HORIZONS), 1, fill=False,
                                   ec=FAMILY_COLOR[FAMILY[m]], lw=2.2 if m == CHRONOS else 0,
                                   zorder=5))
    ax.set_xlabel("Forecast horizon")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.72, label="macro MASE", pad=0.02)
    _title(ax, "Leaderboard — macro MASE and rank in every cell",
           "The four Chronos-2 variants are separated by 0.003 MASE; the gap to the next family "
           "is twenty times that.")
    _save(fig, "U2_leaderboard_heatmap")


# ── U3 · Chronos-2 against each opponent, with confidence ────────────────────

def u3_significance(sig: pd.DataFrame) -> None:
    s = sig[(sig.metric == "MASE")]
    opponents = [m for m in ORDER if m in set(s.opponent)]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    d = s[s.horizon == 24].set_index("opponent")
    ys = np.arange(len(opponents))[::-1]
    for y, o in zip(ys, opponents):
        r = d.loc[o]
        col = COLOR[o]
        sigf = bool(r.significant_95)
        ax.plot([r.ci_lo_pct, r.ci_hi_pct], [y, y], color=col, lw=3 if sigf else 1.6,
                solid_capstyle="round", alpha=1 if sigf else 0.45, zorder=3)
        ax.plot([r.improvement_pct], [y], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{r.improvement_pct:+.1f}%" + ("" if sigf else "  n.s."),
                    (r.ci_hi_pct, y), xytext=(9, 0), textcoords="offset points",
                    color=col if sigf else INK2, fontsize=9,
                    va="center", fontweight="600")
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(ys, [LABEL[o] for o in opponents], fontsize=9.5)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.20)
    ax.set_xlabel("Chronos-2 MASE improvement at 1 d  (%, 95% bootstrap CI)")
    ax.grid(axis="y", alpha=0)
    _title(ax, "Against every opponent, at 1 day")

    ax = axes[1]
    ends = []
    for o in opponents:
        g = s[s.opponent == o].sort_values("horizon")
        faint = FAMILY[o] == "foundation"
        ax.plot(g.horizon, g.improvement_pct, color=COLOR[o], lw=1.5 if faint else 2.2,
                marker="o", ms=4.5, alpha=0.55 if faint else 1, zorder=3)
        ends.append((g.horizon.iloc[-1], g.improvement_pct.iloc[-1], SHORT[o], COLOR[o], "600"))
    ax.axhline(0, color=INK2, lw=1.2, zorder=2)
    _horizon_axis(ax)
    ax.set_ylabel("MASE improvement (%)")
    ax.set_xlim(HORIZONS[0] * 0.88, HORIZONS[-1] * 3.6)
    _end_labels(ax, ends)
    _title(ax, "…and at every horizon")
    fig.suptitle("")
    _save(fig, "U3_significance_vs_all")


# ── U4 · who beats whom ──────────────────────────────────────────────────────

def u4_win_matrix(wm: pd.DataFrame) -> None:
    models = [m for m in ORDER if m in set(wm.model_a)]
    n = len(models)
    M = np.full((n, n), np.nan)
    idx = {m: i for i, m in enumerate(models)}
    for r in wm.itertuples():
        M[idx[r.model_a], idx[r.model_b]] = r.a_win_pct
    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100), aspect="auto")
    ax.set_xticks(range(n), [SHORT[m] for m in models], rotation=45, ha="right", fontsize=8.6)
    ax.set_yticks(range(n), [SHORT[m] for m in models], fontsize=8.6)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=GRID, zorder=2))
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.4,
                    color="#ffffff" if (v > 78 or v < 22) else INK, zorder=3)
    ax.set_xlabel("…against this model")
    ax.set_ylabel("this model wins % of cells…")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="% of 144 tank × horizon cells won", pad=0.02)
    _title(ax, "Head-to-head, all 144 tank × horizon cells",
           "Read a row: how often that model beats each other model on per-tank MAE. 50 = a tie.")
    _save(fig, "U4_win_matrix")


# ── U5 · per tank, per model ─────────────────────────────────────────────────

def u5_per_tank(pt: pd.DataFrame) -> None:
    d = pt[pt.horizon == 24]
    models = _order(d)
    piv = d.pivot(index="tank", columns="model", values="mase")[models]
    order_tanks = d[d.model == CHRONOS].sort_values("mean_actual", ascending=False)["tank"]
    piv = piv.loc[[t for t in order_tanks if t in piv.index]]
    trust = d.drop_duplicates("tank").set_index("tank")["trust"]

    fig, ax = plt.subplots(figsize=(10.4, 8.6))
    im = ax.imshow(piv.to_numpy(), cmap=SEQ, aspect="auto", vmin=0,
                   vmax=float(np.nanpercentile(piv.to_numpy(), 96)))
    ax.set_xticks(range(len(models)), [SHORT[m] for m in models], rotation=45,
                  ha="right", fontsize=8.6)
    tcol = {"healthy": "#12855c", "degraded": "#a8720b", "dead": "#c3352f"}
    ax.set_yticks(range(len(piv)), piv.index, fontsize=7.6)
    for k, t in enumerate(piv.index):
        ax.get_yticklabels()[k].set_color(tcol.get(trust.get(t), INK2))
    for i in range(len(piv)):
        for j in range(len(models)):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                        color="#ffffff" if v > np.nanpercentile(piv.to_numpy(), 62) else INK2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="MASE at 1 d", pad=0.015)
    _title(ax, "Per tank, per model — MASE at the 1-day horizon",
           "Tanks ordered by mean demand, highest first. Label colour is the sensor tier: "
           "green healthy, amber degraded, red dead.")
    _save(fig, "U5_per_tank_heatmap")


# ── U6 · does skill survive the dead sensors ─────────────────────────────────

def u6_skill(sk: pd.DataFrame) -> None:
    models = _order(sk)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for m in models:
        g = sk[sk.model == m]
        lead = m == CHRONOS
        ax.scatter(g.mean_actual, g.skill, s=52 if lead else 26, color=COLOR[m],
                   ec=SURFACE, lw=1.1, alpha=1 if lead else 0.66, zorder=4 if lead else 3,
                   label=SHORT[m] if lead or m in (INCUMBENT, "PatchTST-Tuned") else None)
    ax.axhline(0, color=INK2, lw=1.2, zorder=2)
    ax.annotate("below this line the model is worse than seasonal naive",
                (ax.get_xlim()[0], 0), xytext=(4, -13), textcoords="offset points",
                color=INK2, fontsize=8.5)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("mean demand on that tank (KL/h, symlog)")
    ax.set_ylabel("skill vs SeasonalNaive-24  (1 − MAE ratio)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _title(ax, "Skill against the naive reference, every tank and every model",
           "Every Chronos-2 point is above zero on all 24 tanks — including the dead sensors, "
           "which is what the 'flattered average' objection asks about.")
    _save(fig, "U6_skill_vs_demand")


# ── U7 · calibration: coverage must be read next to width ────────────────────

def u7_calibration(lb: pd.DataFrame, zi: pd.DataFrame) -> None:
    d = lb[lb.horizon == 24].set_index("model")
    z = zi.set_index("model")
    models = [m for m in ORDER if m in d.index]
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ends = []
    for m in models:
        r = d.loc[m]
        neg = float(z.loc[m, "p10_below_zero_pct"]) if m in z.index else 0.0
        lead = m == CHRONOS
        ax.scatter(r.width, r.coverage, s=190 if lead else 120, color=COLOR[m],
                   ec=SURFACE, lw=2, zorder=4 if lead else 3)
        if neg > 20:
            ax.scatter(r.width, r.coverage, s=330 if lead else 250, facecolors="none",
                       edgecolors="#c3352f", lw=1.6, ls=(0, (2, 2)), zorder=5)
        ends.append((r.width, r.coverage,
                     SHORT[m] + (f"  ·  p10<0 on {neg:.0f}%" if neg > 20 else ""),
                     COLOR[m], "700" if lead else "600"))
    ax.axhline(0.80, color=INK2, lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.annotate("nominal 0.80", (ax.get_xlim()[1], 0.80), xytext=(-6, 6),
                textcoords="offset points", color=INK2, fontsize=8.5, ha="right")
    ax.set_xlabel("mean p10–p90 width at 1 d (KL/h)  →  wider")
    ax.set_ylabel("empirical coverage  →  closer to nominal is better")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.30)
    _end_labels(ax, ends, dx=12)
    _title(ax, "Interval calibration — coverage against the width that bought it",
           "Red rings mark models whose p10 goes below zero, i.e. that cover by predicting a "
           "negative outflow. Coverage alone would rank them best.")
    _save(fig, "U7_calibration")


# ── U8 · what the accuracy cost ──────────────────────────────────────────────

def u8_cost(ca: pd.DataFrame) -> None:
    d = ca.dropna(subset=["wall_clock_s"])
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    ends = []
    for r in d.itertuples():
        lead = r.model == CHRONOS
        ax.scatter(r.wall_clock_s, r.macro_mase_h24, s=200 if lead else 118,
                   color=COLOR[r.model], ec=SURFACE, lw=2, zorder=4 if lead else 3)
        ends.append((r.wall_clock_s, r.macro_mase_h24,
                     f"{SHORT[r.model]}  ·  {r.wall_clock_s/60:.1f} min",
                     COLOR[r.model], "700" if lead else "600"))
    ax.set_xscale("log")
    ax.set_xlabel("wall clock for the full 24-origin × 6-horizon backtest (s, log)")
    ax.set_ylabel("macro MASE at 1 d  (lower is better)")
    ax.set_xlim(right=ax.get_xlim()[1] * 6.0)
    _end_labels(ax, ends, dx=12)
    _title(ax, "Accuracy against what it cost to get it",
           "Down and to the left is better. The production candidate is the leftmost point on "
           "the chart and within 0.003 MASE of the best.")
    _save(fig, "U8_cost_accuracy")


# ── U9 · does it fall apart at long lead times ───────────────────────────────

def u9_leadtime(lt: pd.DataFrame) -> None:
    models = _order(lt)
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ends = []
    for m in models:
        g = lt[lt.model == m].sort_values("step")
        roll = g.mae.rolling(24, min_periods=1, center=True).mean()
        lead = m == CHRONOS
        ax.plot(g.step, roll, color=COLOR[m], lw=2.4 if lead else 1.4,
                alpha=1 if lead else 0.82, zorder=4 if lead else 3)
        ends.append((g.step.iloc[-1], roll.iloc[-1], SHORT[m], COLOR[m],
                     "700" if lead else "600"))
    for d in range(24, 169, 24):
        ax.axvline(d, color=GRID, lw=0.8, zorder=1)
    ax.set_xlabel("hours ahead of the forecast origin")
    ax.set_ylabel("MAE (KL/h), 24-hour rolling mean")
    ax.set_xlim(1, 168 * 1.24)
    ax.set_xticks(range(0, 169, 24))
    _end_labels(ax, ends)
    _title(ax, "Error against lead time, out to seven days",
           "A flat line means the model learned the repeating daily shape rather than "
           "extrapolating a trend — which is what makes weekly planning viable.")
    _save(fig, "U9_error_by_leadtime")


# ── U10 · where in the day the error lives ───────────────────────────────────

def u10_diurnal(dn: pd.DataFrame) -> None:
    models = [m for m in ORDER if m in set(dn.model)]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    dem = dn[dn.model == CHRONOS].sort_values("hour")
    for ax, col, lab in zip(axes, ("mae", "bias"),
                            ("MAE (KL/h)", "signed bias (KL/h) — negative is under-forecast")):
        for m in models:
            g = dn[dn.model == m].sort_values("hour")
            lead = m == CHRONOS
            ax.plot(g.hour, g[col], color=COLOR[m], lw=2.4 if lead else 1.3,
                    alpha=1 if lead else 0.75, zorder=4 if lead else 3)
        if col == "bias":
            ax.axhline(0, color=INK2, lw=1.2, zorder=2)
        ax.set_xlabel("hour of day")
        ax.set_ylabel(lab)
        ax.set_xticks(range(0, 24, 3))
    ax2 = axes[0].twinx()
    ax2.fill_between(dem.hour, dem.mean_actual, color="#2a78d6", alpha=0.08, zorder=0)
    ax2.set_ylabel("mean demand (KL/h)", color=INK2)
    ax2.grid(False)
    _title(axes[0], "Error tracks demand, hour by hour")
    _title(axes[1], "The under-forecast is present in every hour")
    _save(fig, "U10_diurnal")


# ── U11 · why the intervals miss ─────────────────────────────────────────────

def u11_zero_inflation(zi: pd.DataFrame) -> None:
    models = [m for m in ORDER if m in set(zi.model)]
    d = zi.set_index("model").loc[models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    ax.bar(x - 0.2, d.miss_below_p10, 0.4, color="#2a78d6", label="below p10", zorder=3)
    ax.bar(x + 0.2, d.miss_above_p90, 0.4, color="#a8ccf4", label="above p90", zorder=3)
    ax.axhline(0.10, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("expected 0.10", (len(models) - 0.5, 0.10), xytext=(0, 5),
                textcoords="offset points", color=INK2, fontsize=8.4, ha="right")
    ax.set_xticks(x, [SHORT[m] for m in models], rotation=45, ha="right", fontsize=8.2)
    ax.set_ylabel("fraction of actuals outside the band")
    ax.legend(frameon=False, fontsize=9)
    _title(ax, "Misses are lower-tail")

    ax = axes[1]
    ax.bar(x, d.p10_below_zero_pct, 0.62,
           color=[COLOR[m] for m in models], zorder=3)
    ax.set_xticks(x, [SHORT[m] for m in models], rotation=45, ha="right", fontsize=8.2)
    ax.set_ylabel("% of rows where p10 < 0")
    _title(ax, "Covering costs a negative p10")

    ax = axes[2]
    ax.bar(x - 0.2, d.coverage, 0.4, color="#2a78d6", label="as published", zorder=3)
    ax.bar(x + 0.2, d.coverage_if_p10_clamped, 0.4, color="#1baf7a",
           label="if p10 clamped to 0", zorder=3)
    ax.axhline(0.80, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(x, [SHORT[m] for m in models], rotation=45, ha="right", fontsize=8.2)
    ax.set_ylabel("p10–p90 coverage")
    ax.legend(frameon=False, fontsize=9)
    _title(ax, "A clamp over-corrects")

    z = d.loc[CHRONOS]
    fig.text(0.5, -0.09,
             f"{z.zero_fraction*100:.0f}% of hourly readings are exactly zero. Chronos-2 places "
             f"p10 above zero on {z.p10_above_zero_pct:.0f}% of rows, so "
             f"{z.below_misses_that_are_zero*100:.0f}% of its lower-tail misses are zero-demand "
             f"hours. Clamping reaches {z.coverage_if_p10_clamped:.3f} — past nominal — which is "
             f"why the fix is asymmetric conformal calibration, not a clamp.",
             ha="center", va="top", fontsize=9, color=INK2, wrap=True)
    _save(fig, "U11_zero_inflation")


# ── U12 · the operational number: how much water ─────────────────────────────

def u12_volume_bias(lb: pd.DataFrame) -> None:
    models = _order(lb)
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ends = []
    for m in models:
        g = lb[lb.model == m].sort_values("horizon")
        lead = m == CHRONOS
        ax.plot(g.horizon, g.volume_bias_pct, color=COLOR[m], lw=2.4 if lead else 1.4,
                marker="o", ms=7 if lead else 4, mfc=COLOR[m], mec=SURFACE, mew=1.5,
                alpha=1 if lead else 0.85, zorder=4 if lead else 3)
        ends.append((g.horizon.iloc[-1], g.volume_bias_pct.iloc[-1], SHORT[m], COLOR[m],
                     "700" if lead else "600"))
    ax.axhline(0, color=INK2, lw=1.3, zorder=2)
    ax.annotate("under-forecast — a refill sized on the mean would be short",
                (HORIZONS[0], 0), xytext=(0, -15), textcoords="offset points",
                color=INK2, fontsize=8.5)
    _horizon_axis(ax)
    ax.set_ylabel("volume bias (%)  — signed, all tanks")
    ax.set_xlim(HORIZONS[0] * 0.88, HORIZONS[-1] * 3.4)
    _end_labels(ax, ends)
    d24 = lb[lb.horizon == 24].set_index("model")["volume_bias_pct"]
    _title(ax, "Signed volume bias — the number that sizes a refill",
           f"At 1 d every model under-forecasts except ETS and the naive reference, which sit near "
           f"zero. Chronos-2 is {d24[CHRONOS]:+.1f}%, the trained deep models "
           f"{d24['PatchTST-Tuned']:+.1f}% and {d24['PatchTST']:+.1f}%.")
    _save(fig, "U12_volume_bias")


# ────────────────────────────────────────────────────────────── driver

def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    r = lambda n: pd.read_csv(OUT / n)
    print("Unified figures — every model in every panel\n")
    lb, sig = r("leaderboard.csv"), r("significance_vs_all.csv")
    wm, pt = r("win_matrix_all_horizons.csv"), r("per_tank.csv")
    sk, lt = r("skill_h24.csv"), r("error_by_leadtime.csv")
    dn, zi, ca = r("diurnal_h24.csv"), r("zero_inflation_h24.csv"), r("cost.csv")

    u1_leaderboard(lb)
    u2_heatmap(lb)
    u3_significance(sig)
    u4_win_matrix(wm)
    u5_per_tank(pt)
    u6_skill(sk)
    u7_calibration(lb, zi)
    u8_cost(ca)
    u9_leadtime(lt)
    u10_diurnal(dn)
    u11_zero_inflation(zi)
    u12_volume_bias(lb)
    print(f"\nfigures -> {PLOTS}")


if __name__ == "__main__":
    main()
