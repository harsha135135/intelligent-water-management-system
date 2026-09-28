"""Figures for the PatchTST search and the Chronos-2+ stack.

Three panels, in the order the argument runs:

    P1  what the PatchTST search found, and which distribution heads were even available to it
    P2  the Chronos-2+ ablation — what each layer is worth, at every horizon
    P3  where the gain lands — per tank, against how much of that tank is exactly zero

Colour follows ``style.py``, so these read the same way as the unified set.

    python -m src.models.plus_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import GRID, INK, INK2, SURFACE, save
from .unified_figures import COLOR, SEQ

SEARCH = Path("results/chronos2/patchtst_search")
PLUS = Path("results/chronos2/plus")
PLOTS = PLUS / "plots"

HEAD_COLOR = {"StudentT": "#6d3fc4", "Laplace": "#a78bfa", "Normal": "#c4b5fd"}
HORDER = ["6 h", "12 h", "1 d", "2 d", "3 d", "7 d"]


def _title(ax, text, sub=None):
    ax.set_title(text, fontsize=13, fontweight="600", color=INK, loc="left",
                 pad=30 if sub else 11)
    if sub:
        ax.annotate(sub, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 9),
                    textcoords="offset points", fontsize=9.4, color=INK2, ha="left", va="bottom")


# ── P1 · the search ──────────────────────────────────────────────────────────

def p1_search() -> None:
    trials = pd.read_csv(SEARCH / "trials.csv")
    ok = trials[trials.status == "ok"].sort_values("val_MASE").reset_index(drop=True)
    heads = json.loads((SEARCH / "head_viability.json").read_text())["result"]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    y = np.arange(len(ok))[::-1]
    ax.barh(y, ok.val_MASE, color=[HEAD_COLOR.get(h, "#999") for h in ok.head],
            height=0.72, zorder=3)
    best = ok.iloc[0]
    ax.barh([y[0]], [best.val_MASE], color=COLOR["PatchTST-Tuned"], height=0.72, zorder=4)
    for yy, (_, r) in zip(y, ok.iterrows()):
        ax.annotate(f"{r.head} · ctx {int(r.context_length)} · d{int(r.d_model)} ·"
                    f" L{int(r.num_encoder_layers)} · {r.target_scaler}",
                    (0.006, yy), fontsize=7.6, va="center", color="#ffffff", zorder=5)
        ax.annotate(f"{r.val_MASE:.4f}", (r.val_MASE, yy), xytext=(5, 0),
                    textcoords="offset points", fontsize=8.2, va="center", color=INK2)
    ax.set_yticks([])
    ax.set_xlabel("validation MASE  (lower is better) — held out before the first test origin")
    ax.set_xlim(0, ok.val_MASE.max() * 1.16)
    ax.grid(axis="y", alpha=0)
    _title(ax, f"The search: {len(ok)} configurations that trained",
           "Each bar is one sampled configuration. The best is highlighted.")

    ax = axes[1]
    names = list(heads)
    trains = [heads[n]["trains"] for n in names]
    ax.barh(np.arange(len(names))[::-1], [1] * len(names),
            color=["#1baf7a" if t else "#c3352f" for t in trains], height=0.66, zorder=3)
    for i, n in zip(np.arange(len(names))[::-1], names):
        ok_i = heads[n]["trains"]
        ax.annotate(n, (0.02, i), fontsize=9.2, va="center", color="#ffffff",
                    fontweight="600", zorder=4)
        ax.annotate("trains" if ok_i else "cannot train", (0.98, i), ha="right",
                    fontsize=8.6, va="center", color="#ffffff", zorder=4)
    ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0, 1); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    _title(ax, "…and the heads it could not use",
           "Every head able to place mass at exactly zero is a quantile-function output, and\n"
           "gluonts' PatchTST computes its loss as a negative log-likelihood.")
    fig.text(0.555, -0.02,
             "This is the ceiling on PatchTST here: the dataset is 24 % exact zeros, and within "
             "this implementation the model is confined to continuous densities that cannot "
             "represent them.", fontsize=9, color=INK2, wrap=True)
    save(fig, PLOTS, "P1_patchtst_search")


# ── P2 · the ablation ────────────────────────────────────────────────────────

def p2_ablation() -> None:
    ab = pd.read_csv(PLUS / "ablation.csv")
    layers = list(dict.fromkeys(ab.layer))
    piv = ab.pivot(index="layer", columns="horizon_label", values="macro_mase") \
            .loc[layers, [h for h in HORDER if h in set(ab.horizon_label)]]
    base = piv.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    x = np.arange(len(piv.columns))
    shades = [SEQ(v) for v in np.linspace(0.30, 0.92, len(layers))]
    for i, (lab, row) in enumerate(piv.iterrows()):
        ax.plot(x, row.to_numpy(), marker="o", ms=7, lw=2.4 if i == len(layers) - 1 else 1.6,
                color=shades[i], mec=SURFACE, mew=1.6,
                label=lab, zorder=4 if i == len(layers) - 1 else 3)
    ax.set_xticks(x, piv.columns)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("macro MASE  (lower is better)")
    ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    _title(ax, "Each layer, at every horizon",
           "Cumulative: every line includes the layers above it in the legend.")

    ax = axes[1]
    imp = 100 * (base - piv) / base
    x = np.arange(len(piv.columns))
    width = 0.8 / max(len(layers) - 1, 1)
    for i, lab in enumerate(layers[1:], start=1):
        ax.bar(x + (i - 1) * width - 0.4 + width / 2, imp.loc[lab], width,
               color=shades[i], label=lab.split(" · ")[0], zorder=3)
    ax.axhline(0, color=INK2, lw=1.1, zorder=2)
    ax.set_xticks(x, piv.columns)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("cumulative MASE improvement over L0 (%)")
    ax.legend(frameon=False, fontsize=9, ncols=2)
    _title(ax, "…and what it is worth, against the published baseline")
    save(fig, PLOTS, "P2_chronos2_plus_ablation")


# ── P3 · where the gain lands ────────────────────────────────────────────────

def p3_per_tank() -> None:
    pt = pd.read_csv(PLUS / "per_tank_h24.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    sc = ax.scatter(pt.zero_share * 100, pt.improvement_pct,
                    s=40 + 240 * (pt.mean_actual / pt.mean_actual.max()) ** 0.5,
                    c=pt.gated_share * 100, cmap=SEQ, ec=SURFACE, lw=1.4, zorder=3)
    ax.axhline(0, color=INK2, lw=1.2, zorder=2)
    for name, r in pt.iterrows():
        if abs(r.improvement_pct) > 4 or r.zero_share > 0.5:
            ax.annotate(name, (r.zero_share * 100, r.improvement_pct), xytext=(7, 0),
                        textcoords="offset points", fontsize=7.4, color=INK2, va="center")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="% of forecast hours the hurdle gated to zero",
                 pad=0.02)
    ax.set_xlabel("% of that tank's hours that are exactly zero")
    ax.set_ylabel("MAE improvement at 1 d over the published mean (%)")
    _title(ax, "The gain goes where the zeros are",
           "Marker size is mean demand. Tanks with more exact-zero hours gain more, which is what "
           "a hurdle built on the zero diagnosis should do.")
    save(fig, PLOTS, "P3_where_the_gain_lands")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    if (SEARCH / "trials.csv").exists():
        p1_search()
    if (PLUS / "ablation.csv").exists():
        p2_ablation()
    if (PLUS / "per_tank_h24.csv").exists():
        p3_per_tank()
    print(f"figures -> {PLOTS}")


if __name__ == "__main__":
    main()
