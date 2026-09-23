"""Score the PyTorch MLP and linear baselines against every model already in the study.

Reads the eleven existing ``results/chronos2/predictions_*.parquet`` and the two produced by
``mlp_forecaster.py`` (kept in ``results/chronos2/mlp/`` so the unified study's glob never picks
them up), restricts all of them to horizons 6/12/24 — the only ones the MLP emits — and scores
them with the same ``metrics.py`` functions as ``score_benchmark``. Nothing is refitted.

**Row parity is enforced, not assumed**: the MLP rows must be exactly the (tank, origin, horizon,
timestamp) keys every other model was scored on, or the script stops.

Writes to ``results/chronos2/mlp/``: ``leaderboard.csv``, ``significance.csv``,
``per_tank_h24.csv``, ``report.md`` and three figures.

    python -m src.models.mlp_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import style
from .backtest import assert_comparable
from .metrics import aggregate_metrics, attach_scales, per_tank_metrics
from .mlp_forecaster import SCORED_HORIZONS
from .significance import boot_ci, dm_test

RESULTS = Path("results/chronos2")
OUT = RESULTS / "mlp"
PLOTS = OUT / "plots"

BASE = "MLP"
OPPONENTS = ["Linear-AR", "SeasonalNaive", "NPTS", "PatchTST-Tuned", "Chronos2-ZS"]
COLOR = {**style.COLOR, "MLP": "#8b4fc8", "Linear-AR": "#8a857a", "PatchTST-Tuned": "#c9a227"}
LABEL = {**style.LABEL, "MLP": "MLP (PyTorch, this study)",
         "Linear-AR": "Linear-AR (PyTorch, this study)", "PatchTST-Tuned": "PatchTST-Tuned"}


def deep_dive_tank() -> str:
    """Chosen by a rule fixed before any result was looked at: the healthy tank with the highest
    mean demand. Picking the tank the MLP happens to do best on would make the plot an advert."""
    mb = pd.read_csv("eda/tank_mass_balance.csv")
    return str(mb[mb["trust"] == "healthy"].sort_values("mean_kl", ascending=False)
               .iloc[0]["item_id"])


def load_scored() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(RESULTS.glob("predictions_*.parquet"))]
    frames += [pd.read_parquet(p) for p in sorted(OUT.glob("predictions_*.parquet"))]
    preds = pd.concat(frames, ignore_index=True)
    preds = preds[preds["horizon"].isin(SCORED_HORIZONS)]

    from ..data.curate import load_curated_hourly
    panel = load_curated_hourly(with_features=False)
    scored = attach_scales(preds, panel).dropna(subset=["actual", "pred"]).copy()

    keys = ["item_id", "origin", "horizon", "timestamp"]
    ref = scored[scored.model == "SeasonalNaive"][keys].sort_values(keys).reset_index(drop=True)
    for model in ("MLP", "Linear-AR"):
        mine = scored[scored.model == model][keys].sort_values(keys).reset_index(drop=True)
        if not mine.equals(ref):
            raise AssertionError(f"{model} was not scored on the benchmark's exact rows")
    assert_comparable(scored)

    ok = scored["scale_mae"] > 0
    scored["abs_err"] = (scored["actual"] - scored["pred"]).abs()
    scored["scaled_abs"] = np.where(ok, scored["abs_err"] / scored["scale_mae"], np.nan)
    return scored


def leaderboard(scored: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_metrics(per_tank_metrics(scored))
    agg["rank_mase"] = agg.groupby("horizon")["macro_mase"].rank(method="min").astype(int)
    return agg.sort_values(["horizon", "rank_mase"]).reset_index(drop=True)


def _per_origin(scored: pd.DataFrame, model: str, horizon: int) -> np.ndarray:
    """Macro MASE per origin: mean over tanks of each tank's mean scaled error at that origin.
    Same construction as ``unified_analysis._macro_per_origin``, so the tests are comparable."""
    s = scored[(scored.model == model) & (scored.horizon == horizon)]
    per_tank = s.groupby(["origin", "item_id"])["scaled_abs"].mean()
    return per_tank.groupby("origin").mean().sort_index().to_numpy()


def significance(scored: pd.DataFrame) -> pd.DataFrame:
    """Paired tests on every tank, then again on healthy tanks only.

    The second pass exists because a macro mean over tanks lets one broken sensor dominate: a
    dead tank scoring MASE 4 moves the average more than ten healthy tanks scoring 0.7. A claim
    that only survives with the dead sensors included is a claim about those sensors.
    """
    trust = json.loads(Path("eda/tank_trust.json").read_text())
    healthy = {t for t, tier in trust.items() if tier == "healthy"}
    subsets = {"all tanks": scored, "healthy only": scored[scored.item_id.isin(healthy)]}
    rows = []
    for subset, data in subsets.items():
        rows += _significance_rows(data, subset)
    return pd.DataFrame(rows)


def _significance_rows(scored: pd.DataFrame, subset: str) -> list[dict]:
    rows = []
    for opponent in OPPONENTS:
        for horizon in SCORED_HORIZONS:
            b, o = _per_origin(scored, BASE, horizon), _per_origin(scored, opponent, horizon)
            d = b - o                         # negative = the MLP has lower error
            mean, lo, hi, p_boot = boot_ci(d)
            dm, p_dm = dm_test(d, horizon)
            rows.append({
                "subset": subset, "base": BASE, "opponent": opponent, "horizon": horizon,
                "base_mase": float(b.mean()), "opponent_mase": float(o.mean()),
                "improvement_pct": float(-100 * mean / o.mean()),
                "ci_lo_pct": float(-100 * hi / o.mean()), "ci_hi_pct": float(-100 * lo / o.mean()),
                "p_bootstrap": p_boot, "dm_stat": dm, "p_dm": p_dm,
                "significant_95": bool(p_boot < 0.05 and p_dm < 0.05),
                "origins_won": int((d < 0).sum()), "n_origins": int(len(d)),
            })
    return rows


def per_tank_h24(scored: pd.DataFrame) -> pd.DataFrame:
    pt = per_tank_metrics(scored[scored.horizon == 24])
    wide = pt.pivot(index="item_id", columns="model", values="mase")
    trust = json.loads(Path("eda/tank_trust.json").read_text())
    cols = ["MLP", "Linear-AR", "SeasonalNaive", "NPTS", "PatchTST-Tuned", "Chronos2-ZS"]
    out = wide[cols].copy()
    out.insert(0, "trust", [trust.get(i, "") for i in out.index])
    out["mlp_beats_seasonal_naive"] = out["MLP"] < out["SeasonalNaive"]
    out["mlp_beats_linear"] = out["MLP"] < out["Linear-AR"]
    out["mlp_beats_chronos2"] = out["MLP"] < out["Chronos2-ZS"]
    return out.reset_index()


# ────────────────────────────────────────────────────────────── figures

def fig_leaderboard(lb: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    g = lb[lb.horizon == 24].sort_values("macro_mase")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = [COLOR.get(m, "#c8c6c0") for m in g.model]
    ax.barh(g.model, g.macro_mase, color=colors)
    ax.axvline(1.0, color=style.INK2, lw=0.8, ls="--")
    ax.invert_yaxis()
    for y, v in enumerate(g.macro_mase):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8.5, color=style.INK)
    ax.set_xlabel("macro MASE at 24 h (lower is better)")
    ax.set_title("Every model on the same 24-hour rows", loc="left", fontsize=11)
    style.caption(fig, "Coloured bars: the two PyTorch models trained in this study, Chronos-2 "
                  "zero-shot, NPTS, SeasonalNaive and the tuned PatchTST. Dashed line: MASE 1.0.")
    style.save(fig, PLOTS, "M1_leaderboard_h24")


def fig_loss_curves(tank: str) -> None:
    import matplotlib.pyplot as plt
    h = pd.read_csv(OUT / "training_history.csv")
    h = h[h.item_id == tank]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, model in zip(axes, ["MLP", "Linear-AR"], strict=True):
        g = h[h.model == model]
        ax.plot(g.epoch, g.train_loss, color=COLOR[model], lw=1.6, label="training")
        ax.plot(g.epoch, g.val_loss, color=style.INK2, lw=1.6, ls="--", label="validation")
        best = g.loc[g.val_loss.idxmin()]
        ax.scatter([best.epoch], [best.val_loss], color=style.INK, zorder=3, s=18)
        ax.annotate(f"kept: epoch {int(best.epoch)}", (best.epoch, best.val_loss),
                    xytext=(6, 8), textcoords="offset points", fontsize=8.5)
        ax.set_title(model, loc="left", fontsize=10)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("MSE (z-scored demand)")
    axes[0].legend(frameon=False)
    style.caption(fig, f"{tank}: training and validation loss. Validation is the last 28 days "
                  "before the first backtest origin; the dot marks the weights that were kept.")
    style.save(fig, PLOTS, "M2_loss_curves")


def fig_forecasts(scored: pd.DataFrame, tank: str) -> None:
    import matplotlib.pyplot as plt
    s = scored[(scored.item_id == tank) & (scored.horizon == 24)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    # Reindex to every hour so a sensor gap draws as a break, not a straight line across it.
    actual = (s[s.model == "SeasonalNaive"].drop_duplicates("timestamp")
              .set_index("timestamp")["actual"].sort_index().asfreq("h"))
    ax.plot(actual.index, actual.to_numpy(), color=style.INK, lw=1.2, label="actual")
    for model in ["MLP", "Linear-AR", "SeasonalNaive", "Chronos2-ZS"]:
        g = s[s.model == model]
        for k, (_, blk) in enumerate(g.groupby("origin")):
            blk = blk.sort_values("timestamp")
            ax.plot(blk.timestamp, blk.pred, color=COLOR[model], lw=1.1, alpha=0.9,
                    label=LABEL.get(model, model) if k == 0 else None)
    ax.set_ylabel("outflow (KL / hour)")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper left")
    style.caption(fig, f"{tank}: every 24-hour forecast from the 24 backtest origins, laid end "
                  "to end. Origins are 23 h apart, so consecutive windows overlap by one hour.")
    style.save(fig, PLOTS, "M3_forecasts_h24")


# ────────────────────────────────────────────────────────────── report

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(lb, sig, pt, tank, manifest) -> None:
    board = lb.pivot(index="model", columns="horizon", values="macro_mase")
    board = board.sort_values(24).reset_index()
    board.columns = ["model"] + [f"MASE {h}h" for h in SCORED_HORIZONS]

    s = sig[["subset", "opponent", "horizon", "base_mase", "opponent_mase", "improvement_pct",
             "ci_lo_pct", "ci_hi_pct", "p_bootstrap", "p_dm", "significant_95", "origins_won"]]
    tiers = pt.groupby("trust")[["MLP", "Linear-AR", "SeasonalNaive", "Chronos2-ZS"]].mean()
    epochs = [v["MLP_best_epoch"] for v in manifest["tanks"].values()]
    epochs_lin = [v["Linear-AR_best_epoch"] for v in manifest["tanks"].values()]

    lines = [
        "# PyTorch MLP baseline — results",
        "",
        "Generated by `python -m src.models.mlp_comparison`. Every model is scored on the "
        "identical rows at 6, 12 and 24 h (enforced), with the benchmark's own MASE.",
        "",
        "## Leaderboard (macro MASE, lower is better)",
        "",
        _md_table(board),
        "",
        "## MLP against each opponent (paired over 24 origins)",
        "",
        "`improvement_pct` > 0 means the MLP has lower MASE. Significant = both the paired "
        "bootstrap and Diebold-Mariano (HLN) give p < 0.05.",
        "",
        _md_table(s),
        "",
        "## 24-hour MASE by sensor trust tier (mean over tanks)",
        "",
        _md_table(tiers.reset_index()),
        "",
        "## Per-tank wins at 24 h",
        "",
        f"- MLP beats SeasonalNaive on **{int(pt.mlp_beats_seasonal_naive.sum())}/24** tanks",
        f"- MLP beats Linear-AR on **{int(pt.mlp_beats_linear.sum())}/24** tanks",
        f"- MLP beats Chronos-2 zero-shot on **{int(pt.mlp_beats_chronos2.sum())}/24** tanks",
        "",
        "## Training",
        "",
        f"- Fit wall clock: {manifest['wall_clock_fit_s']} s for 24 tanks x 2 models on CPU "
        f"(torch {manifest['train']['torch']}).",
        f"- Best validation epoch, MLP: median {int(np.median(epochs))}, "
        f"{sum(e <= 3 for e in epochs)}/24 tanks at epoch 3 or earlier. "
        f"Linear-AR: median {int(np.median(epochs_lin))}.",
        f"- Test origins whose 168-hour input had a gap: "
        f"{manifest['test_origins_with_filled_input']} of {24 * 24} "
        f"({manifest['test_input_hours_filled']} hours forward-filled).",
        f"- Deep-dive tank (rule: healthy, highest mean demand): **{tank}** — see "
        "`plots/M2_loss_curves` and `plots/M3_forecasts_h24`.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


def main() -> None:
    scored = load_scored()
    print(f"{len(scored):,} scored rows | {scored.model.nunique()} models | horizons "
          f"{SCORED_HORIZONS} | row parity OK")
    lb = leaderboard(scored)
    sig = significance(scored)
    pt = per_tank_h24(scored)
    tank = deep_dive_tank()
    manifest = json.loads((OUT / "mlp_manifest.json").read_text())

    lb.to_csv(OUT / "leaderboard.csv", index=False)
    sig.to_csv(OUT / "significance.csv", index=False)
    pt.to_csv(OUT / "per_tank_h24.csv", index=False)
    fig_leaderboard(lb)
    fig_loss_curves(tank)
    fig_forecasts(scored, tank)
    write_report(lb, sig, pt, tank, manifest)

    show = lb[lb.horizon == 24][["model", "macro_mase", "macro_mae", "rank_mase"]]
    print("\n24 h leaderboard\n" + show.to_string(index=False))
    print("\n" + sig[["subset", "opponent", "horizon", "improvement_pct", "p_bootstrap", "p_dm",
                      "significant_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
