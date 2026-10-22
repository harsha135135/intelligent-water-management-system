"""Compare two studies of the same models on different evaluation windows.

The published study scores 24 origins in Mar–Apr 2026; a refresh scores the same protocol on a
later window. A model choice is only as good as its replication, so this asks, for every model on
both rosters: does its rank hold, and do the head-to-head tests reach the same verdict?

Levels are not compared directly. Each window's MASE is scaled by that window's own pre-origin
history, and a refresh with a longer (and possibly earlier-starting) panel has different
denominators. Ranks and paired tests are comparable across windows; raw MASE levels are shown
for reference only.

Reads ``unified/leaderboard.csv``, ``unified/significance_vs_all.csv`` and, where present,
``unified/significance_timestral.csv`` from each study. Writes ``window_comparison.csv``,
``window_pairs.csv`` and ``window_comparison.md`` into the second study's directory.

    python -m src.models.window_comparison --a results/chronos2 \\
        --b results/refresh_2026-10-22 --flag Chronos2-COV Chronos2-COV-LEAN ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HL = {6: "6 h", 12: "12 h", 24: "1 d", 48: "2 d", 72: "3 d", 168: "7 d"}
CHRONOS, TIMESTRAL, TIMESFM = "Chronos2-ZS", "Timestral-209M-ZS", "TimesFM2.5-ZS"
TIMESFM3 = "TimesFM3-ZS"
# (base, opponent, table) — the comparisons the deployment decision turns on.
PAIRS = [(CHRONOS, TIMESTRAL, "significance_vs_all"), (CHRONOS, TIMESFM, "significance_vs_all"),
         (CHRONOS, "NPTS", "significance_vs_all"), (CHRONOS, "Chronos2-COV-XL", "significance_vs_all"),
         (CHRONOS, "PatchTST-Tuned", "significance_vs_all"),
         (TIMESTRAL, TIMESFM, "significance_timestral"), (TIMESTRAL, "NPTS", "significance_timestral"),
         (CHRONOS, TIMESFM3, "significance_vs_all"), (TIMESTRAL, TIMESFM3, "significance_timestral"),
         (TIMESFM3, TIMESFM, "significance_timesfm3")]


def reading(row) -> str:
    """Which tests separate the pair, and in whose favour."""
    boot = bool(row.significant_95)
    dm = bool(pd.notna(row.p_dm) and row.p_dm < 0.05)
    if not boot and not dm:
        return "neither test"
    who = row.base if row.improvement_pct > 0 else row.opponent
    which = ("both tests" if boot and dm else "bootstrap only" if boot else "Diebold–Mariano only")
    return f"{which}, for {who}"


def pair_rows(results: Path, window: str) -> pd.DataFrame:
    rows = []
    for base, opp, table in PAIRS:
        p = results / "unified" / f"{table}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p)
        s = s[(s.base == base) & (s.opponent == opp) & (s.metric == "MASE")]
        for r in s.itertuples(index=False):
            rows.append({"window": window, "base": base, "opponent": opp,
                         "horizon": int(r.horizon), "horizon_label": HL[int(r.horizon)],
                         "improvement_pct": r.improvement_pct, "ci_lo_pct": r.ci_lo_pct,
                         "ci_hi_pct": r.ci_hi_pct, "p_bootstrap": r.p_bootstrap, "p_dm": r.p_dm,
                         "significant_95": r.significant_95, "reading": reading(r)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="first (earlier) study")
    ap.add_argument("--b", type=Path, required=True, help="second (later) study; outputs go here")
    ap.add_argument("--labels", nargs=2, default=["published window", "refresh window"])
    ap.add_argument("--flag", nargs="*", default=[],
                    help="models whose result in --b is qualified (e.g. calendar-incomplete)")
    ap.add_argument("--benchmark-only", nargs="*", default=[],
                    help="models scored for reference that may not be deployed (marked ‡)")
    ap.add_argument("--note", default="", help="a sentence printed under the title")
    args = ap.parse_args()
    la, lb_ = args.labels

    a = pd.read_csv(args.a / "unified" / "leaderboard.csv")
    b = pd.read_csv(args.b / "unified" / "leaderboard.csv")
    both = sorted(set(a.model) & set(b.model), key=list(b.model.unique()).index)
    m = (a[["model", "horizon", "macro_mase", "rank_mase"]]
         .merge(b[["model", "horizon", "macro_mase", "rank_mase"]], on=["model", "horizon"],
                suffixes=("_a", "_b")))
    m = m[m.model.isin(both)]
    m["flag_b"] = m.model.isin(args.flag)
    m.to_csv(args.b / "window_comparison.csv", index=False)

    mean_rank = m.groupby("model")[["rank_mase_a", "rank_mase_b"]].mean().loc[both]
    rho_all, p_all = spearmanr(mean_rank.rank_mase_a, mean_rank.rank_mase_b)
    rho_h = {h: spearmanr(g.rank_mase_a, g.rank_mase_b)[0] for h, g in m.groupby("horizon")}

    pairs = pd.concat([pair_rows(args.a, la), pair_rows(args.b, lb_)], ignore_index=True)
    pairs.to_csv(args.b / "window_pairs.csv", index=False)

    lines = [f"# Two-window comparison — {la} vs {lb_}", "",
             "Ranks and paired tests are compared across windows; raw MASE levels are not, "
             "because each window's MASE is scaled by its own pre-origin history.", "",
             *([args.note, ""] if args.note else []),
             f"Spearman rank correlation of the mean-rank leaderboards: **{rho_all:.3f}** "
             f"(p = {p_all:.3g}, {len(both)} models). Per horizon: "
             + ", ".join(f"{HL[h]} {r:.3f}" for h, r in sorted(rho_h.items())) + ".", "",
             f"| Model | mean rank, {la} | mean rank, {lb_} | MASE 1 d, {la} | "
             f"MASE 1 d, {lb_} |", "|---|---|---|---|---|"]
    d24 = m[m.horizon == 24].set_index("model")
    for model in mean_rank.sort_values("rank_mase_b").index:
        flag = (" †" if model in args.flag else "") + (" ‡" if model in args.benchmark_only else "")
        lines.append(f"| {model}{flag} | {mean_rank.loc[model, 'rank_mase_a']:.2f} | "
                     f"{mean_rank.loc[model, 'rank_mase_b']:.2f} | "
                     f"{d24.loc[model, 'macro_mase_a']:.4f} | {d24.loc[model, 'macro_mase_b']:.4f} |")
    if args.flag:
        lines += ["", f"† qualified in the {lb_}: {', '.join(args.flag)}."]
    if args.benchmark_only:
        lines += ["", f"‡ benchmark only, not deployable: {', '.join(args.benchmark_only)}."]
    lines += ["", "## Head-to-head on MASE, both windows", "",
              "`improvement` is the base's MASE relative to the opponent's, from the base's "
              "side: positive = the base has the lower error.", "",
              "| Base | Opponent | Horizon | " + " | ".join(f"{w}: improvement · tests"
                                                         for w in (la, lb_)) + " |",
              "|---|---|---|---|---|"]
    for (base, opp), g in pairs.groupby(["base", "opponent"], sort=False):
        for h in sorted(g.horizon.unique()):
            cells = []
            for w in (la, lb_):
                r = g[(g.window == w) & (g.horizon == h)]
                cells.append(f"{r.improvement_pct.iloc[0]:+.2f} % · {r.reading.iloc[0]}"
                             if len(r) else "—")
            lines.append(f"| {base} | {opp} | {HL[h]} | " + " | ".join(cells) + " |")
    (args.b / "window_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
